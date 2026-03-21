"""Script 20 — Alignement DPO de Qwen3-1.7B+SFT via TRL DPOTrainer.

Stratégie :
- Charge le modèle de base Qwen3-1.7B.
- Fusionne les poids LoRA SFT (merge_and_unload) → modèle dense SFT.
- Ajoute un nouveau LoRA DPO entraînable sur le modèle fusionné.
- Passe ref_model=None : TRL désactive l'adaptateur DPO pour les passes
  de référence → le modèle fusionné SFT (sans LoRA DPO) sert de référence.
  C'est la référence correcte pour l'alignement DPO.
"""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import torch
import yaml

# Workaround Unsloth/Qwen3 : forward patches produisent des tenseurs non-contigus.
# cublasLt gère correctement ce cas, cuBLAS standard crash.
torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from unsloth import FastLanguageModel
except ImportError:
    print(
        "Unsloth n'est pas installé. Installer avec :\n  uv pip install unsloth",
        file=sys.stderr,
    )
    sys.exit(1)

import mlflow
import mlflow.transformers
from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast, set_seed
from trl import DPOConfig, DPOTrainer
from utils import SYSTEM_PROMPT, get_latest_checkpoint, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
SFT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dpo"

DPO_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "dpo"

MAX_SEQ_LENGTH = 1024
BETA = 0.1  # pénalité KL — force de l'alignement
LEARNING_RATE = 5e-5
EPOCHS = 2
# DPO charge chosen + rejected en parallèle → mémoire doublée → batch_size réduit
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch = 16, identique au SFT
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
SEED = 42

MLFLOW_EXPERIMENT = "dpo-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "dpo-qwen3-1.7b-triage"


# ── Config loader ──────────────────────────────────────────────────────────────


def load_training_config(config_path: Path) -> dict:
    """Load hyperparameters from a YAML config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dict with keys 'training' and 'lora'.
    """
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as f:
        return yaml.safe_load(f)


# ── Batch transform functions ──────────────────────────────────────────────────
# Defined at module level for picklability (required when num_proc > 1).


def _format_dpo_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, list]:
    """Formate un batch DPO via apply_chat_template.

    Double appel au template (même pattern que script 10) :
    1. ``[system, user]`` + ``add_generation_prompt=True``  → prompt_text
    2. ``[system, user, assistant]``                         → full_text
    completion = full_text[len(prompt_text):]

    Args:
        batch: Dict de listes avec clés ``prompt``, ``chosen``, ``rejected``.
        tokenizer: Tokenizer Qwen3 dont le chat template pilote le formatage.

    Returns:
        Dict de listes avec les mêmes clés, formatées en ChatML.
    """
    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []
    for prompt, chosen, rejected in zip(
        batch["prompt"], batch["chosen"], batch["rejected"], strict=True
    ):
        system_user = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(prompt)},
        ]
        prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            system_user,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_chosen: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            system_user + [{"role": "assistant", "content": str(chosen)}],
            tokenize=False,
            add_generation_prompt=False,
        )
        full_rejected: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            system_user + [{"role": "assistant", "content": str(rejected)}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt_text)
        chosens.append(full_chosen[len(prompt_text) :])
        rejecteds.append(full_rejected[len(prompt_text) :])
    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


# ── Fonctions ─────────────────────────────────────────────────────────────────


@mlflow.trace(span_type="RETRIEVER", name="load_sft_merged_with_dpo_lora")
def load_sft_merged_with_dpo_lora(
    model_name: str,
    sft_checkpoint: Path,
    max_seq_length: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle de base, fusionne le LoRA SFT, ajoute un LoRA DPO entraînable.

    Étapes :
    1. FastLanguageModel charge le modèle de base Qwen3.
    2. PeftModel.from_pretrained applique les poids LoRA SFT (depuis checkpoints/sft/).
    3. merge_and_unload() fusionne le LoRA SFT dans les poids denses → modèle SFT dense.
    4. FastLanguageModel.get_peft_model ajoute un nouveau LoRA DPO (entraînable).

    Avec ref_model=None dans DPOTrainer, TRL désactive l'adaptateur DPO pour les
    passes de référence → utilise le modèle SFT fusionné comme référence. ✓

    Args:
        model_name: Identifiant HuggingFace du modèle de base.
        sft_checkpoint: Répertoire du checkpoint LoRA SFT.
        max_seq_length: Longueur max des séquences.

    Returns:
        Tuple (modèle avec LoRA DPO entraînable, tokenizer).
    """
    # Étape 1 : base model
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    # Étape 2 : appliquer les poids SFT
    model = PeftModel.from_pretrained(base_model, str(sft_checkpoint))

    # Étape 3 : fusionner SFT dans les poids denses
    model = model.merge_and_unload()  # type: ignore[reportCallIssue]

    # Étape 4 : ajouter un LoRA DPO frais et entraînable
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    return model, tokenizer


@mlflow.trace(span_type="RETRIEVER", name="load_dpo_datasets")
def load_dpo_datasets(
    dpo_dir: Path,
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[Dataset, Dataset]:
    """Charge et formate le DatasetDict DPO pour le DPOTrainer.

    Utilise ``_format_dpo_batch`` avec ``Dataset.map(batched=True)`` pour
    exploiter Apache Arrow sans round-trip pandas. Même pattern que script 10.

    Args:
        dpo_dir: Répertoire contenant le DatasetDict HF (splits train/val).
        tokenizer: Tokenizer Qwen3 dont le chat template pilote le formatage.

    Returns:
        Tuple (train_dataset, val_dataset) avec colonnes
        {prompt, chosen, rejected} formatées via apply_chat_template.
    """
    dpo = DatasetDict(load_from_disk(str(dpo_dir)))  # type: ignore[arg-type]
    _format_fn = partial(_format_dpo_batch, tokenizer=tokenizer)
    train_dataset = dpo["train"].map(_format_fn, batched=True)
    val_dataset = dpo["val"].map(_format_fn, batched=True)
    return train_dataset, val_dataset


def build_dpo_config(output_dir: Path) -> DPOConfig:
    """Construit la DPOConfig TRL.

    Args:
        output_dir: Répertoire de sortie pour les checkpoints.

    Returns:
        Instance DPOConfig configurée pour l'entraînement DPO.
    """
    return DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        beta=BETA,
        loss_type="sigmoid",  # DPO classique (Rafailov et al. 2023)  # type: ignore[reportArgumentType]
        max_length=MAX_SEQ_LENGTH,
        # NOTE : max_prompt_length et max_completion_length ne sont pas supportés
        # par la version patchée de DPOConfig dans Unsloth (UnslothDPOTrainer.py).
        # La protection contre la troncature des réponses est assurée partiellement
        # par max_length=1024 qui plafonne la séquence totale prompt+réponse.
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="mlflow",
        run_name="dpo-qwen3-1.7b-triage",
        seed=SEED,
    )


def log_model_info(model: PreTrainedModel, logger) -> None:
    """Affiche le ratio de paramètres entraînables (LoRA DPO) vs total.

    Args:
        model: Modèle avec adaptateur LoRA DPO.
        logger: Logger pour l'affichage.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Paramètres entraînables : %.2fM / %.2fB (%.3f%%)",
        trainable / 1e6,
        total / 1e9,
        trainable / total * 100,
    )


def main() -> None:
    """Pipeline DPO complet.

    Idempotent : skip si checkpoints/dpo/adapter_model.safetensors existe.
    Reprend depuis le dernier checkpoint intermédiaire si disponible.
    """
    parser = argparse.ArgumentParser(description="Entraînement DPO")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dpo.yaml",
        help="Chemin vers le fichier de config YAML (défaut: configs/dpo.yaml)",
    )
    parser.add_argument("--beta", type=float, default=None, help="Override pénalité KL DPO")
    parser.add_argument("--epochs", type=int, default=None, help="Override nombre d'epochs")
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    t = cfg["training"]
    lora = cfg["lora"]

    global MAX_SEQ_LENGTH, BETA, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
    global LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_ACCUM, SEED

    MAX_SEQ_LENGTH = t["max_seq_length"]
    BETA = t["beta"]
    LEARNING_RATE = t["learning_rate"]
    EPOCHS = t["epochs"]
    BATCH_SIZE = t["batch_size"]
    GRAD_ACCUM = t["grad_accum"]
    SEED = t["seed"]
    LORA_R = lora["r"]
    LORA_ALPHA = lora["alpha"]
    LORA_DROPOUT = lora["dropout"]
    LORA_TARGET_MODULES = lora["target_modules"]

    # CLI overrides (optionnels — écrasent le YAML si fournis)
    if args.beta is not None:
        BETA = args.beta
    if args.epochs is not None:
        EPOCHS = args.epochs

    logger = get_logger("20_train_dpo", verbose=args.verbose)
    set_seed(SEED)

    # Idempotence
    adapter_path = DPO_CHECKPOINT / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info("Adaptateur DPO déjà présent : {} — skip.", adapter_path)
        return

    # Vérifications préalables
    if not (SFT_CHECKPOINT / "adapter_model.safetensors").exists():
        logger.error(
            "Checkpoint SFT non trouvé : %s. Lancer 11_train_sft.py d'abord.", SFT_CHECKPOINT
        )
        sys.exit(1)

    if not DPO_FINAL_DIR.exists():
        logger.error("Dataset manquant : {}. Lancer le pipeline S1 d'abord.", DPO_FINAL_DIR)
        sys.exit(1)

    # Resume depuis un checkpoint intermédiaire ?
    resume_path = get_latest_checkpoint(DPO_CHECKPOINT)
    if resume_path:
        logger.info(
            "Checkpoint intermédiaire trouvé : %s — reprise de l'entraînement.", resume_path
        )

    # Chargement du modèle (base + SFT merged + DPO LoRA)
    # Le tokenizer est nécessaire pour formater les datasets via apply_chat_template.
    logger.info(
        "Chargement du modèle : base + SFT merged + DPO LoRA (r={}, beta={:.2f})...",
        LORA_R,
        BETA,
    )
    model, tokenizer = load_sft_merged_with_dpo_lora(MODEL_NAME, SFT_CHECKPOINT, MAX_SEQ_LENGTH)

    # Chargement et formatage des datasets (nécessite le tokenizer pour apply_chat_template)
    logger.info("Chargement des datasets DPO...")
    train_dataset, val_dataset = load_dpo_datasets(DPO_FINAL_DIR, tokenizer)
    logger.info("  train: {} paires | val: {} paires", len(train_dataset), len(val_dataset))
    log_model_info(model, logger)

    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # System metrics : CPU, RAM, GPU utilization + VRAM (nécessite pynvml)
    mlflow.enable_system_metrics_logging()

    # Le start_run explicite garantit que le callback HuggingFace utilise ce run
    # pour les métriques de steps (loss curves) au lieu de créer un nested run.
    with mlflow.start_run(run_name="dpo-qwen3-1.7b-triage"):
        # Configuration DPO
        config = build_dpo_config(DPO_CHECKPOINT)

        # DPOTrainer
        # ref_model=None : TRL utilise le modèle SFT fusionné (sans adaptateur DPO) comme référence.
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

        # Log hyperparamètres
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "sft_checkpoint": str(SFT_CHECKPOINT),
                "beta": BETA,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": GRAD_ACCUM,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "train_pairs": len(train_dataset),
                "val_pairs": len(val_dataset),
            }
        )

        # Entraînement
        logger.info("Lancement de l'entraînement DPO...")
        try:
            trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(
                    "OOM CUDA ! Suggestions :\n"
                    "  1. Réduire BATCH_SIZE à 1 et augmenter GRAD_ACCUM à 16\n"
                    "  2. Réduire max_length à 512 dans DPOConfig"
                )
            raise

        # Sauvegarde
        DPO_CHECKPOINT.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(DPO_CHECKPOINT))
        tokenizer.save_pretrained(str(DPO_CHECKPOINT))
        logger.info("Adaptateur LoRA DPO sauvegardé dans {}", DPO_CHECKPOINT)

        # pip_requirements is set explicitly to bypass mlflow's auto-detection,
        # which tries to import tensorflow (not installed) and crashes.
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="adapter",
            task="text-generation",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={"base_model": MODEL_NAME, "lora_r": LORA_R, "beta": BETA, "stage": "dpo"},
            pip_requirements=["transformers", "torch", "peft", "accelerate"],
        )

    logger.info("=== Entraînement DPO terminé. ===")


if __name__ == "__main__":
    main()
