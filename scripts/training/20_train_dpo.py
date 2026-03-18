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
import sys
from pathlib import Path

import torch

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
        "Unsloth n'est pas installé. Installer avec :\n"
        "  uv pip install unsloth",
        file=sys.stderr,
    )
    sys.exit(1)

import mlflow
import pandas as pd
from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast, set_seed
from trl import DPOConfig, DPOTrainer

from utils import format_dpo_prompt, format_dpo_response, get_latest_checkpoint, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
SFT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dpo"

DPO_TRAIN_PATH = PROJECT_ROOT / "data" / "final" / "dpo_train.parquet"
DPO_VAL_PATH = PROJECT_ROOT / "data" / "final" / "dpo_val.parquet"

MAX_SEQ_LENGTH = 1024
BETA = 0.1        # pénalité KL — force de l'alignement
LEARNING_RATE = 5e-5
EPOCHS = 2
# DPO charge chosen + rejected en parallèle → mémoire doublée → batch_size réduit
BATCH_SIZE = 2
GRAD_ACCUM = 8    # effective batch = 16, identique au SFT
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
SEED = 42

MLFLOW_EXPERIMENT = "dpo-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")


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
    model = model.merge_and_unload()

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
    train_path: Path,
    val_path: Path,
) -> tuple[Dataset, Dataset]:
    """Charge et formate les Parquets DPO pour le DPOTrainer.

    Applique format_dpo_prompt() sur 'prompt' et format_dpo_response()
    sur 'chosen' et 'rejected'. Le DPOTrainer tokenise ces chaînes
    directement sans ré-appliquer de chat template.

    Args:
        train_path: Chemin vers dpo_train.parquet.
        val_path: Chemin vers dpo_val.parquet.

    Returns:
        Tuple (train_dataset, val_dataset) avec colonnes
        {prompt, chosen, rejected} formatées en ChatML.
    """

    def _format(df: pd.DataFrame) -> Dataset:
        records = [
            {
                "prompt": format_dpo_prompt(row["prompt"]),
                "chosen": format_dpo_response(row["chosen"]),
                "rejected": format_dpo_response(row["rejected"]),
            }
            for _, row in df.iterrows()
        ]
        return Dataset.from_list(records)

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    return _format(df_train), _format(df_val)


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
        loss_type="sigmoid",           # DPO classique (Rafailov et al. 2023)
        max_length=MAX_SEQ_LENGTH,
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
        trainable / 1e6, total / 1e9, trainable / total * 100,
    )


def main() -> None:
    """Pipeline DPO complet.

    Idempotent : skip si checkpoints/dpo/adapter_model.safetensors existe.
    Reprend depuis le dernier checkpoint intermédiaire si disponible.
    """
    parser = argparse.ArgumentParser(description="Entraînement DPO")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument("--beta", type=float, default=BETA, help="Pénalité KL DPO")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Nombre d'epochs")
    args = parser.parse_args()

    logger = get_logger("20_train_dpo", verbose=args.verbose)
    set_seed(SEED)

    # Idempotence
    adapter_path = DPO_CHECKPOINT / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info("Adaptateur DPO déjà présent : %s — skip.", adapter_path)
        return

    # Vérifications préalables
    if not (SFT_CHECKPOINT / "adapter_model.safetensors").exists():
        logger.error("Checkpoint SFT non trouvé : %s. Lancer 11_train_sft.py d'abord.", SFT_CHECKPOINT)
        sys.exit(1)

    for path in [DPO_TRAIN_PATH, DPO_VAL_PATH]:
        if not path.exists():
            logger.error("Fichier manquant : %s. Lancer le pipeline S1 d'abord.", path)
            sys.exit(1)

    # Resume depuis un checkpoint intermédiaire ?
    resume_path = get_latest_checkpoint(DPO_CHECKPOINT)
    if resume_path:
        logger.info("Checkpoint intermédiaire trouvé : %s — reprise de l'entraînement.", resume_path)

    # Chargement et formatage des datasets
    logger.info("Chargement des datasets DPO...")
    train_dataset, val_dataset = load_dpo_datasets(DPO_TRAIN_PATH, DPO_VAL_PATH)
    logger.info("  train: %d paires | val: %d paires", len(train_dataset), len(val_dataset))

    # Chargement du modèle (base + SFT merged + DPO LoRA)
    logger.info("Chargement du modèle : base + SFT merged + DPO LoRA (r=%d, beta=%.2f)...", LORA_R, args.beta)
    model, tokenizer = load_sft_merged_with_dpo_lora(MODEL_NAME, SFT_CHECKPOINT, MAX_SEQ_LENGTH)
    log_model_info(model, logger)

    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Le start_run explicite garantit que le callback HuggingFace utilise ce run
    # pour les métriques de steps (loss curves) au lieu de créer un nested run.
    with mlflow.start_run(run_name="dpo-qwen3-1.7b-triage"):
        # Configuration DPO
        config = build_dpo_config(DPO_CHECKPOINT)
        config.beta = args.beta
        config.num_train_epochs = args.epochs

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
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "sft_checkpoint": str(SFT_CHECKPOINT),
            "beta": args.beta,
            "learning_rate": LEARNING_RATE,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "train_pairs": len(train_dataset),
            "val_pairs": len(val_dataset),
        })

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
        logger.info("Adaptateur LoRA DPO sauvegardé dans %s", DPO_CHECKPOINT)

        # Log des artefacts MLflow — poids LoRA finaux (fichiers top-level uniquement,
        # on exclut les sous-dossiers checkpoint-N/ qui sont des sauvegardes intermédiaires)
        for f in sorted(DPO_CHECKPOINT.iterdir()):
            if f.is_file():
                mlflow.log_artifact(str(f), artifact_path="adapter")

    logger.info("=== Entraînement DPO terminé. ===")


if __name__ == "__main__":
    main()
