"""Script 11 — Entraînement SFT de Qwen3-1.7B avec LoRA via Unsloth + TRL."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

# Workaround : Unsloth patche Qwen3 avec des forward produisant des tenseurs non-contigus.
# cuBLAS standard crash sur ces tenseurs, cublasLt les gère correctement.
torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:
    print(
        "Unsloth n'est pas installé. Installer avec :\n"
        "  uv pip install unsloth\n"
        "Pour CUDA 12.9 : voir https://github.com/unslothai/unsloth#installation",
        file=sys.stderr,
    )
    sys.exit(1)

import mlflow
import mlflow.transformers
from datasets import load_from_disk
from dotenv import load_dotenv
from transformers import PreTrainedModel, PreTrainedTokenizerFast, set_seed
from trl import SFTConfig, SFTTrainer
from utils import get_latest_checkpoint, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "sft_tokenized"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"

MAX_SEQ_LENGTH = 1024
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LEARNING_RATE = 2e-4
EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 4
SEED = 42

MLFLOW_EXPERIMENT = "sft-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
REGISTERED_MODEL_NAME = "sft-qwen3-1.7b-triage"


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


# ── Fonctions ─────────────────────────────────────────────────────────────────


@mlflow.trace(span_type="RETRIEVER", name="load_model_and_tokenizer")
def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle via Unsloth et applique la configuration LoRA.

    Args:
        model_name: Identifiant du modèle sur HuggingFace Hub.
        max_seq_length: Longueur maximale de séquence.
        load_in_4bit: Quantification 4-bit si GPU < 16 GB.

    Returns:
        Tuple (modèle avec LoRA, tokenizer).
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

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


def build_sft_config(output_dir: Path) -> SFTConfig:
    """Construit la configuration SFT pour TRL 0.29.

    Args:
        output_dir: Répertoire de sortie pour les checkpoints.

    Returns:
        Configuration SFT complète.
    """
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="mlflow",
        run_name="sft-qwen3-1.7b-triage",
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        # TRL 0.29 native prompt/completion masking: loss is computed only on
        # completion tokens when the dataset has "prompt" and "completion" columns.
        completion_only_loss=True,
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    train_dataset,
    val_dataset,
    config: SFTConfig,
) -> SFTTrainer:
    """Instancie le SFTTrainer avec masquage de loss sur le prompt.

    TRL 0.29 handles prompt masking natively via ``completion_only_loss=True``
    in ``SFTConfig`` when the dataset exposes ``prompt`` and ``completion`` columns.
    No custom data collator is needed.

    Args:
        model: Modèle avec adaptateur LoRA.
        tokenizer: Tokenizer configuré.
        train_dataset: Dataset with ``prompt`` and ``completion`` columns.
        val_dataset: Dataset de validation.
        config: Configuration SFT.

    Returns:
        Trainer prêt à l'entraînement.
    """
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )


def log_model_info(model: PreTrainedModel, logger) -> None:
    """Affiche le nombre de paramètres total vs entraînables (LoRA).

    Args:
        model: Modèle avec adaptateur LoRA.
        logger: Logger pour l'affichage.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / total * 100
    logger.info(
        "Paramètres entraînables : %.2fM / %.2fB (%.2f%%)",
        trainable / 1e6,
        total / 1e9,
        pct,
    )


def setup_mlflow() -> None:
    """Configure MLflow tracking URI et experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def main() -> None:
    """Pipeline d'entraînement SFT avec LoRA.

    Idempotent : skip si adapter_model.safetensors existe.
    Reprend depuis le dernier checkpoint intermédiaire si disponible.
    """
    parser = argparse.ArgumentParser(description="Entraînement SFT LoRA")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sft.yaml",
        help="Chemin vers le fichier de config YAML (défaut: configs/sft.yaml)",
    )
    args = parser.parse_args()

    cfg = load_training_config(args.config)
    t = cfg["training"]
    lora = cfg["lora"]

    global MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
    global LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_ACCUM, SEED

    MAX_SEQ_LENGTH = t["max_seq_length"]
    LORA_R = lora["r"]
    LORA_ALPHA = lora["alpha"]
    LORA_DROPOUT = lora["dropout"]
    LORA_TARGET_MODULES = lora["target_modules"]
    LEARNING_RATE = t["learning_rate"]
    EPOCHS = t["epochs"]
    BATCH_SIZE = t["batch_size"]
    GRAD_ACCUM = t["grad_accum"]
    SEED = t["seed"]

    logger = get_logger("11_train_sft", verbose=args.verbose)
    set_seed(SEED)

    # Idempotence : vérifier si l'adaptateur final existe
    adapter_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info("Adaptateur LoRA déjà présent : %s — skip.", adapter_path)
        return

    # Vérifier les données tokenisées
    if not (TOKENIZED_DIR / "train").exists():
        logger.error(
            "Datasets tokenisés non trouvés dans %s. Lancer 10_prepare_tokenizer.py d'abord.",
            TOKENIZED_DIR,
        )
        sys.exit(1)

    # Resume depuis un checkpoint intermédiaire ?
    resume_path = get_latest_checkpoint(CHECKPOINT_DIR)
    if resume_path:
        logger.info(
            "Checkpoint intermédiaire trouvé : %s — reprise de l'entraînement.", resume_path
        )

    # Chargement des datasets
    logger.info("Chargement des datasets tokenisés...")
    train_dataset = load_from_disk(str(TOKENIZED_DIR / "train"))
    val_dataset = load_from_disk(str(TOKENIZED_DIR / "val"))
    logger.info("  train: %d | val: %d", len(train_dataset), len(val_dataset))

    # Pré-concaténation prompt+completion en colonne "text"
    train_dataset = train_dataset.map(lambda x: {"text": x["prompt"] + x["completion"]})
    val_dataset = val_dataset.map(lambda x: {"text": x["prompt"] + x["completion"]})

    # Chargement du modèle + LoRA
    logger.info(
        "Chargement du modèle %s + LoRA (r=%d, alpha=%d)...", MODEL_NAME, LORA_R, LORA_ALPHA
    )
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, MAX_SEQ_LENGTH)
    log_model_info(model, logger)

    # Configuration MLflow
    setup_mlflow()

    # System metrics : CPU, RAM, GPU utilization + VRAM (nécessite pynvml)
    mlflow.enable_system_metrics_logging()

    # Le start_run explicite garantit que le callback HuggingFace utilise ce run
    # pour les métriques de steps (loss curves) au lieu de créer un nested run.
    with mlflow.start_run(run_name="sft-qwen3-1.7b-triage"):
        # Configuration SFT + Trainer
        config = build_sft_config(CHECKPOINT_DIR)
        trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, config)
        logger.info("Loss masking enabled: gradient computed on completion tokens only.")

        # Log des hyperparamètres
        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation": GRAD_ACCUM,
                "max_seq_length": MAX_SEQ_LENGTH,
                "train_examples": len(train_dataset),
                "val_examples": len(val_dataset),
            }
        )

        # Entraînement
        logger.info("Lancement de l'entraînement...")
        try:
            trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(
                    "OOM CUDA ! Suggestions :\n"
                    "  1. Réduire BATCH_SIZE à 2 et augmenter GRAD_ACCUM à 8\n"
                    "  2. Réduire MAX_SEQ_LENGTH à 512"
                )
            raise

        # Sauvegarde de l'adaptateur LoRA + tokenizer
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(CHECKPOINT_DIR))
        tokenizer.save_pretrained(str(CHECKPOINT_DIR))
        logger.info("Adaptateur LoRA sauvegardé dans %s", CHECKPOINT_DIR)

        # Sauvegarde des hyperparamètres d'entraînement pour les rapports d'évaluation
        training_config = {
            "model_name": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "max_seq_length": MAX_SEQ_LENGTH,
            "seed": SEED,
        }
        (CHECKPOINT_DIR / "training_config.json").write_text(
            json.dumps(training_config, indent=2), encoding="utf-8"
        )
        logger.info("Hyperparamètres sauvegardés dans %s/training_config.json", CHECKPOINT_DIR)

        # Log the PEFT model natively — mlflow.transformers handles PeftModel
        # serialisation, tokenizer packaging, and Model Registry registration
        # without any custom wrapper class.
        # pip_requirements is set explicitly to bypass mlflow's auto-detection,
        # which tries to import tensorflow (not installed) and crashes.
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="adapter",
            task="text-generation",
            registered_model_name=REGISTERED_MODEL_NAME,
            metadata={"base_model": MODEL_NAME, "lora_r": LORA_R, "stage": "sft"},
            pip_requirements=["transformers", "torch", "peft", "accelerate"],
        )

    logger.info("=== Entraînement SFT terminé. ===")


if __name__ == "__main__":
    main()
