"""Script 11 — Entraînement SFT de Qwen3-1.7B avec LoRA via HuggingFace + PEFT + TRL.

Workarounds appliqués :
- cublasLt forcé : PyTorch 2.10 cu128 vs CUDA système 12.9 — cuBLAS standard échoue
  sur toute opération half-precision (fp16/bf16). cublasLt utilise un code path compatible.
- Unsloth bypassé : ses patches Qwen3 (`original_apply_qkv`) produisent des tenseurs
  non-contigus, incompatibles même avec cublasLt. HF standard + PEFT est ~2x plus lent
  mais stable. Pour un POC de 4660 exemples, l'impact est acceptable.
"""

import argparse
import sys
from pathlib import Path

import torch

# Workaround : PyTorch cu128 vs CUDA 12.9 — cuBLAS standard crash sur fp16/bf16 GEMM.
# cublasLt utilise un code path différent qui fonctionne avec le driver 575+.
torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import mlflow
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from utils import get_latest_checkpoint, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = "unsloth/Qwen3-1.7B-Base"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "sft_tokenized"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"

MAX_SEQ_LENGTH = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE = 2e-4
EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
SEED = 42

MLFLOW_EXPERIMENT = "sft-qwen3-1.7b-triage"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")


# ── Fonctions ─────────────────────────────────────────────────────────────────


def load_model_and_tokenizer(
    model_name: str,
    max_seq_length: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle via HuggingFace standard et applique LoRA via PEFT.

    Utilise AutoModelForCausalLM (pas Unsloth FastLanguageModel) pour éviter
    les patches Qwen3 incompatibles avec bf16 + PEFT LoRA.

    Args:
        model_name: Identifiant du modèle sur HuggingFace Hub.
        max_seq_length: Longueur maximale de séquence (pour le tokenizer).

    Returns:
        Tuple (modèle avec LoRA, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

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
        fp16=False,
        bf16=True,
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
        dataset_text_field="text",
    )


def build_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    train_dataset,
    val_dataset,
    config: SFTConfig,
) -> SFTTrainer:
    """Instancie le SFTTrainer.

    Args:
        model: Modèle avec adaptateur LoRA.
        tokenizer: Tokenizer configuré.
        train_dataset: Dataset d'entraînement (colonne text = ChatML complet).
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
        trainable / 1e6, total / 1e9, pct,
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
    args = parser.parse_args()

    logger = get_logger("11_train_sft", verbose=args.verbose)
    set_seed(SEED)

    # Idempotence : vérifier si l'adaptateur final existe
    adapter_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if adapter_path.exists():
        logger.info("Adaptateur LoRA déjà présent : %s — skip.", adapter_path)
        return

    # Vérifier les données tokenisées
    if not (TOKENIZED_DIR / "train").exists():
        logger.error("Datasets tokenisés non trouvés dans %s. Lancer 10_prepare_tokenizer.py d'abord.", TOKENIZED_DIR)
        sys.exit(1)

    # Resume depuis un checkpoint intermédiaire ?
    resume_path = get_latest_checkpoint(CHECKPOINT_DIR)
    if resume_path:
        logger.info("Checkpoint intermédiaire trouvé : %s — reprise de l'entraînement.", resume_path)

    # Chargement des datasets
    logger.info("Chargement des datasets tokenisés...")
    train_dataset = load_from_disk(str(TOKENIZED_DIR / "train"))
    val_dataset = load_from_disk(str(TOKENIZED_DIR / "val"))
    logger.info("  train: %d | val: %d", len(train_dataset), len(val_dataset))

    # Pré-concaténation prompt+completion en colonne "text"
    train_dataset = train_dataset.map(lambda x: {"text": x["prompt"] + x["completion"]})
    val_dataset = val_dataset.map(lambda x: {"text": x["prompt"] + x["completion"]})

    # Chargement du modèle + LoRA (HF standard, pas Unsloth)
    logger.info("Chargement du modèle %s + LoRA (r=%d, alpha=%d)...", MODEL_NAME, LORA_R, LORA_ALPHA)
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, MAX_SEQ_LENGTH)
    log_model_info(model, logger)

    # Configuration MLflow
    setup_mlflow()

    # Configuration SFT + Trainer
    config = build_sft_config(CHECKPOINT_DIR)
    trainer = build_trainer(model, tokenizer, train_dataset, val_dataset, config)

    # Note : la loss est calculée sur toute la séquence (prompt + réponse).
    # Impact pour un POC : léger surapprentissage du format prompt, acceptable à 4660 exemples.
    logger.info("Entraînement sur séquence complète (pas de loss masking).")

    # Log des hyperparamètres
    mlflow.log_params({
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
    })

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

    # Log des artefacts MLflow
    adapter_config = CHECKPOINT_DIR / "adapter_config.json"
    if adapter_config.exists():
        mlflow.log_artifact(str(adapter_config))

    logger.info("=== Entraînement SFT terminé. ===")


if __name__ == "__main__":
    main()
