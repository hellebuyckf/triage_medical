"""Script 22 — Fusion des poids LoRA DPO et export au format HuggingFace.

Stratégie :
- Charge le modèle de base Qwen3-1.7B.
- Applique le LoRA SFT et fusionne (merge_and_unload) → modèle SFT dense.
- Applique le LoRA DPO et fusionne (merge_and_unload via PEFT standard).
- Sauvegarde le modèle complet dans checkpoints/dpo_merged/ (format HuggingFace).
- Vérifie l'export avec une inférence test via transformers standard (sans Unsloth)
  pour valider la compatibilité vLLM.
"""

import argparse
import os
import re
import sys
from pathlib import Path

_ARTIFACT_RE = re.compile(
    r"(ForCanBeConverted|𫟦|\uFFFD+|\n\s*(?:user|assistant)\s*\n.*)",
    re.DOTALL | re.IGNORECASE,
)

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
        "Unsloth n'est pas installé. Installer avec :\n  uv pip install unsloth",
        file=sys.stderr,
    )
    sys.exit(1)

from dotenv import load_dotenv
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast
from utils import SYSTEM_PROMPT, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# ── Constantes ────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
SFT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sft"
DPO_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "dpo"
EXPORT_DIR = PROJECT_ROOT / "checkpoints" / "dpo_merged"

MAX_SEQ_LENGTH = 1024
PUSH_TO_HUB = False
HF_REPO_ID = "your-username/qwen3-1.7b-triage-medical"  # si PUSH_TO_HUB=True

# Prompt de vérification post-export
VERIFY_PROMPT = "Quels sont les symptômes d'un infarctus du myocarde ?"


# ── Fonctions ─────────────────────────────────────────────────────────────────


def merge_lora_weights(
    model_name: str,
    sft_checkpoint: Path,
    dpo_checkpoint: Path,
    max_seq_length: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Charge le modèle de base sur CPU, fusionne SFT puis DPO LoRA.

    La fusion est effectuée entièrement en RAM CPU (32 GB+) pour éviter
    tout crash CUDA OOM. La VRAM (16 GB typique) est insuffisante pour
    maintenir le modèle de base + les deux LoRA + les tenseurs intermédiaires
    de merge_and_unload() en parallèle.

    Flux de fusion en deux étapes :
    1. base (CPU) → PeftModel(SFT) → merge_and_unload() → modèle SFT dense (CPU).
    2. SFT dense (CPU) → PeftModel(DPO) → merge_and_unload() → modèle DPO dense (CPU).

    Le modèle résultant est sauvegardé sur disque ; la vérification post-export
    recharge depuis le disque avec device_map="auto" → GPU.

    Args:
        model_name: Identifiant HuggingFace du modèle de base.
        sft_checkpoint: Répertoire du checkpoint LoRA SFT.
        dpo_checkpoint: Répertoire du checkpoint LoRA DPO.
        max_seq_length: Longueur max des séquences.

    Returns:
        Tuple (modèle fusionné SFT+DPO sur CPU, tokenizer).
    """
    # Étape 1 : base model sur CPU — évite tout risque d'OOM VRAM pendant la fusion
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        device_map="cpu",
    )

    # Étape 2 : fusionner les poids SFT (opération RAM)
    model = PeftModel.from_pretrained(base_model, str(sft_checkpoint), device_map="cpu")
    model = model.merge_and_unload()  # type: ignore[reportCallIssue]

    # Étape 3 : appliquer et fusionner les poids DPO (sur le modèle SFT fusionné)
    model = PeftModel.from_pretrained(model, str(dpo_checkpoint), device_map="cpu")
    model = model.merge_and_unload()  # type: ignore[reportCallIssue]

    return model, tokenizer  # type: ignore[reportReturnType]


def save_merged_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    export_dir: Path,
    push_to_hub: bool = False,
    repo_id: str = "",
) -> None:
    """Sauvegarde le modèle fusionné au format HuggingFace (safetensors bf16).

    Le modèle reçu est déjà un modèle dense (SFT + DPO fusionnés via
    merge_lora_weights). On le sauvegarde directement au format standard
    HuggingFace compatible vLLM/llama.cpp.

    Fichiers générés dans export_dir/ :
    - config.json, generation_config.json
    - model.safetensors (ou shards pour modèles > 5 GB)
    - tokenizer.json, tokenizer_config.json, special_tokens_map.json

    Args:
        model: Modèle dense SFT+DPO fusionné (sorti de merge_lora_weights).
        tokenizer: Tokenizer Qwen3.
        export_dir: Répertoire de sortie local.
        push_to_hub: Si True, upload vers HuggingFace Hub après sauvegarde locale.
        repo_id: Identifiant du repo Hub (ex: "username/model-name").
    """
    export_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde standard HuggingFace — compatible vLLM et llama.cpp
    model.save_pretrained(str(export_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(export_dir))

    if push_to_hub and repo_id:
        model.push_to_hub(repo_id, safe_serialization=True)  # type: ignore[reportArgumentType]
        tokenizer.push_to_hub(repo_id)


def verify_export(
    export_dir: Path,
    test_prompt: str = VERIFY_PROMPT,
    logger=None,
) -> None:
    """Charge le modèle exporté via FastLanguageModel et génère une réponse test.

    Utilise FastLanguageModel.from_pretrained car Unsloth patche les classes
    transformers globalement à l'import (Qwen3Attention, etc.) — AutoModelForCausalLM
    reçoit ces patches sans que apply_qkv soit initialisé, causant un AttributeError.
    FastLanguageModel initialise correctement tous les attributs Unsloth.

    Log un WARNING si 'URGENCE' n'est pas présent dans la réponse générée.

    Args:
        export_dir: Répertoire du modèle exporté.
        test_prompt: Instruction médicale de test.
        logger: Logger pour les messages. Si None, utilise print.
    """
    log = logger.info if logger else print
    warn = logger.warning if logger else print

    log("Vérification de l'export (chargement via FastLanguageModel)...")

    # FastLanguageModel est requis : Unsloth patche Qwen3Attention globalement
    # à l'import ; AutoModelForCausalLM reçoit le forward patché mais sans apply_qkv.
    # for_inference() est intentionnellement omis : il active un chemin KV-cache
    # spécialisé (fast_forward_inference) incompatible avec les shapes d'un modèle
    # fusionné rechargé depuis le disque (shape mismatch dans les rotary embeddings).
    # model.eval() + torch.no_grad() est suffisant pour la vérification.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(export_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model.eval()

    # Formatage du prompt en ChatML via apply_chat_template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_prompt},
    ]
    chatml_prompt: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chatml_prompt, return_tensors="pt").to(model.device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids: list[int] = [im_end_id]
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != im_end_id:
        stop_ids.append(tokenizer.eos_token_id)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Décoder uniquement les nouveaux tokens, tronquer au premier stop token
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][input_len:]
    for stop_id in stop_ids:
        positions = (generated_ids == stop_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            generated_ids = generated_ids[: positions[0]]
            break

    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response = _ARTIFACT_RE.sub("", raw).strip()

    log("Prompt test : %s", test_prompt)
    log("Réponse générée :\n%s", response)

    if "URGENCE" not in response.upper():
        warn(
            "WARNING : 'URGENCE' absent de la réponse générée. "
            "Le modèle pourrait ne pas respecter le format de triage. "
            "Vérifier le modèle avant déploiement."
        )
    else:
        log("Format de triage validé : 'URGENCE' présent dans la réponse.")

    # Libérer la mémoire GPU
    del model
    torch.cuda.empty_cache()


def print_export_summary(export_dir: Path, logger) -> None:
    """Affiche la liste et la taille totale des fichiers exportés.

    Args:
        export_dir: Répertoire du modèle exporté.
        logger: Logger pour les messages.
    """
    files = sorted(export_dir.iterdir())
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    total_gb = total_bytes / 1e9

    logger.info("Fichiers exportés dans %s :", export_dir)
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            logger.info("  %-50s %8.1f MB", f.name, size_mb)
    logger.info("Taille totale : %.2f GB", total_gb)


def main() -> None:
    """Pipeline d'export DPO complet.

    Idempotent : skip si checkpoints/dpo_merged/config.json existe déjà.
    """
    parser = argparse.ArgumentParser(description="Export modèle DPO fusionné")
    parser.add_argument("--verbose", action="store_true", help="Logging DEBUG")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Passer la vérification d'inférence post-export",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload vers HuggingFace Hub après export local",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=HF_REPO_ID,
        help="Identifiant du repo HuggingFace Hub",
    )
    args = parser.parse_args()

    logger = get_logger("22_export_model", verbose=args.verbose)

    # Idempotence : skip la fusion si le modèle exporté est déjà présent,
    # SAUF si --push-to-hub est demandé (on recharge depuis l'export local).
    merged_config = EXPORT_DIR / "config.json"
    if merged_config.exists() and not args.push_to_hub:
        logger.info("Modèle fusionné déjà présent : %s — skip.", EXPORT_DIR)
        return

    if merged_config.exists() and args.push_to_hub:
        logger.info(
            "Modèle fusionné déjà présent dans %s — push vers HF Hub sans re-fusionner.",
            EXPORT_DIR,
        )
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(EXPORT_DIR),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        model.push_to_hub(args.repo_id, safe_serialization=True)
        tokenizer.push_to_hub(args.repo_id)
        logger.info("Modèle poussé sur HF Hub : %s", args.repo_id)
        return

    # Vérifications préalables
    if not (SFT_CHECKPOINT / "adapter_model.safetensors").exists():
        logger.error(
            "Checkpoint SFT non trouvé : %s. Lancer 11_train_sft.py d'abord.",
            SFT_CHECKPOINT,
        )
        sys.exit(1)

    if not (DPO_CHECKPOINT / "adapter_model.safetensors").exists():
        logger.error(
            "Checkpoint DPO non trouvé : %s. Lancer 20_train_dpo.py d'abord.",
            DPO_CHECKPOINT,
        )
        sys.exit(1)

    # Chargement et fusion des poids LoRA (SFT puis DPO)
    logger.info("Chargement du modèle de base et fusion des LoRA (SFT + DPO)...")
    model, tokenizer = merge_lora_weights(
        MODEL_NAME, SFT_CHECKPOINT, DPO_CHECKPOINT, MAX_SEQ_LENGTH
    )

    # Sauvegarde via Unsloth (merged_16bit)
    logger.info("Sauvegarde du modèle fusionné dans %s ...", EXPORT_DIR)
    save_merged_model(
        model,
        tokenizer,
        EXPORT_DIR,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
    )
    logger.info("Export terminé.")

    # Récapitulatif des fichiers
    print_export_summary(EXPORT_DIR, logger)

    # Vérification de l'export
    if not args.skip_verify:
        verify_export(EXPORT_DIR, VERIFY_PROMPT, logger)

    logger.info("=== Export DPO terminé. Modèle prêt dans %s ===", EXPORT_DIR)


if __name__ == "__main__":
    main()
