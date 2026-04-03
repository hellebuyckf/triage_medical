"""Script 03b — Generate hard-negative DPO pairs from SFT misclassifications.

## Motivation

Standard synthetic DPO pairs (script 03) have a fundamental flaw: the
``chosen`` and ``rejected`` responses share the *same clinical body* — only
the urgency label differs.  Because the two texts are nearly identical, the
DPO gradient concentrates on 2-3 tokens (the label) and provides almost no
signal for the rest of the generation.  In practice this caused a ~10%
accuracy regression vs the SFT baseline across three experiments.

Hard negatives fix this by using the SFT model's *actual* wrong predictions
as ``rejected`` responses.  The flawed reasoning is baked into the full
response, giving DPO a real corrective signal at every token.

## How it works

1. **Load** the fine-tuned SFT model (base + LoRA adapter from checkpoints/sft).
2. **Infer** on the *train* split (4 544 examples) in batches of 8.
   Qwen3 chain-of-thought is suppressed by pre-filling ``<think>\\n\\n</think>``
   so the urgency label appears immediately (required by the parser).
3. **Filter** to keep only misclassified examples where:
   - the generated response has a parseable urgency label, AND
   - that label differs from the ground-truth label.
4. **Build** one DPO pair per misclassified example:
   - ``chosen``  = ground-truth response from ``data/final/sft/train``
   - ``rejected`` = the SFT's actual wrong generation (wrong label + wrong reasoning)
5. **Save** to ``data/processed/dpo_hard_negatives`` as a HuggingFace Dataset.

## Key numbers (2026-03-25 run)

- Train set : 4 544 examples
- Misclassified : 1 674 (36.8 % SFT error rate)
- Chosen label distribution  : moderate 987 / max 350 / deferred 337
- Rejected label distribution: max 813 / deferred 705 / moderate 156
  → SFT mainly over-predicts ``max`` for moderate cases, and
    under-predicts ``max`` (predicts ``deferred``) for true max cases.

## Integration

This script is called by ``make sft-errors`` (after ``train-sft``).
Script 03 then merges these hard negatives with ~480 synthetic pairs
(``make rebuild-dpo``).  The combined dataset (~2 154 pairs) is used for
DPO training via ``make dpo-pipeline-hard``.

Results after merging hard negatives (DPO v4 vs SFT on test set):
  Accuracy  : 63.80 % vs 63.80 % (regression eliminated, was -10 % before)
  F1 Macro  : 0.633 vs 0.624   (+0.009)
  F2 Macro  : 0.635 vs 0.629   (+0.005)
  Recall max: 75 % vs ~65 %    (+10 pp, 139/186 correct vs 63-69)
  max→deferred errors: 31 vs 115-121  (-75 %)

## Output schema

``data/processed/dpo_hard_negatives`` is a HuggingFace Dataset with columns:
  prompt   : str  — original user instruction
  chosen   : str  — ground-truth triage response (correct urgency label)
  rejected : str  — SFT-generated response (wrong urgency label + reasoning)
  source   : str  — always "sft_hard_negative"
  language : str  — "en" or "fr"

Prerequisites:
  - ``data/final/sft`` must exist  (run ``make data-pipeline`` first)
  - ``checkpoints/sft`` must exist (run ``make train-sft`` first)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

if torch.cuda.is_available(): torch.backends.cuda.preferred_blas_library("cublaslt")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from utils import DPO_COLUMNS, SYSTEM_PROMPT, extract_urgency_from_response, get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"
SFT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "sft"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "dpo_hard_negatives"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
SEED = 42
DPO_SOURCE = "sft_hard_negative"


# ── Model loading & inference ──────────────────────────────────────────────────


def load_sft_model(
    model_name: str,
    checkpoint_dir: Path,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
    """Load the base model and apply SFT LoRA weights.

    Args:
        model_name: HuggingFace model identifier for the base model.
        checkpoint_dir: Directory containing the LoRA adapter weights.

    Returns:
        Tuple of (model in eval mode, tokenizer with left padding).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.eval()
    model.generation_config.max_length = None  # type: ignore[reportAttributeAccessIssue]
    return model, tokenizer  # type: ignore[reportReturnType]


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    instructions: list[str],
) -> list[str]:
    """Generate triage responses for a list of instructions using batched inference.

    Suppresses Qwen3 chain-of-thought by pre-filling an empty <think> block.

    Args:
        model: Fine-tuned SFT model in eval mode.
        tokenizer: Tokenizer with left padding.
        instructions: List of user instruction strings.

    Returns:
        List of decoded response strings (empty string on OOM).
    """
    im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")  # type: ignore[assignment]
    responses: list[str] = [""] * len(instructions)

    for start in tqdm(range(0, len(instructions), BATCH_SIZE), desc="SFT inference"):
        batch = instructions[start : start + BATCH_SIZE]
        prompts = [
            str(
                tokenizer.apply_chat_template(  # type: ignore[union-attr]
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instr},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            + "<think>\n\n</think>\n"
            for instr in batch
        ]
        inputs = tokenizer(  # type: ignore[call-overload]
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)  # type: ignore[union-attr]
        input_length: int = inputs["input_ids"].shape[1]  # type: ignore[index]

        try:
            with torch.no_grad():
                output_ids = model.generate(  # type: ignore[reportCallIssue]
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=im_end_id,
                )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                torch.cuda.empty_cache()
                continue
            raise

        for i in range(len(batch)):
            gen_ids = output_ids[i][input_length:]
            eos_pos = (gen_ids == im_end_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_ids = gen_ids[: eos_pos[0]]
            responses[start + i] = str(tokenizer.decode(gen_ids, skip_special_tokens=True))

    return responses


# ── Hard negative construction ─────────────────────────────────────────────────


def build_hard_negatives(
    df: pd.DataFrame,
    responses: list[str],
) -> list[dict]:
    """Build DPO hard-negative pairs from SFT misclassifications.

    A hard negative is a pair where:
    - chosen  = ground-truth response (correct urgency label)
    - rejected = SFT-generated response (wrong urgency label + flawed reasoning)

    Both responses must have a parseable urgency label for the pair to be kept.

    Args:
        df: DataFrame with columns instruction, response, urgency_level, language.
        responses: SFT-generated responses, aligned with df rows.

    Returns:
        List of DPO pair dicts with keys from DPO_COLUMNS.
    """
    pairs: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        generated = responses[i]
        if not generated:
            continue

        predicted = extract_urgency_from_response(generated)
        true_label = row["urgency_level"]

        if predicted is None or predicted == true_label:
            continue  # parseable only if wrong prediction

        pairs.append(
            {
                "prompt": row["instruction"],
                "chosen": row["response"],
                "rejected": generated,
                "source": DPO_SOURCE,
                "language": row.get("language", "en"),
            }
        )

    return pairs


# ── Pipeline ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Run SFT inference on train split and save hard-negative DPO pairs."""
    logger = get_logger("03b_sft_errors")

    if OUTPUT_PATH.exists():
        ds = Dataset.load_from_disk(str(OUTPUT_PATH))
        logger.info(
            "Hard negatives already built at {} ({} pairs). Skipping.", OUTPUT_PATH, len(ds)
        )
        return

    adapter_path = CHECKPOINT_DIR / "adapter_model.safetensors"
    if not adapter_path.exists():
        logger.error("SFT checkpoint not found at {}. Run 11_train_sft.py first.", adapter_path)
        sys.exit(1)

    if not SFT_FINAL_DIR.exists():
        logger.error("SFT final dataset not found at {}. Run data-pipeline first.", SFT_FINAL_DIR)
        sys.exit(1)

    logger.info("Loading SFT train split...")
    sft = DatasetDict(load_from_disk(str(SFT_FINAL_DIR)))  # type: ignore[arg-type]
    urgency_feature = sft["train"].features["urgency_level"]
    df_train = pd.DataFrame(sft["train"].to_pandas())
    df_train["urgency_level"] = df_train["urgency_level"].map(urgency_feature.int2str)
    logger.info("Train split: {} examples.", len(df_train))

    logger.info("Loading SFT model from {}...", CHECKPOINT_DIR)
    model, tokenizer = load_sft_model(MODEL_NAME, CHECKPOINT_DIR)
    logger.info("Model loaded.")

    logger.info("Running inference on train split...")
    responses = generate_responses(model, tokenizer, list(df_train["instruction"]))

    logger.info("Building hard-negative pairs from misclassifications...")
    pairs = build_hard_negatives(df_train, responses)
    logger.info(
        "  {} / {} examples misclassified → {} hard-negative pairs.",
        len(pairs),
        len(df_train),
        len(pairs),
    )

    if not pairs:
        logger.warning("No misclassified examples found. Skipping save.")
        return

    ds = Dataset.from_list(pairs).select_columns(DPO_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_PATH))
    logger.info("Hard negatives saved to {}.", OUTPUT_PATH)

    from collections import Counter

    labels = [extract_urgency_from_response(p["chosen"]) for p in pairs]
    logger.info("Chosen label distribution: {}", dict(Counter(labels)))
    rejected_labels = [extract_urgency_from_response(p["rejected"]) for p in pairs]
    logger.info("Rejected label distribution: {}", dict(Counter(rejected_labels)))


if __name__ == "__main__":
    main()
