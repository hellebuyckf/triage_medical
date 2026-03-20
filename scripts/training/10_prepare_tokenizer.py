"""Script 10 — Tokenizer preparation and prompt/completion formatting for SFT."""

import argparse
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

load_dotenv(dotenv_path=_SCRIPTS_DIR.parent / ".env", override=False)

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from utils import get_logger

PROJECT_ROOT = _SCRIPTS_DIR.parent

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-1.7B")
MAX_SEQ_LENGTH = 1024

# Number of CPU cores for .map() on pure-Python transforms.
# Keep num_proc=1 for tokenization: HF fast tokenizers use Rust parallelism
# internally — combining with fork(num_proc>1) causes deadlocks.
NUM_PROC = 4

SFT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "sft"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "processed" / "sft_tokenized"


# ── Batch transform functions ──────────────────────────────────────────────────
# Module-level functions are picklable, which is required for num_proc > 1.


def _format_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, list]:
    """Format a batch of SFT rows into prompt/completion pairs.

    Uses ``tokenizer.apply_chat_template`` as the single source of truth for
    special tokens (BOS, EOS, role markers). The prompt boundary is derived by
    applying the template twice:

    1. ``[user]`` + ``add_generation_prompt=True``  → exact prompt text
    2. ``[user, assistant]``                         → full text

    completion = full_text[len(prompt_text):]  — no fragile string search.

    Args:
        batch: Dict of lists with keys ``instruction``, ``response``,
            and ``urgency_level``, as provided by ``Dataset.map(batched=True)``.
        tokenizer: Loaded tokenizer whose chat template drives the formatting.

    Returns:
        Dict of lists with keys ``prompt``, ``completion``, ``urgency_level``.
    """
    prompts: list[str] = []
    completions: list[str] = []
    for instruction, response in zip(batch["instruction"], batch["response"], strict=True):
        prompt_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            [{"role": "user", "content": str(instruction)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(response)},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt_text)
        completions.append(full_text[len(prompt_text) :])
    return {
        "prompt": prompts,
        "completion": completions,
        "urgency_level": batch["urgency_level"],
    }


def _compute_length_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, list]:
    """Compute token length for each example in a batch.

    Uses ``apply_chat_template`` to build the exact string the model will see,
    then tokenizes without truncation to measure real lengths.

    Called with ``num_proc=1`` to avoid deadlocks between HF fast tokenizer
    (Rust parallelism) and Python multiprocessing (fork).

    Args:
        batch: Dict of lists with keys ``instruction`` and ``response``.
        tokenizer: Loaded tokenizer; truncation is disabled to measure real lengths.

    Returns:
        Dict with a single key ``_length`` containing token counts per example.
    """
    lengths: list[int] = []
    for instruction, response in zip(batch["instruction"], batch["response"], strict=True):
        full_text: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
            [
                {"role": "user", "content": str(instruction)},
                {"role": "assistant", "content": str(response)},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(full_text, truncation=False)["input_ids"]  # type: ignore[call-overload]
        lengths.append(len(tokens))  # type: ignore[arg-type]
    return {"_length": lengths}


# ── Analysis ──────────────────────────────────────────────────────────────────


def analyze_lengths(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> dict[str, int | float]:
    """Compute token-length distribution statistics for a dataset split.

    Adds a temporary ``_length`` column via ``Dataset.map(batched=True)`` then
    reads it column-by-column — no DataFrame allocation.

    Args:
        ds: Dataset split with ``instruction`` and ``response`` columns.
        tokenizer: Loaded tokenizer.
        max_seq_length: Threshold above which examples are counted as truncated.

    Returns:
        Dict with p50, p75, p90, p95, p99, max, mean, n_truncated, pct_truncated.
    """
    ds_with_len = ds.map(
        partial(_compute_length_batch, tokenizer=tokenizer),
        batched=True,
        batch_size=256,
        num_proc=1,  # fast tokenizer: Rust parallelism — no fork
        desc="Computing token lengths",
    )
    arr = np.array(ds_with_len["_length"])

    n_truncated = int((arr > max_seq_length).sum())
    return {
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "n_truncated": n_truncated,
        "pct_truncated": round(n_truncated / len(arr) * 100, 2),
    }


# ── Formatting ────────────────────────────────────────────────────────────────


def format_to_prompt_completion(ds: Dataset, tokenizer: PreTrainedTokenizerFast) -> Dataset:
    """Convert an SFT Dataset split to prompt/completion format.

    The prompt/completion format is native to TRL 0.29 SFTTrainer and
    automatically enables loss masking on the prompt portion.

    Args:
        ds: Dataset split with ``instruction``, ``response``, and
            ``urgency_level`` columns.
        tokenizer: Loaded tokenizer whose chat template drives the formatting.

    Returns:
        Dataset with ``prompt``, ``completion``, and ``urgency_level`` columns.
    """
    return ds.map(
        partial(_format_batch, tokenizer=tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=[c for c in ds.column_names if c not in ("urgency_level",)],
        desc="Formatting prompt/completion",
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Preparation pipeline: length analysis + prompt/completion formatting.

    Idempotent: skips if TOKENIZED_DIR/train already exists.
    """
    parser = argparse.ArgumentParser(description="Tokenizer preparation + SFT formatting")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logger = get_logger("10_prepare_tokenizer", verbose=args.verbose)

    # Idempotence
    if (TOKENIZED_DIR / "train").exists():
        logger.info("Tokenized datasets already present in %s — skip.", TOKENIZED_DIR)
        return

    if not SFT_FINAL_DIR.exists():
        logger.error("Missing dataset: %s. Run the S1 pipeline first.", SFT_FINAL_DIR)
        sys.exit(1)

    # Load tokenizer
    logger.info("Loading tokenizer from %s...", MODEL_NAME)
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Qwen3-Base has no chat_template in its tokenizer_config.json (base ≠ instruct).
    # apply_chat_template raises ValueError if the attribute is absent, so we inject
    # the standard Qwen3 ChatML template that all instruct variants use.
    if not tokenizer.chat_template:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        logger.info(
            "chat_template not found on base model — injected standard Qwen3 ChatML template."
        )

    logger.info(
        "Tokenizer loaded. Vocab size: %d, pad_token: '%s'",
        tokenizer.vocab_size,
        tokenizer.pad_token,
    )

    # Load splits directly as Dataset — no pandas conversion
    logger.info("Loading splits from %s...", SFT_FINAL_DIR)
    sft: DatasetDict = DatasetDict(load_from_disk(str(SFT_FINAL_DIR)))  # type: ignore[arg-type]
    ds_train: Dataset = sft["train"]
    ds_val: Dataset = sft["val"]
    ds_test: Dataset = sft["test"]
    logger.info("  train: %d | val: %d | test: %d", len(ds_train), len(ds_val), len(ds_test))

    # Token-length analysis on train set
    logger.info("Analysing token lengths (train set)...")
    stats = analyze_lengths(ds_train, tokenizer, MAX_SEQ_LENGTH)
    logger.info("Length distribution:")
    for key in ["p50", "p75", "p90", "p95", "p99", "max", "mean"]:
        logger.info("  %s: %s", key, stats[key])

    if stats["n_truncated"] > 0:
        logger.warning(
            "%d examples (%.1f%%) exceed MAX_SEQ_LENGTH=%d and will be truncated.",
            stats["n_truncated"],
            stats["pct_truncated"],
            MAX_SEQ_LENGTH,
        )
    else:
        logger.info("No examples exceed MAX_SEQ_LENGTH=%d.", MAX_SEQ_LENGTH)

    if stats["p95"] <= 512:
        logger.info("Recommendation: MAX_SEQ_LENGTH=512 is sufficient (p95=%d).", stats["p95"])
    elif stats["p95"] <= 1024:
        logger.info("Recommendation: MAX_SEQ_LENGTH=1024 is appropriate (p95=%d).", stats["p95"])
    else:
        logger.info("Recommendation: MAX_SEQ_LENGTH=2048 required (p95=%d).", stats["p95"])

    # Format to prompt/completion
    logger.info("Formatting splits to prompt/completion...")
    splits = {
        "train": format_to_prompt_completion(ds_train, tokenizer),
        "val": format_to_prompt_completion(ds_val, tokenizer),
        "test": format_to_prompt_completion(ds_test, tokenizer),
    }

    # Save
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    for name, ds in splits.items():
        out_path = TOKENIZED_DIR / name
        ds.save_to_disk(str(out_path))
        logger.info("  %s: %d examples saved to %s", name, len(ds), out_path)

    logger.info("=== Preparation complete. ===")


if __name__ == "__main__":
    main()
