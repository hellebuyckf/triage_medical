# Project 14 — Medical Triage AI Agent

> **OpenClassrooms AI Engineering — Week 1–3: Data + SFT + DPO**

A proof-of-concept (POC) for an AI-powered medical triage agent for Centre Hospitalier Saint-Aurélien (CHSA).
The agent fine-tunes **Qwen3-1.7B-Base** to classify medical emergencies (`max` / `moderate` / `deferred`) from symptom descriptions and return structured recommendations.

**Approach:** SFT → DPO → vLLM deployment | **Duration:** 4 weeks

---

## Project Structure

```
project14/
├── CLAUDE.md                         # Project-specific AI assistant instructions
├── pyproject.toml                    # Dependencies (uv)
├── Makefile                          # Pipeline orchestration
├── configs/
│   ├── datasets.yaml                 # HuggingFace dataset IDs and configs
│   ├── sft.yaml                      # SFT hyperparameters (LoRA, training)
│   └── dpo.yaml                      # DPO hyperparameters (LoRA, training, beta)
├── data/
│   ├── raw/                          # Git-ignored — raw HuggingFace datasets
│   ├── processed/                    # Git-ignored — intermediate files
│   └── final/                        # Deliverable — final Parquet files
├── scripts/
│   ├── utils.py                      # Shared module (logger, urgency inference, validators)
│   ├── data_prep/
│   │   ├── 01_download.py            # Step 1: Download HuggingFace datasets
│   │   ├── 02_build_sft.py           # Step 2: Build SFT dataset
│   │   ├── 03_build_dpo.py           # Step 3: Build DPO dataset
│   │   ├── 04_anonymize.py           # Step 4: GDPR anonymization (Presidio)
│   │   ├── 05_split_and_validate.py  # Step 5: Train/val/test split + validation
│   │   └── 06_push_to_hub.py         # Step 6: Push datasets to HuggingFace Hub
│   └── training/
│       ├── 10_prepare_tokenizer.py   # Tokenize + format ChatML, save Arrow datasets
│       ├── 11_train_sft.py           # SFT LoRA training (Unsloth + TRL SFTTrainer)
│       ├── 12_evaluate_sft.py        # SFT evaluation — accuracy / F1 / eval report
│       ├── 20_train_dpo.py           # DPO alignment (TRL DPOTrainer, ref_model=None)
│       ├── 21_evaluate_dpo.py        # DPO evaluation — SFT vs DPO comparison
│       └── 22_export_model.py        # Merge SFT+DPO LoRA → dense HuggingFace model
├── checkpoints/                      # Git-ignored — LoRA adapters and merged model
│   ├── sft/
│   ├── dpo/
│   └── dpo_merged/
├── reports/
│   ├── sft/                          # Timestamped SFT eval reports (immune to make clean-sft)
│   └── dpo/                          # Timestamped DPO eval reports (immune to make clean-dpo)
└── specs/                            # Project specifications (private submodule)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Python | 3.11+ |
| Package Manager | `uv` |
| Base Model | Qwen3-1.7B-Base |
| Fine-tuning | PEFT (LoRA) + Unsloth |
| Alignment | DPO via TRL |
| GDPR Anonymization | Presidio + spaCy (`fr_core_news_md`, `en_core_web_md`) |
| Data Format | Parquet (pyarrow) |
| Experiment Tracking | MLflow / Weights & Biases |
| Serving | vLLM + FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## Setup

**Prerequisites:** Python 3.11+, [`uv`](https://docs.astral.sh/uv/) installed.

```bash
# Clone the repository
git clone <repo-url>
cd project14

# Install dependencies and spaCy language models
make setup
```

This installs all Python dependencies and downloads the required spaCy models (`fr_core_news_md`, `en_core_web_md`).

---

## Usage

### Full retrain (no data rebuild)

```bash
make retrain        # clean SFT+DPO checkpoints → SFT pipeline → DPO pipeline
```

### Pipeline by stage

```bash
# Stage 1 — Data
make data-pipeline  # download → build-sft → build-dpo → anonymize → split

# Stage 2 — SFT
make sft-pipeline   # prepare-tokenizer → train-sft → evaluate-sft

# Stage 3 — DPO
make dpo-pipeline   # train-dpo → evaluate-dpo → export-model
```

### Individual steps

```bash
make download             # Download datasets from HuggingFace
make build-sft            # Build SFT dataset (~6 500 instruction/response pairs)
make build-dpo            # Build DPO dataset (chosen/rejected pairs)
make anonymize            # GDPR anonymization + report
make split                # Train/val/test split + validation report
make prepare-tokenizer    # Tokenize + format ChatML, save Arrow datasets
make train-sft            # SFT LoRA training
make evaluate-sft         # SFT evaluation → reports/sft/eval_report_<timestamp>.md
make train-dpo            # DPO alignment
make evaluate-dpo         # DPO evaluation → reports/dpo/eval_report_<timestamp>.md
make export-model         # Merge SFT+DPO LoRA → checkpoints/dpo_merged/
```

All scripts are **idempotent**: re-running skips already completed steps.

### Training config (YAML)

Hyperparameters live in `configs/sft.yaml` and `configs/dpo.yaml`. Edit them directly, then retrain without touching Python code:

```bash
# Default — loads configs/sft.yaml automatically
make train-sft

# Custom config
make train-sft SFT_CONFIG=configs/sft_fast.yaml

# CLI override for quick experiments (bypasses YAML for that param only)
uv run python scripts/training/20_train_dpo.py --beta 0.05
```

### Evaluation options

```bash
make evaluate-sft EVAL_VAL=1    # Also evaluate on val set (biased — model was selected on val loss)
make evaluate-dpo EVAL_VAL=1
```

### Push to HuggingFace Hub

```bash
make push-datasets HF_USERNAME=<user>         # Publish SFT + DPO final datasets
make push-model HF_USERNAME=<user>            # Push merged model (username/qwen3-triage-dpo)
make push-datasets HF_USERNAME=<user> HF_PRIVATE=1  # Private repos
```

### Cleanup

```bash
make clean          # Remove data/raw/ and data/processed/
make clean-sft      # Remove checkpoints/sft/ and sft_tokenized/ (keeps reports/)
make clean-dpo      # Remove checkpoints/dpo/ and dpo_merged/ (keeps reports/)
make clean-all      # Remove all data/ and checkpoints/
```

---

## Data Sources

| Dataset | HuggingFace ID | Usage | Language | Size |
|---|---|---|---|---|
| FrenchMedMCQA | `nthngdy/frenchmedmcqa` | SFT | FR | ~3 105 |
| MedQuAD | `keivalya/MedQuad-MedicalQnADataset` | SFT | EN | ~16 407 |
| MediQAl (mcqu) | `ANR-MALADES/MediQAl` (config: `mcqu`) | SFT | FR | ~17 017 |
| MediQAl (oeq) | `ANR-MALADES/MediQAl` (config: `oeq`) | SFT | FR | ~4 969 |
| UltraMedical-Preference | `TsinghuaC3I/UltraMedical-Preference` | DPO | EN | 100K+ |

---

## Output Schema

### SFT — `data/final/sft_train.parquet`, `sft_val.parquet`, `sft_test.parquet`

| Column | Type | Description |
|---|---|---|
| `instruction` | `str` | Patient symptoms / question |
| `response` | `str` | Clinical evaluation + recommendations |
| `source` | `str` | Source dataset name |
| `language` | `str` | `"fr"` or `"en"` |
| `urgency_level` | `str` | `"max"`, `"moderate"`, or `"deferred"` |
| `confidence` | `float` | `0.7` (inferred) or `1.0` (annotated) |

### DPO — `data/final/dpo_train.parquet`, `dpo_val.parquet`

| Column | Type | Description |
|---|---|---|
| `prompt` | `str` | Instruction / question |
| `chosen` | `str` | Clinically preferred response |
| `rejected` | `str` | Non-preferred response |
| `source` | `str` | `"ultramedical_preference"` |
| `language` | `str` | `"en"` |

---

## Urgency Level Inference

The pipeline infers urgency from keyword matching:

- **`max`** (confidence 0.8): critical keywords (chest pain, stroke, cardiac arrest, dyspnée, AVC…)
- **`deferred`** (confidence 0.8): deferred keywords (cold, appointment, can wait, rhume…)
- **`moderate`** (confidence 0.7): all other cases

Target class distribution: **~33% per urgency level** (undersampled if imbalanced, `seed=42`).

---

## Training Hyperparameters

All tunable hyperparameters live in `configs/`. No Python code changes needed to experiment.

### `configs/sft.yaml`

```yaml
training:
  max_seq_length: 1024
  learning_rate: 0.0002
  epochs: 5
  batch_size: 4
  grad_accum: 4    # effective batch = 16
  seed: 42

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
```

### `configs/dpo.yaml`

Same structure, plus `beta: 0.1` (KL penalty strength) under `training:`.

---

## Deliverables by Week

| Week | Deliverable | Status |
|---|---|---|
| **S1** | `data/final/` — 5 Parquet files + `rgpd_report.md` + `stats_report.md` | ✅ Done |
| **S2** | SFT checkpoint (`checkpoints/sft/`) + MLflow run + `reports/sft/eval_report_*.md` | ✅ Done |
| **S3** | DPO checkpoint + merged model (`checkpoints/dpo_merged/`) + `reports/dpo/eval_report_*.md` | 🔄 In progress |
| **S4** | Cloud `/triage` endpoint + CI/CD pipeline + technical report (≤ 20 pages) | ⏳ Upcoming |

---

## Medical & Legal Constraints

- This agent performs **triage**, not **diagnosis** — all outputs must include disclaimers.
- Strict GDPR anonymization via Presidio before any training.
- All interactions must be logged for medical audit.
- **Never commit identifiable patient data.**

---

## Development Notes

- `random_state=42` and `seed=42` everywhere for reproducibility.
- Logging via Python's `logging` module (INFO by default, DEBUG with `--verbose`).
- All file paths use `pathlib.Path` (no hardcoded strings).
- All scripts are idempotent: re-running skips already completed work.
- `cublasLt` workaround in all training scripts: Unsloth patches Qwen3 forward passes producing non-contiguous tensors incompatible with standard cuBLAS.
- Eval reports are timestamped (`reports/sft/eval_report_YYYYMMDD_HHMMSS.md`) and never overwritten by `make clean-sft` / `make clean-dpo`.
