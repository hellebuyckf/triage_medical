# README — Pipeline de Données (project14)

Ce document décrit la structure des datasets, le pipeline de préparation, et la procédure
pour publier les datasets sur HuggingFace Hub.

---

## Structure des dossiers

```
data/
├── raw/                              # HF Arrow (inchangé — généré par 01_download.py)
│   ├── frenchmedmcqa/                #   DatasetDict {train, validation, test}
│   ├── medquad/                      #   DatasetDict {train}
│   ├── mediql_mcqu/                  #   DatasetDict {train, validation, test}
│   ├── mediql_oeq/                   #   DatasetDict {test}
│   └── ultramedical_preference/      #   DatasetDict {train, validation, test}
│
├── processed/                        # HF Arrow (générés par 02–04)
│   ├── sft_raw/                      #   Dataset  — 6 498 lignes
│   ├── dpo_raw/                      #   Dataset  — 1 000 lignes
│   ├── sft_anonymized/               #   Dataset  — 6 498 lignes
│   ├── dpo_anonymized/               #   Dataset  — 1 000 lignes
│   ├── sft_tokenized/                #   Dataset  — formaté prompt/completion (généré par 10)
│   └── rgpd_report.md
│
└── final/                            # HF Arrow — DatasetDict (générés par 05)
    ├── sft/                          #   DatasetDict {train: 4 957, val: 620, test: 620}
    │   └── dataset_dict.json
    ├── dpo/                          #   DatasetDict {train: 900, val: 100}
    │   └── dataset_dict.json
    ├── stats_report.md
    └── rgpd_report.md
```

> **Pourquoi HuggingFace Arrow ?**
> Le format Arrow (`.save_to_disk()` / `load_from_disk()`) est le standard natif de l'écosystème HF.
> Il permet l'intégration directe avec `Dataset.map()`, le streaming, et `push_to_hub()`,
> sans conversion intermédiaire — contrairement au Parquet qui nécessite un `.read_parquet()` + `Dataset.from_pandas()`.

---

## Pipeline en 5 étapes

```
01_download.py
  └─► data/raw/  (Arrow IPC)

02_build_sft.py  +  03_build_dpo.py
  └─► data/processed/sft_raw/  +  dpo_raw/  (Dataset HF)

04_anonymize.py
  └─► data/processed/sft_anonymized/  +  dpo_anonymized/  (Dataset HF)

05_split_and_validate.py
  └─► data/final/sft/  +  dpo/  (DatasetDict HF)
```

| Script | Entrée | Sortie | Transformation clé |
|--------|--------|--------|-------------------|
| `01_download.py` | HuggingFace Hub | `data/raw/` | Téléchargement + cache local |
| `02_build_sft.py` | 4 raw datasets | `processed/sft_raw/` (6 498 lignes) | MCQ/QA → `instruction/response` + `infer_urgency` |
| `03_build_dpo.py` | ultramedical raw | `processed/dpo_raw/` (1 000 lignes) | Filtrage qualité + sous-échantillonnage stratifié |
| `04_anonymize.py` | sft_raw + dpo_raw | `processed/*_anonymized/` | Presidio (spaCy FR+EN), 6 types d'entités RGPD |
| `05_split_and_validate.py` | *_anonymized | `final/sft/` + `final/dpo/` | Split stratifié SFT 80/10/10 · DPO 90/10 + validation |

### Lancer le pipeline complet

```bash
make data-pipeline
```

### Étapes individuelles (toutes idempotentes)

```bash
make download        # Télécharge depuis HuggingFace
make build-sft       # Construit le dataset SFT
make build-dpo       # Construit le dataset DPO
make anonymize       # Anonymisation RGPD + rapport
make split           # Split + rapport de validation
```

---

## Schémas des datasets

### SFT — `data/final/sft/`  `{train, val, test}`

| Colonne | Type | Description |
|---------|------|-------------|
| `instruction` | `str` | Description de symptômes / question du patient |
| `response` | `str` | Évaluation clinique + recommandations structurées |
| `source` | `str` | Dataset d'origine (`frenchmedmcqa`, `medquad`, …) |
| `language` | `str` | `"fr"` ou `"en"` |
| `urgency_level` | `str` | `"max"`, `"moderate"` ou `"deferred"` |
| `confidence` | `float` | `0.7` (inféré) ou `1.0` (annoté manuellement) |

### DPO — `data/final/dpo/`  `{train, val}`

| Colonne | Type | Description |
|---------|------|-------------|
| `prompt` | `str` | Instruction médicale |
| `chosen` | `str` | Réponse cliniquement préférée |
| `rejected` | `str` | Réponse non préférée |
| `source` | `str` | `"ultramedical_preference"` |
| `language` | `str` | `"en"` |

### Charger les datasets en Python

```python
from datasets import load_from_disk

# DatasetDict complet
sft = load_from_disk("data/final/sft")
df_train = sft["train"].to_pandas()   # → pd.DataFrame
df_val   = sft["val"].to_pandas()
df_test  = sft["test"].to_pandas()

# Depuis HuggingFace Hub (après push)
from datasets import load_dataset
sft = load_dataset("ton_username/project14-sft")
```

---

## Publication sur HuggingFace Hub

### Pré-requis

S'authentifier avec l'une des deux méthodes suivantes :

```bash
# Option 1 — login interactif (stocke le token dans ~/.cache/huggingface/)
huggingface-cli login

# Option 2 — variable d'environnement (scripts non-interactifs, CI/CD)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

> Le token doit avoir la permission **write** sur ton espace HF.
> Créer ou gérer les tokens sur : https://huggingface.co/settings/tokens

### Commandes Make

```bash
# Publier les datasets finaux (sft + dpo) — dépôts publics
make push-datasets HF_USERNAME=ton_username

# Publier en mode privé
make push-datasets HF_USERNAME=ton_username HF_PRIVATE=1

# Publier aussi les datasets intermédiaires (sft_raw, dpo_raw, anonymized)
make push-datasets-all HF_USERNAME=ton_username
```

Ou en exportant la variable en amont :

```bash
export HF_USERNAME=ton_username
make push-datasets
```

### Dépôts créés

| Cible Make | Dépôt HuggingFace | Contenu |
|---|---|---|
| `push-datasets` | `{username}/project14-sft` | DatasetDict `{train, val, test}` |
| `push-datasets` | `{username}/project14-dpo` | DatasetDict `{train, val}` |
| `push-datasets-all` | `{username}/project14-sft-raw` | Dataset brut avant anonymisation |
| `push-datasets-all` | `{username}/project14-dpo-raw` | Dataset DPO brut |
| `push-datasets-all` | `{username}/project14-sft-anonymized` | SFT post-Presidio |
| `push-datasets-all` | `{username}/project14-dpo-anonymized` | DPO post-Presidio |

### Réutilisation depuis une autre machine

Une fois les datasets publiés, les charger directement sans re-lancer le pipeline :

```python
from datasets import load_dataset

sft = load_dataset("ton_username/project14-sft")
dpo = load_dataset("ton_username/project14-dpo")

# Accès aux splits
df_train = sft["train"].to_pandas()
df_val   = sft["val"].to_pandas()
df_test  = sft["test"].to_pandas()
```

---

## Explorer les données avec DuckDB

Le script `explore_data.py` expose tous les datasets comme des vues SQL :

```bash
# Shell interactif
uv run python scripts/explore_data.py

# Stats rapides (tailles, distributions)
uv run python scripts/explore_data.py --stats

# Requête directe
uv run python scripts/explore_data.py --query "SELECT urgency_level, COUNT(*) FROM sft_train GROUP BY 1"
```

Exemples de requêtes :

```sql
-- Vues disponibles : sft_train, sft_val, sft_test, dpo_train, dpo_val
-- + sft_raw, dpo_raw, sft_anonymized, dpo_anonymized
-- + raw_frenchmedmcqa_train, raw_medquad_train, ...

SELECT source, COUNT(*) FROM sft_train GROUP BY source;
SELECT urgency_level, COUNT(*) FROM sft_train GROUP BY urgency_level;
SELECT instruction, response FROM sft_test WHERE urgency_level = 'max' LIMIT 3;
SELECT * FROM dpo_train LIMIT 5;
```
