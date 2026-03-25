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
│   ├── dpo_raw/                      #   Dataset  — ~2 154 lignes (480 synthetic + 1 674 hard neg)
│   ├── dpo_hard_negatives/           #   Dataset  — 1 674 lignes (généré par 03b, requiert SFT)
│   ├── sft_anonymized/               #   Dataset  — 6 498 lignes
│   ├── dpo_anonymized/               #   Dataset  — ~2 154 lignes
│   ├── sft_tokenized/                #   Dataset  — formaté prompt/completion (généré par 10)
│   └── rgpd_report.md
│
└── final/                            # HF Arrow — DatasetDict (générés par 05)
    ├── sft/                          #   DatasetDict {train: 4 544, val: 568, test: 569}
    │   └── dataset_dict.json
    ├── dpo/                          #   DatasetDict {train: ~1 938, val: ~216}
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

02_build_sft.py  +  03_build_dpo.py  [+  03b_sft_errors.py  (optionnel, après SFT)]
  └─► data/processed/sft_raw/  +  dpo_raw/  +  dpo_hard_negatives/  (Dataset HF)

04_anonymize.py
  └─► data/processed/sft_anonymized/  +  dpo_anonymized/  (Dataset HF)

05_split_and_validate.py
  └─► data/final/sft/  +  dpo/  (DatasetDict HF)
```

| Script | Entrée | Sortie | Transformation clé |
|--------|--------|--------|-------------------|
| `01_download.py` | HuggingFace Hub | `data/raw/` | Téléchargement + cache local |
| `02_build_sft.py` | 4 raw datasets | `processed/sft_raw/` (6 498 lignes) | MCQ/QA → `instruction/response` + `infer_urgency` |
| `03_build_dpo.py` | `sft_raw` + `dpo_hard_negatives` (opt.) | `processed/dpo_raw/` (~2 154 lignes) | Paires synthétiques (swap table) + dédoublonnage sur prompt + merge hard negatives |
| `03b_sft_errors.py` | `final/sft/train` + `checkpoints/sft` | `processed/dpo_hard_negatives/` (1 674 lignes) | Inférence SFT → paires (chosen=GT, rejected=erreur réelle) sur les 36.8% d'exemples mal classés |
| `04_anonymize.py` | sft_raw + dpo_raw | `processed/*_anonymized/` | Presidio (spaCy FR+EN), 6 types d'entités RGPD |
| `05_split_and_validate.py` | *_anonymized | `final/sft/` + `final/dpo/` | Split stratifié SFT 80/10/10 · DPO 90/10 + validation |

### Lancer le pipeline complet

```bash
make data-pipeline   # données de base (sans hard negatives)
```

Pour le pipeline DPO avec hard negatives (recommandé, après SFT) :

```bash
make dpo-pipeline-hard   # sft-errors → rebuild-dpo → train-dpo → evaluate-dpo → export
```

### Étapes individuelles (toutes idempotentes)

```bash
make download        # Télécharge depuis HuggingFace
make build-sft       # Construit le dataset SFT
make build-dpo       # Construit le dataset DPO (merge hard negatives si disponibles)
make sft-errors      # Génère les hard negatives DPO (requiert checkpoints/sft)
make rebuild-dpo     # Reconstruit dpo_raw + re-lance anonymize + split
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

Le token HuggingFace est lu automatiquement depuis le fichier **`.env` à la racine du projet** :

```bash
# .env  (ne jamais committer ce fichier — déjà dans .gitignore)
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

> Créer ou gérer les tokens sur : https://huggingface.co/settings/tokens
> Le token doit avoir la permission **write**.

Ordre de priorité du chargement :
1. `.env` à la racine du projet (via `python-dotenv`)
2. Variable `HF_TOKEN` déjà définie dans le shell (non écrasée)
3. Credentials stockés par `huggingface-cli login` (dernier recours)

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
