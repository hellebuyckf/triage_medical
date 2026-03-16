# SFT Iterations — Qwen3-1.7B LoRA Fine-tuning

Historique des itérations d'entraînement SFT sur le dataset de triage médical.
Modèle de base : `unsloth/Qwen3-1.7B-Base` — Checkpoint : `checkpoints/sft/`

---

## Résumé des résultats

| # | Date | LoRA r | Epochs | Val Acc | Test Acc | F1 Macro (test) | Format Compliance | Non-parseables | Statut |
|---|------|--------|--------|---------|----------|-----------------|-------------------|----------------|--------|
| 1 | 2026-03-10 | 16 | 3 | 0.0% | 66.7% | 0.556 | 0.0% | 582/620 | ❌ Format cassé |
| 2 | 2026-03-11 | 16 | 3 | 0.0% | 0.0% | 0.000 | 0.0% | 595/620 | ❌ Régression totale |
| 3 | 2026-03-12 matin | 16 | 3 | 60.7% | 61.6% | 0.553 | 36.4% | 170/620 | ⚠️ Format partiel |
| 4 | 2026-03-12 soir | 32 | 5 | 64.9% | 67.6% | 0.638 | 30.3% | 176/620 | ⚠️ Accuracy ↑ Format ↓ |
| 5 | 2026-03-13 | 32 | 5 | 59.2% | 62.3% | 0.619 | 96.3% | 0/620 | ⚠️ Format ✅ Accuracy ↓ |
| 6 | 2026-03-14 | 32 | 5 | **65.0%** | **67.6%** | **0.675** | **93.2%** | **0/620** | ✅ Meilleur checkpoint |

---

## Hyperparamètres fixes (toutes itérations)

| Paramètre | Valeur |
|-----------|--------|
| `learning_rate` | 2e-4 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 (effective batch = 16) |
| `max_seq_length` | 1024 |
| `lora_dropout` | 0.05 |
| `optimizer` | AdamW |

---

## Détail des itérations

### Itération 1 — 2026-03-10
**MLflow run :** `a7be3303`

**Config LoRA :** r=16, alpha=32, epochs=3

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | 0.0% | 66.7% |
| F1 Macro | 0.000 | 0.556 |
| Format Compliance | 0.0% | 0.5% |
| Non-parseables | 582/620 | 580/620 |

**Diagnostic :** Le modèle génère des réponses médicales correctes sur le fond, mais sans le préfixe `URGENCE X`. Le parser ne peut pas extraire le label → 0 prédiction valide sur val. Le test_accuracy à 66.7% est trompeur : c'est le hasard sur une seule classe parseable.

**Root cause :** `format_triage_response()` non appliqué dans `02_build_sft.py` lors de la construction du dataset SFT. Les données d'entraînement ne contiennent pas le format cible.

**Fix appliqué :** Ajout du wrapper `format_triage_response(urgency_level, raw_response)` dans toutes les fonctions `transform_*` de `02_build_sft.py` → pipeline data-all relancé.

---

### Itération 2 — 2026-03-11
**MLflow run :** `b20f5658`

**Config LoRA :** r=16, alpha=32, epochs=3

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | 0.0% | 0.0% |
| F1 Macro | 0.000 | 0.000 |
| Format Compliance | 0.0% | 0.0% |
| Non-parseables | 595/620 | 595/620 |

**Diagnostic :** Régression par rapport à iter1 — 100% non-parseable sur les deux splits. Le modèle a appris à ne jamais émettre de label d'urgence.

**Root cause :** Presidio (spaCy EN) détecte les mots français `"URGENCE"`, `"MODÉRÉE"`, `"MAXIMALE"`, `"DIFFÉRÉE"` comme entités `PERSON` (noms étrangers pour le NER anglais). Résultat : les labels dans les réponses SFT sont remplacés par `<PERSON>` lors de l'anonymisation → le modèle n'a jamais vu le pattern `URGENCE X` pendant l'entraînement.

**Fix appliqué :** Ajout de `"urgence"`, `"maximale"`, `"modérée"`, `"différée"` (et variantes) à `MEDICAL_TERMS_ALLOWLIST` dans `utils.py` → `filter_presidio_false_positives()` protège ces termes → pipeline data-all relancé.

---

### Itération 3 — 2026-03-12 (matin)
**MLflow run :** `437cbfd6`

**Config LoRA :** r=16, alpha=32, epochs=3

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | 60.7% | 61.6% |
| F1 Macro | 0.527 | 0.553 |
| Format Compliance | 36.4% | 38.7% |
| Non-parseables | 170/620 | 156/620 |

**Matrice de confusion (Val) :**

|  | Prédit max | Prédit moderate | Prédit deferred |
|--|-----------|----------------|----------------|
| **Réel max** | 16 | 29 | 45 |
| **Réel moderate** | 10 | 132 | 29 |
| **Réel deferred** | 9 | 55 | 125 |

**Diagnostic :** Premier run fonctionnel — le modèle génère des labels d'urgence. Mais seulement 36% des réponses respectent le format complet (avec section `Évaluation clinique :` et `Recommandations :`). La classe `max` est très mal apprise (recall 16/90 = 18%) car sous-représentée et difficile à discriminer du contenu textuel.

**Observations :** Format compliance de 36% indique que le modèle génère parfois `URGENCE MODÉRÉE` seul sans le bloc structuré. La capacité LoRA (r=16) semble insuffisante pour mémoriser le template.

---

### Itération 4 — 2026-03-12 (soir)
**MLflow run :** `f1f1db0f`

**Config LoRA :** r=32, alpha=64, epochs=5

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | 64.9% | 67.6% |
| F1 Macro | 0.611 | 0.638 |
| Format Compliance | 30.3% | 29.2% |
| Non-parseables | 176/620 | 166/620 |

**Matrice de confusion (Val) :**

|  | Prédit max | Prédit moderate | Prédit deferred |
|--|-----------|----------------|----------------|
| **Réel max** | 35 | 19 | 34 |
| **Réel moderate** | 18 | 120 | 30 |
| **Réel deferred** | 13 | 42 | 133 |

**Diagnostic :** Augmentation de la capacité LoRA (r=16→32) et des epochs (3→5). Accuracy +4 pts et F1 +0.085 — la classe `max` s'améliore nettement (recall 35/88 = 40% vs 18% avant). Mais format compliance régresse de 36% → 30% : avec plus d'epochs le modèle sur-apprend les variations de style plutôt que le template fixe.

---

### Itération 5 — 2026-03-13
**MLflow run :** `9b34c3a4`

**Config LoRA :** r=32, alpha=64, epochs=5 + cublasLt workaround Unsloth

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | 59.2% | 62.3% |
| F1 Macro | 0.589 | 0.619 |
| Format Compliance | **96.3%** | **97.9%** |
| Non-parseables | **0/620** | **0/620** |

**Matrice de confusion (Val) :**

|  | Prédit max | Prédit moderate | Prédit deferred |
|--|-----------|----------------|----------------|
| **Réel max** | 151 | 15 | 35 |
| **Réel moderate** | 78 | 106 | 26 |
| **Réel deferred** | 53 | 46 | 110 |

**Diagnostic :** Percée sur le format — 0 non-parseable, 96% compliance. Le fix `MEDICAL_TERMS_ALLOWLIST` était bien la root cause des itérations précédentes. Cependant l'accuracy chute à 59% : le modèle sur-prédit `max` (78 FP sur `moderate`, 53 FP sur `deferred`). Le biais `max` vient probablement du déséquilibre de classe dans les données d'entraînement combiné à un over-fitting après 5 epochs.

**Changement infra :** Ajout de `torch.backends.cuda.preferred_blas_library("cublaslt")` → résout incompatibilité Unsloth+Qwen3 → gain de vitesse ×2 sur RTX 4060 Ti.

---

### Itération 6 — 2026-03-14 ✅ Meilleur checkpoint
**MLflow run :** `317816c6`

**Config LoRA :** r=32, alpha=64, epochs=5 (même checkpoint qu'iter5, évaluation corrigée)

| Métrique | Val | Test |
|----------|-----|------|
| Accuracy | **65.0%** | **67.6%** |
| F1 Macro | **0.649** | **0.675** |
| Format Compliance | 93.2% | 94.7% |
| Non-parseables | 0/620 | 0/620 |

**Matrice de confusion (Val) :**

|  | Prédit max | Prédit moderate | Prédit deferred |
|--|-----------|----------------|----------------|
| **Réel max** | 150 | 14 | 37 |
| **Réel moderate** | 59 | 121 | 30 |
| **Réel deferred** | 38 | 39 | 132 |

**Diagnostic :** Meilleur équilibre accuracy/format à ce jour. Recall `max` = 74.6% (150/201), recall `moderate` = 57.6% (121/210), recall `deferred` = 63.5% (132/209). La classe `moderate` reste la plus difficile (confondue avec `max` et `deferred`).

**Points restants à améliorer :**
- 59 FP `moderate → max` : confusion sur les cas intermédiaires (maladies chroniques avec symptômes modérés)
- 38 FP `deferred → max` : "pain" ou "syndrome" dans le titre biaise vers `max`
- Accuracy cible semaine 2 : 70% — manque 5 pts

---

## Analyse des problèmes systémiques rencontrés

### 1. Format compliance vs accuracy : tension persistante
Les itérations 3-4 montrent une accuracy correcte mais format bas. L'itération 5 inverse. L'itération 6 combine les deux grâce à la correction de la pipeline data (Presidio allowlist). **Leçon :** un bug data en amont peut masquer plusieurs itérations de tuning.

### 2. Biais vers la classe `max`
Le modèle sur-prédit `max` dans toutes les itérations. Causes probables :
- Le mot-clé `"symptoms"` (MedQuAD) déclenche l'inférence `max` dans `infer_urgency()`
- Dataset déséquilibré côté `max` après le split stratifié

### 3. Génération de tokens parasites
Certaines réponses contiennent des tokens Unicode (`𫟦`, `ForCanBeConverted`) en fin de génération → artifact Qwen3 lié au template de chat. À filtrer en post-processing lors de l'évaluation.

---

## Prochaines étapes

- [ ] Ajuster `infer_urgency()` pour réduire le biais `max` sur les questions génériques
- [ ] Tester `epochs=3` avec `r=32` pour limiter l'over-fitting du biais `max`
- [ ] Ajouter un post-processing dans `12_evaluate_sft.py` pour filtrer les tokens parasites
- [ ] Lancer DPO sur le meilleur checkpoint SFT (iter6)
