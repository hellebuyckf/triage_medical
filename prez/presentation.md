# Soutenance Projet 14 : Fine-Tuning LLM pour le Triage Médical
## Agent IA pour le Centre Hospitalier Saint-Aurélien (CHSA)

---

## 1. Introduction et Problématique

* **Contexte** : Augmentation de 40% des messages patients aux urgences du CHSA.
* **Problématique** : Évaluation manuelle chronophage, hétérogénéité des demandes, risque de retarder les urgences vitales.
* **Objectif POC** : Développer un agent de triage (LLM) capable de classifier l'urgence (Maximale, Modérée, Différée) avec une justification clinique structurée.
* **Contraintes** : RGPD (anonymisation), Latence < 2s, Recall Max ≥ 75%.

---

## 2. Sources de données et transformation

*   **SFT (Supervised Fine-Tuning) via Hugging Face Hub** :
    *   **FrenchMedMCQA** (FR) : Q&A médical spécialisé.
    *   **MedQuAD** (EN) : Questions patients sur les maladies.
    *   **MediQAl** (FR) : Consultations médicales réelles anonymisées.
*   **DPO (Alignment) via Hugging Face Hub** :
    *   **UltraMedical-Preference** (EN) : 112 000 paires de préférences cliniques.
*   **Transformations & Datasets finaux** :
    1. **Filtrage & Mapping** : Conversion des formats hétérogènes vers le format cible `(Prompt / Urgence / Justification)`.
    2. **Équilibrage** : Sous-échantillonnage pour obtenir **33% par classe** d'urgence.
    3. **Anonymisation** : Pipeline Presidio (PERSON, DATE, PHONE) avec liste blanche médicale.
    4. **Volume final** : **6 498 exemples SFT** et **2 154 paires DPO** (dont 78% de *hard negatives*).

---

## 3. Supervised Fine-Tuning (SFT) : Config LoRA

*   **Modèle** : Qwen3-1.7B (Base) + Unsloth (compatible écosystème **Hugging Face**).
*   **Paramètres LoRA (Low-Rank Adaptation)** :
    *   **Rank `r = 32`** : Dimension de la matrice de bas rang. Un rang de 32 (vs 16 standard) est nécessaire ici pour capturer la complexité de la classification à 3 classes et le formatage structuré.
    *   **Alpha `α = 64`** : Facteur de mise à l'échelle (Scaling). Un ratio `α = 2r` stabilise l'entraînement en équilibrant la contribution des poids appris par rapport aux poids originaux.
*   **Cible** : `q/k/v/o_proj` + `gate/up/down_proj` (MLP et Attention) pour une adaptation complète.

---

## 4. Alignement par Préférences Directes (DPO)

*   **Rôle du paramètre Beta (`β`)** :
    * Contrôle la force de la **pénalité KL** (Kullback-Leibler) par rapport au modèle SFT de référence.
*   **Configuration `β = 1.0`** (Contrainte forte) :
    *   **Pourquoi ?** Évite que le modèle ne diverge trop du SFT.
    *   **Résultat** : Performance stable (64.7% Acc) et maintien du recall sur les classes non-urgentes.
*   **Comparaison `β = 0.5`** (Contrainte faible) :
    * Avait entraîné une **sur-correction** : le modèle prédisait "MAXIMALE" systématiquement pour "sécuriser" la réponse, effaçant la distinction entre urgence modérée et différée (recall différé tombé à 33%).

---

## 5. Frameworks clés du projet

Le succès du POC repose sur l'intégration de frameworks spécialisés :

*   **Entraînement (Écosystème Hugging Face)** :
    *   **TRL (Transformer Reinforcement Learning)** : Bibliothèque **Hugging Face** pour les `SFTTrainer` et `DPOTrainer`.
    *   **PEFT (LoRA)** : Bibliothèque **Hugging Face** pour l'adaptation fine de Qwen3-1.7B.
    *   **Unsloth** : Accélération LoRA (×2 plus rapide, -70% de mémoire VRAM).
*   **Inférence & Déploiement** :
    *   **vLLM** : Moteur d'inférence (PagedAttention, Continuous Batching).
    *   **FastAPI** : Gateway API asynchrone et validation Pydantic.
*   **Infrastructure & MLOps** :
    *   **Terraform** : Infrastructure-as-Code (IaC) pour Google Cloud (GCP).
    *   **MLflow** : Tracking des expériences et Model Registry.
    *   **Microsoft Presidio** : Pipeline d'anonymisation RGPD.


---

## 6. Architecture & Infrastructure (GCP)

*   **L'avantage vLLM** : 
    *   Latence stable de **850 ms** vs 3-5s (Continuous Batching).
    *   Optimisation mémoire GPU (PagedAttention).
*   **Composants Cloud** :
    *   **Gateway API (Cloud Run + FastAPI)** : Validation et routage.
    *   **Inférence (GCE + vLLM)** : VM avec **GPU NVIDIA L4** (24 Go VRAM).
    *   **Tracking (MLflow + Cloud SQL)** : Suivi centralisé des métriques.
*   **Schéma d'Architecture** :
    ![Architecture du Projet](./diagramme/fossflow-export-2026-04-15T14_42_44.995Z.png)

---

## 7. Déploiement CI/CD

* **Qualité** : Ruff (linting), Pyright (types), Pytest (tests unitaires avec mocks vLLM).
* **Pipeline** : GitHub Actions → Google Artifact Registry → Cloud Run (Gateway) & GCE (Inférence).
* **MLflow** : Versioning des modèles et tracking des hyperparamètres (Beta, Rank, Alpha).

---

## 8. Résultats finaux et Conclusion

| Métrique | Cible POC | SFT (Test) | **DPO (Test)** | Statut |
|:---------|:---------:|:----------:|:--------------:|:------:|
| **Accuracy globale** | ≥ 60 % | 65.0 % | **64.7 %** | **[OK]** |
| **F1-macro** | > 0.60 | 0.646 | **0.644** | **[OK]** |
| **Recall MAXIMALE** | **≥ 75 %** | 79.6 % | **78.5 %** | **[OK]** |
| **Recall DIFFÉRÉE** | — | 63.5 % | **66.1 %** | **[OK]** |
| **Compliance format** | ≥ 70 % | 75.9 % | **76.3 %** | **[OK]** |

* **Conclusion** : 4/4 critères validés. Le passage à l'échelle nécessitera du RAG (protocoles locaux) et une validation sur 500 cas réels annotés par des urgentistes.
