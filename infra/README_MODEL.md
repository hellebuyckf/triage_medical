---
language:
  - fr
  - en
license: apache-2.0
base_model: unsloth/Qwen3-1.7B
tags:
  - medical
  - triage
  - classification
  - lora
  - sft
  - dpo
  - qwen3
pipeline_tag: text-generation
---

# Qwen3-1.7B — Agent de Triage Médical (SFT + DPO)

Modèle fine-tuné pour le triage médical à 3 niveaux d'urgence.  
Développé dans le cadre du POC **Centre Hospitalier Saint-Aurélien (CHSA)** — OpenClassrooms P14.

> ⚠️ **Usage POC uniquement.** Ce modèle ne remplace pas l'avis d'un professionnel de santé.  
> Labels d'urgence inférés par règles, non annotés par des cliniciens.

---

## Utilisation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "FrancoisFormation/qwen3-triage-dpo"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

SYSTEM_PROMPT = (
    "Tu es un assistant médical de triage. "
    "Pour chaque question ou symptôme, tu dois :\n"
    "1. Déterminer le niveau d'urgence : URGENCE MAXIMALE, URGENCE MODÉRÉE ou URGENCE DIFFÉRÉE.\n"
    "2. Fournir une évaluation clinique concise.\n"
    "3. Donner des recommandations claires.\n\n"
    "Format de réponse :\n"
    "URGENCE [MAXIMALE|MODÉRÉE|DIFFÉRÉE]\n\n"
    "Évaluation clinique : ...\n\n"
    "Recommandations : ..."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Douleur thoracique intense avec sudation et nausées depuis 30 minutes."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Sortie attendue :**
```
URGENCE MAXIMALE

Évaluation clinique : Tableau clinique évocateur d'un syndrome coronarien aigu...

Recommandations : Appelez le 15 (SAMU) ou rendez-vous aux urgences immédiatement. Ne restez pas seul.
```

---

## Niveaux d'urgence

| Label | Signification | Recommandation type |
|---|---|---|
| `URGENCE MAXIMALE` | Risque vital immédiat | SAMU (15) / urgences immédiates |
| `URGENCE MODÉRÉE` | Consultation urgente sous 24–48h | Médecin dans les 24 à 48 heures |
| `URGENCE DIFFÉRÉE` | Pas de risque immédiat | Rendez-vous médecin traitant |

---

## Performances — Test Set (569 exemples, distribution équilibrée 33/33/33)

| Métrique | SFT seul | SFT + DPO | Seuil POC | Statut |
|---|---|---|---|---|
| Accuracy globale | 65.03% | **64.67%** | ≥ 60% | ✅ |
| F1 Macro | 0.6463 | **0.6437** | — | — |
| F2 Macro (β=2) | 0.6470 | **0.6441** | ≥ 0.60 | ✅ |
| **Recall URGENCE MAX** | 79.57% | **78.49%** | ≥ 75% | ✅ |
| Format Compliance | 75.9% | **76.3%** | ≥ 70% | ✅ |

> **F2 Macro (β=2)** : pénalise 4× plus les faux négatifs que les faux positifs — aligné avec la logique médicale (rater une urgence critique est plus grave que sur-escalader).
>
> **Recall MAX** : fraction des cas réellement critiques correctement identifiés. Métrique prioritaire pour ce POC.

### Matrice de confusion — Modèle DPO final

|  | Prédit MAX | Prédit MODÉRÉ | Prédit DIFFÉRÉ |
|---|---|---|---|
| **Réel MAX** (186) | **146** | 15 | 25 |
| **Réel MODÉRÉ** (194) | 72 | **97** | 25 |
| **Réel DIFFÉRÉ** (189) | 35 | 29 | **125** |

---

## Pipeline d'entraînement

### Données

| Dataset | Source HuggingFace | Usage | Langue | Taille |
|---|---|---|---|---|
| FrenchMedMCQA | `nthngdy/frenchmedmcqa` | SFT | FR | ~3 105 |
| MedQuAD | `keivalya/MedQuad-MedicalQnADataset` | SFT | EN | ~16 407 |
| MediQAl (mcqu) | `ANR-MALADES/MediQAl` | SFT | FR | ~17 017 |
| MediQAl (oeq) | `ANR-MALADES/MediQAl` | SFT | FR | ~4 969 |
| Paires DPO synthétiques | SFT train (auto-généré) | DPO | FR/EN | ~957 |

**Split final SFT :** 4 544 train / 568 val / 569 test  
**Anonymisation RGPD :** Microsoft Presidio + spaCy (`fr_core_news_md`, `en_core_web_md`)

### Phase 1 — SFT (Supervised Fine-Tuning)

| Paramètre | Valeur |
|---|---|
| Base model | `unsloth/Qwen3-1.7B` |
| Framework | Unsloth + PEFT (LoRA) + TRL SFTTrainer |
| LoRA r / alpha | 64 / 128 |
| LoRA dropout | 0.0 |
| Target modules | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 2e-4 |
| Epochs | 8 |
| Batch effectif | 16 (batch=4 × grad_accum=4) |
| Max seq length | 1 024 tokens |
| Seed | 42 |
| Hardware | NVIDIA RTX 4060 Ti (16 GB VRAM) |

### Phase 2 — DPO (Direct Preference Optimization)

| Paramètre | Valeur |
|---|---|
| Algorithme | DPO sigmoid (Rafailov et al. 2023) |
| Framework | TRL DPOTrainer + Unsloth |
| Référence | Modèle SFT fusionné (ref_model=None) |
| Beta (KL penalty) | 0.3 |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Learning rate | 5e-6 |
| Epochs | 1 |
| Batch effectif | 16 (batch=1 × grad_accum=16) |
| Paires DPO | 861 train / 96 val |
| Seed | 42 |

**Construction des paires DPO :**  
Paires synthétiques (chosen = réponse correcte, rejected = même corps clinique avec label d'urgence erroné) + hard negatives extraits des erreurs du modèle SFT. Distribution des labels équilibrée (max/moderate/deferred présents dans chosen ET rejected) pour éviter le reward hacking.

---

## Limites et biais

- **Labels inférés, non cliniques** : les niveaux d'urgence sont assignés par règles à base de mots-clés, non par des médecins. Les erreurs de labelling propagent des biais dans le modèle.
- **Distribution artificielle** : le test set est équilibré 33/33/33 par sous-échantillonnage. En conditions réelles, la distribution des urgences est très différente.
- **Modèle compact (1.7B)** : capacité de raisonnement limitée sur des cas ambigus ou multi-pathologies.
- **Pas de validation clinique** : ce modèle n'a pas été évalué par des professionnels de santé ni sur des données patients réelles.
- **Langues** : entraîné sur FR et EN, performances non garanties sur d'autres langues.

---

## Contexte légal

Ce modèle est un POC académique. Il **ne constitue pas un dispositif médical** au sens du règlement (UE) 2017/745. Toute utilisation en contexte clinique réel requiert une validation réglementaire complète.

---

## Citation

```bibtex
@misc{hellebuyck2026triage,
  author    = {Hellebuyck, François},
  title     = {Qwen3-1.7B Medical Triage Agent (SFT + DPO)},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/FrancoisFormation/qwen3-triage-dpo}
}
```
