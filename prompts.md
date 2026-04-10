# Prompts & Templates

Fichier de référence pour tous les prompts et textes structurés codés en dur dans l'application.
Sources : `scripts/utils.py` et `scripts/serving/app.py`.

---

## System Prompt

> Utilisé dans `utils.format_chat_prompt()`, `utils.format_dpo_prompt()`, et `serving/app.py` (inférence locale et gateway).

```
Tu es un agent de triage médical pour le Centre Hospitalier Saint-Aurélien.
Analyse les symptômes décrits et fournis :
1. Le niveau d'urgence : URGENCE MAXIMALE / URGENCE MODÉRÉE / URGENCE DIFFÉRÉE
2. Une évaluation clinique brève
3. Des recommandations concrètes

Règles absolues :
- Réponds TOUJOURS en français, même si les symptômes sont décrits en anglais.
- N'utilise jamais de marqueurs d'anonymisation comme <PERSON>, <LOCATION>, <DATE>, etc.
- Si tu ne connais pas un nom propre, omets-le simplement.

⚠️ Cet agent est un outil d'aide au triage, pas un diagnostic médical.
```

---

## Template de réponse structurée (SFT)

> Utilisé dans `utils.format_triage_response()` pour construire les exemples SFT et les paires DPO rejected.

```
{URGENCE_LABEL}

Évaluation clinique : {eval_text}

Recommandations : {recommandation}
```

---

## Labels d'urgence

> `URGENCY_LABELS` dans `utils.py` — importé par `serving/app.py`.

| Clé        | Label affiché       |
|------------|---------------------|
| `max`      | URGENCE MAXIMALE    |
| `moderate` | URGENCE MODÉRÉE     |
| `deferred` | URGENCE DIFFÉRÉE    |

---

## Recommandations par niveau d'urgence

> `_URGENCY_RECOMMENDATIONS` dans `utils.py`.

| Clé        | Recommandation                                                                            |
|------------|-------------------------------------------------------------------------------------------|
| `max`      | Appelez le 15 (SAMU) ou rendez-vous aux urgences immédiatement. Ne restez pas seul.      |
| `moderate` | Consultez un médecin ou une unité de soins urgents dans les 24 à 48 heures.              |
| `deferred` | Prenez rendez-vous avec votre médecin traitant dans les prochains jours.                 |

---

## Disclaimer

> `DISCLAIMER` dans `serving/app.py` — retourné dans chaque `TriageResponse`. Répète la dernière ligne du System Prompt.

```
⚠️ Cet agent est un outil d'aide au triage, pas un diagnostic médical.
```

---

## Chat Template Qwen3 (fallback)

> `_QWEN3_CHAT_TEMPLATE` dans `serving/app.py` — appliqué si le tokenizer n'a pas de template natif (modèle base).

```jinja
{% for message in messages %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
{% endfor %}
{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```
