# Data Lineage — Project 14

Pipeline complet de `data/raw/` vers `data/final/`, en 7 scripts numérotés.

```mermaid
flowchart TD

    %% ── SOURCES HUGGINGFACE ──────────────────────────────────────────────
    subgraph HF["🤗 HuggingFace Hub"]
        direction TB
        HF1["nthngdy/frenchmedmcqa<br/>(MCQ · FR)<br/>1 080 rows"]
        HF2["keivalya/MedQuad<br/>(QA · EN)<br/>16 407 rows"]
        HF3["ANR-MALADES/MediQAl mcqu<br/>(MCQ clinical · FR)<br/>17 017 rows"]
        HF4["ANR-MALADES/MediQAl oeq<br/>(open-ended · FR)<br/>4 969 rows"]
        HF5["TsinghuaC3I/UltraMedical-Preference<br/>(DPO pairs · EN)<br/>112 362 rows"]
    end

    %% ── SCRIPT 01 ───────────────────────────────────────────────────────
    S01(["01_download.py"])

    HF1 & HF2 & HF3 & HF4 & HF5 --> S01

    subgraph RAW["data/raw/  (Arrow IPC)"]
        direction TB
        R1["raw/frenchmedmcqa/<br/>train · val · test"]
        R2["raw/medquad/<br/>train"]
        R3["raw/mediql_mcqu/<br/>train · val · test"]
        R4["raw/mediql_oeq/<br/>test"]
        R5["raw/ultramedical_preference/<br/>train · val · test"]
    end

    S01 --> R1 & R2 & R3 & R4 & R5

    %% ── SCRIPT 02 ───────────────────────────────────────────────────────
    S02(["02_build_sft.py<br/>format_triage_response<br/>infer_urgency"])

    R1 -->|"MCQ → 86 rows"| S02
    R2 -->|"QA → 3 455 rows"| S02
    R3 -->|"MCQ clinical → 2 341 rows"| S02
    R4 -->|"open-ended → 616 rows"| S02

    subgraph PROC_SFT["data/processed/"]
        P1[("sft_raw.parquet<br/>6 498 rows<br/>{instruction, response, source,<br/>language, urgency_level, confidence}")]
    end

    S02 --> P1

    %% ── SCRIPT 03 ───────────────────────────────────────────────────────
    S03(["03_build_dpo.py<br/>filter_dpo_quality<br/>subsample_by_label_type"])

    R5 -->|"filter + subsample → 1 000 pairs"| S03

    subgraph PROC_DPO["data/processed/"]
        P2[("dpo_raw.parquet<br/>1 000 rows<br/>{prompt, chosen, rejected,<br/>source, language}")]
    end

    S03 --> P2

    %% ── SCRIPT 04 ───────────────────────────────────────────────────────
    S04(["04_anonymize.py<br/>Presidio · spaCy FR+EN<br/>PERSON, LOCATION, DATE_TIME,<br/>PHONE, EMAIL, NRP"])

    P1 --> S04
    P2 --> S04

    subgraph ANON["data/processed/"]
        A1[("sft_anonymized.parquet<br/>6 498 rows")]
        A2[("dpo_anonymized.parquet<br/>1 000 rows")]
        A3["rgpd_report.md"]
    end

    S04 --> A1 & A2 & A3

    %% ── SCRIPT 05 ───────────────────────────────────────────────────────
    S05(["05_split_and_validate.py<br/>stratified split on urgency_level<br/>SFT 80/10/10 · DPO 90/10"])

    A1 --> S05
    A2 --> S05

    subgraph FINAL["data/final/  (Parquet)"]
        direction TB
        F1[("sft_train.parquet<br/>4 957 rows")]
        F2[("sft_val.parquet<br/>620 rows")]
        F3[("sft_test.parquet<br/>620 rows")]
        F4[("dpo_train.parquet<br/>900 rows")]
        F5[("dpo_val.parquet<br/>100 rows")]
        F6["stats_report.md"]
        F7["rgpd_report.md"]
    end

    S05 --> F1 & F2 & F3 & F4 & F5 & F6 & F7

    %% ── ANNOTATION LOOP (optionnel) ──────────────────────────────────────
    S06(["06_annotate_sft.py<br/>Gemini LLM<br/>ré-étiquetage urgency_level<br/>+ exemples synthétiques"])

    F1 -->|"export subset"| S06

    subgraph ANNOT["data/annotation/"]
        AN1["sft_to_annotate.jsonl"]
        AN2["train_split_augmented.json<br/>{id → urgency_level_annotated}<br/>+ synth_ examples"]
    end

    S06 --> AN1 --> AN2

    %% ── STYLES ───────────────────────────────────────────────────────────
    classDef hf        fill:#fef3c7,stroke:#d97706,color:#000
    classDef raw       fill:#dbeafe,stroke:#3b82f6,color:#000
    classDef proc      fill:#ede9fe,stroke:#7c3aed,color:#000
    classDef anon      fill:#fce7f3,stroke:#db2777,color:#000
    classDef final     fill:#dcfce7,stroke:#16a34a,color:#000
    classDef annot     fill:#fff7ed,stroke:#ea580c,color:#000
    classDef script    fill:#1e293b,stroke:#475569,color:#fff

    class HF1,HF2,HF3,HF4,HF5 hf
    class R1,R2,R3,R4,R5 raw
    class P1,P2 proc
    class A1,A2,A3 anon
    class F1,F2,F3,F4,F5,F6,F7 final
    class AN1,AN2 annot
    class S01,S02,S03,S04,S05,S06 script
```

## Résumé des étapes

| Script | Entrée | Sortie | Transformation clé |
|--------|--------|--------|--------------------|
| `01_download.py` | HuggingFace Hub | `data/raw/` (Arrow) | Téléchargement + cache local |
| `02_build_sft.py` | 4 raw datasets | `sft_raw.parquet` (6 498 rows) | MCQ/QA → `instruction/response` + `infer_urgency` |
| `03_build_dpo.py` | ultramedical raw | `dpo_raw.parquet` (1 000 rows) | Filtrage qualité + sous-échantillonnage stratifié |
| `04_anonymize.py` | sft_raw + dpo_raw | `*_anonymized.parquet` | Presidio (spaCy FR+EN) sur 6 entités RGPD |
| `05_split_and_validate.py` | *_anonymized | `data/final/*.parquet` | Split stratifié (urgency_level) 80/10/10 |
| `06_annotate_sft.py` | sft_train subset | `train_split_augmented.json` | Ré-étiquetage + génération synthétique via Gemini |

## Schémas des fichiers finaux

**SFT** — `{instruction, response, source, language, urgency_level, confidence}`
**DPO** — `{prompt, chosen, rejected, source, language}`
