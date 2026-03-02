#!/bin/bash
#
mkdir -p ./data/raw/MediQAl
mkdir -p ./data/raw/UltraMedical-Preference
mkdir -p ./data/raw/MedQuad-MedicalQnADataset
mkdir -p ./data/raw/frenchmedmcqa

 hf download ANR-MALADES/MediQAl --repo-type=dataset --local-dir ./data/raw/MediQAl/.
 hf download TsinghuaC3I/UltraMedical-Preference --repo-type=dataset --local-dir ./data/raw/UltraMedical-Preference/.
 hf download keivalya/MedQuad-MedicalQnADataset --repo-type=dataset --local-dir ./data/raw/MedQuad-MedicalQnADataset/.
 hf download nthngdy/frenchmedmcqa --repo-type=dataset --local-dir ./data/raw/frenchmedmcqa/.
