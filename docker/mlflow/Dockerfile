FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.13"

# mlflow.db monté depuis le host (métadonnées runs/métriques/params)
# mlruns/ monté depuis le host (artefacts : poids LoRA, rapports...)
VOLUME ["/mlflow.db", "/mlruns"]

EXPOSE 5000

CMD ["mlflow", "ui", \
     "--backend-store-uri", "sqlite:////mlflow.db", \
     "--default-artifact-root", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
