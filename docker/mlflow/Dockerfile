FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow>=2.13"

# Le répertoire mlruns/ est monté en volume depuis le host
VOLUME ["/mlruns"]

EXPOSE 5000

CMD ["mlflow", "ui", \
     "--backend-store-uri", "/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
