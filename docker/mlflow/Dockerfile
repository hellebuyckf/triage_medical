FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "mlflow[auth]>=2.13" \
    psycopg2-binary \
    google-cloud-storage

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 5000
ENTRYPOINT ["/start.sh"]
