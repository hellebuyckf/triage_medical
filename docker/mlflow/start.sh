#!/bin/bash
# Entrypoint MLflow — génère basic_auth.ini puis démarre le serveur
set -e

# Créer le fichier de config basic-auth depuis les variables d'env
cat > "${MLFLOW_AUTH_CONFIG_PATH}" <<EOF
[mlflow]
default_permission = NO_PERMISSIONS
database_uri = postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}
admin_username = ${MLFLOW_ADMIN_USERNAME}
admin_password = ${MLFLOW_ADMIN_PASSWORD}
EOF

exec mlflow server \
    --backend-store-uri "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --app-name basic-auth \
    --workers 1 \
    --host 0.0.0.0 \
    --port 5000
