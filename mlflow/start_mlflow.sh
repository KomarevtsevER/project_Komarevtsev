#!/bin/bash
# Скрипт запуска MLflow UI с SQLite хранилищем
# Запускать из папки mlflow: sh start_mlflow.sh

mlflow ui \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5000