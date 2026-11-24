FROM python:3.12-slim

WORKDIR /ml_service

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

VOLUME ["/models"]

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Сборка из папки services/ml_service:
# docker build -t mobile-price-service:2 .

# Запуск из папки services/ml_service:
# docker run --name mobile-price-service-v2 -p 8001:8000 -v "$(pwd)/../models:/models:ro" mobile-price-service:2