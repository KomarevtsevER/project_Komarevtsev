FROM python:3.12-slim

COPY . /ml_service

VOLUME ["/models"]

WORKDIR /ml_service

RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Сборка из папки services/ml_service:
# docker build -t mobile-price-service:1 .

# Запуск из папки services/ml_service:
# docker run --name mobile-price-service -p 8001:8000 -v "$(pwd)/../models:/models:ro" mobile-price-service:1