FROM python:3.12-slim

WORKDIR /requests_service

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "requests.py"]

# Сборка из папки services/requests:
# docker build -t mobile-price-requests:1 .

# Запуск в подготовленной тестовой сети:
# docker run --rm --network mobile-price-lab4-test mobile-price-requests:1