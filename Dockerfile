# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-lc", "exec gunicorn app:app -w 4 -k gthread --threads 4 -b 0.0.0.0:${PORT:-8080} --timeout 60"]
