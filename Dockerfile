FROM python:3.11-slim

WORKDIR /app

# System dependencies for psycopg2 and others
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Imagen de producción (sin --reload). En desarrollo, docker-compose.yml sobrescribe el comando.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
