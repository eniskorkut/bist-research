FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY alerts.csv ./alerts.csv

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

CMD ["python", "scripts/run_volume_alert.py", "--index", "XU030", "--dry-run", "--check-interval", "5"]
