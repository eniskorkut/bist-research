FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir git+https://github.com/saidsurucu/borsa-mcp.git \
    && python -c "import unified_mcp_server"

CMD ["borsa-mcp"]
