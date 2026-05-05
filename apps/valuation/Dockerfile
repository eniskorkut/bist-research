FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY apps/valuation/pyproject.toml ./apps/valuation/pyproject.toml
COPY apps/valuation/README.md ./apps/valuation/README.md
COPY apps/valuation/src ./apps/valuation/src
COPY apps/valuation/scripts ./apps/valuation/scripts
COPY apps/valuation/tests ./apps/valuation/tests
COPY apps/market_radar/pyproject.toml ./apps/market_radar/pyproject.toml
COPY apps/market_radar/README.md ./apps/market_radar/README.md
COPY apps/market_radar/src ./apps/market_radar/src
COPY apps/market_radar/scripts ./apps/market_radar/scripts
COPY apps/market_radar/tests ./apps/market_radar/tests
COPY scripts ./scripts
COPY conftest.py ./conftest.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ./apps/valuation ./apps/market_radar \
    && pip install --no-cache-dir pytest

CMD ["streamlit", "run", "apps/valuation/src/valuation/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8502"]
