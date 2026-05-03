FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir openbb openbb-yfinance openbb-mcp-server

EXPOSE 8001

CMD ["openbb-mcp", "--transport", "streamable-http", "--port", "8001", "--host", "0.0.0.0"]
