# BIST Research

Iki parcalı BIST araştırma ortamı.

## Mimari

- `borsapy-alert`: canlı quote/hacim izler, hacim alarmı üretir.
- `borsa-mcp`: Codex/LLM tarafına MCP finansal veri tool server sağlar.
- Telegram yok. Al-sat yok. Sadece araştırma, veri izleme ve analiz.

## borsapy-alert

`borsapy` ile `TradingViewStream` bağlantısı açar. `XU100`, `XU030`, `XBANK` gibi endekslerin bileşenlerini veya elle verilen sembolleri izler. Her sembol için `bp.Ticker(symbol).history(period="1ay")` üzerinden son 20 günlük ortalama hacmi hesaplar.

Alarm kriterleri:

- Gün içi toplam hacim / 20 günlük ortalama hacim >= `1.5`
- Son 5 dakikalık hacim artışı / 20 günlük ortalama hacim >= `0.05`
- Aynı sembol için cooldown: 15 dakika

Alarm terminale yazılır ve `alerts.csv` dosyasına append edilir. `--dry-run` sadece alarm satırını dry-run olarak işaretler.

## borsa-mcp

`saidsurucu/borsa-mcp` reposundan kurulur. MCP server `borsa-mcp` komutu ile başlar. Canlı alarm burada yapılmaz. Geçmiş veri, bilanço, teknik analiz, KAP haberleri ve doğal dil tool çağrıları için kullanılır.

## Docker Compose

```shell
cp .env.example .env
docker compose build
docker compose up
```

Varsayılan servis komutları:

```shell
python scripts/run_volume_alert.py --index XU030 --dry-run --check-interval 5
borsa-mcp
```

`alerts.csv` host dosyası olarak `./borsapy-alert/alerts.csv` konumuna bağlıdır.

## Local Test Komutları

```shell
cd borsapy-alert
python -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/run_volume_alert.py --index XU030 --dry-run --duration 300 --max-symbols 5
python scripts/run_volume_alert.py --symbols THYAO ASELS GARAN --dry-run --duration 120
```

BIST sembol/history hızlı kontrol:

```shell
python - <<'PY'
import borsapy as bp
print(bp.Index("XU030").component_symbols)
print(bp.Ticker("THYAO").history(period="1ay").tail())
PY
```

borsa-mcp hızlı kontrol:

```shell
docker compose build borsa-mcp
docker compose run --rm borsa-mcp python -c "import unified_mcp_server; print('ok')"
docker compose run --rm borsa-mcp borsa-mcp
```

## Codex/MCP Kullanımı

Container içinde `borsa-mcp` stdio MCP server olarak başlar. Codex veya başka MCP istemcisinde komut olarak `docker compose run --rm borsa-mcp borsa-mcp` kullanılabilir. Uzun yaşayan servis için `docker compose up borsa-mcp` çalıştırılır.

## alerts.csv

Kolonlar:

- `timestamp_utc`
- `symbol`
- `last_price`
- `current_volume`
- `avg_20d_volume`
- `day_ratio`
- `volume_5m_delta`
- `velocity_ratio`
- `reason`
- `dry_run`

Bu proje yatırım tavsiyesi değildir. Sadece araştırma, veri izleme ve analiz aracıdır.
