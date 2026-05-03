# BIST Research

Iki parcali BIST arastirma ortami + US research modulu.

## Mimari

- `borsapy-alert`: canli quote/hacim izler, hacim alarmi uretir.
- `borsa-mcp`: Codex/LLM tarafina MCP finansal veri tool server saglar.
- `apps/us_research`: ABD hisseleri icin OpenBB + edgartools arastirma modulu.
- `openbb-mcp`: OpenBB endpointlerini MCP server olarak sunar.
- Telegram yok. Al-sat yok. Sadece arastirma, veri izleme ve analiz.

## borsapy-alert

`borsapy` ile `TradingViewStream` baglantisi acar. `XU100`, `XU030`, `XBANK` gibi endekslerin bilesenlerini veya elle verilen sembolleri izler. Her sembol icin `bp.Ticker(symbol).history(period="1ay")` uzerinden son 20 gunluk ortalama hacmi hesaplar.

Alarm kriterleri:

- Gun ici toplam hacim / 20 gunluk ortalama hacim >= `1.5`
- Son 5 dakikalik hacim artisi / 20 gunluk ortalama hacim >= `0.05`
- Ayni sembol icin cooldown: 15 dakika

Alarm terminale yazilir ve `alerts.csv` dosyasina append edilir. `--dry-run` sadece alarm satirini dry-run olarak isaretler.

## borsa-mcp

`saidsurucu/borsa-mcp` reposundan kurulur. MCP server `borsa-mcp` komutu ile baslar. Canli alarm burada yapilmaz. Gecmis veri, bilanco, teknik analiz, KAP haberleri ve dogal dil tool cagrilari icin kullanilir.

## US Research

`apps/us_research` OpenBB ve edgartools kullanir:

- OpenBB (`openbb`, `openbb-yfinance`): ABD OHLCV, fiyat ve hacim verisi.
- `edgartools`: SEC EDGAR filing ve finansal tablolar (10-K, 10-Q, 8-K, XBRL).
- OpenBB yfinance provider icin API key gerekmez.
- SEC icin API key yerine `EDGAR_IDENTITY` gerekir.

## Docker Compose

```shell
cp .env.example .env
docker compose build
docker compose up
```

Varsayilan servis komutlari:

```shell
python scripts/run_volume_alert.py --index XU030 --dry-run --check-interval 5
borsa-mcp
python scripts/test_openbb.py
openbb-mcp --transport streamable-http --port 8001 --host 0.0.0.0
```

`alerts.csv` host dosyasi olarak `./borsapy-alert/alerts.csv` konumuna baglidir.

## Local Test Komutlari

```shell
cd borsapy-alert
python -m venv .venv
. .venv/bin/activate
pip install -e .
python scripts/run_volume_alert.py --index XU030 --dry-run --duration 300 --max-symbols 5
python scripts/run_volume_alert.py --symbols THYAO ASELS GARAN --dry-run --duration 120
```

BIST sembol/history hizli kontrol:

```shell
python - <<'PY'
import borsapy as bp
print(bp.Index("XU030").component_symbols)
print(bp.Ticker("THYAO").history(period="1ay").tail())
PY
```

borsa-mcp hizli kontrol:

```shell
docker compose build borsa-mcp
docker compose run --rm borsa-mcp python -c "import unified_mcp_server; print('ok')"
docker compose run --rm borsa-mcp borsa-mcp
```

US hizli kontrol:

```shell
docker compose build us-research
docker compose run --rm us-research python scripts/test_openbb.py
docker compose run --rm us-research python scripts/test_edgartools.py
docker compose run --rm us-research python scripts/scan_us_volume.py --symbols AAPL NVDA MSFT TSLA --lookback 20 --min-ratio 1.5 --start-date 2024-01-01
```

OpenBB MCP test:

```shell
docker compose build openbb-mcp
docker compose up openbb-mcp
```

## Codex/MCP Kullanimi

Container icinde `borsa-mcp` stdio MCP server olarak baslar. Codex veya baska MCP istemcisinde komut olarak `docker compose run --rm borsa-mcp borsa-mcp` kullanilabilir.

`openbb-mcp` servisi OpenBB + `openbb-yfinance` + `openbb-mcp-server` ile ayakta kalir. Varsayilan olarak `8001` portunda `streamable-http` transport ile dinler.

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

Ucretsiz veri kaynaklari gecikmeli veya eksik olabilir. Bu proje yatirim tavsiyesi degildir. Sadece arastirma, veri izleme ve analiz aracidir.
