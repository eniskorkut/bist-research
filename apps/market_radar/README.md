# BIST Pozitif Ilgi Radari

Pozitif hacim, TL hacim, fiyat ve trend hareketi ile BIST hisselerinde ilgi artisini tarar.

Bu modul standalone Streamlit servisi olarak calistirilmaz. Unified dashboard icinde import edilir ve `valuation-app` uzerinden acilir.
Tarama evreni manuel hisse listesi degildir; varsayilan olarak borsapy `XUTUM` bilesenlerinden gelen tum Borsa Istanbul hisseleri taranir.

## Evren Cache

XUTUM sembol listesi `/data/market_radar_cache.sqlite` icinde `universe_cache` tablosunda saklanir.

- **TTL**: 24 saat.
- **Fresh cache** varsa borsapy'ye tekrar sorulmaz.
- **Stale cache** varsa ve borsapy hata verirse stale cache fallback olarak kullanilir.
- `--force` veya "Cache'i yenile" kullanildiginda cache goz ardi edilir.
- UI'da evren kaynagi (`fresh_cache` / `stale_cache` / `borsapy`) ve sembol sayisi gosterilir.

## Hata Toleransi

- Tekil sembol hatasi tum taramayi durdurmaz.
- Hata veren semboller `failed_symbols` listesinde toplanir.
- Basarili semboller sonuc uretmeye devam eder.
- Streamlit'te "Hata veren semboller" expander'i icinde gosterilir.
- CLI ciktisinda `failed_symbols` ozet ve detayi yazdirilir.

## Tarama Ozeti

Tarama sonrasi asagidaki metrikler gosterilir:
- `universe_symbol_count`: Evrende kac sembol var
- `scanned_symbols`: Kac sembol tarandi (basarili + hatali)
- `successful_symbols`: Kac sembol basarili tarandi
- `failed_symbols`: Kac sembol hata verdi
- `result_count`: Filtrelerden gecen sonuc sayisi
- `cache_source`: Evren kaynagi

## Calistirma

```bash
docker compose build valuation-app
docker compose up valuation-app
```

Tarayici:

```text
http://localhost:8502
```

Dashboard sidebar'inda `Pozitif Ilgi Radari` sayfasini sec.

## CLI

```bash
docker compose run --rm valuation-app python apps/market_radar/scripts/scan_bist_interest.py \
  --min-score 40 \
  --force
```

Sadece hacim ortalamasını geçenler:

```bash
docker compose run --rm valuation-app python apps/market_radar/scripts/scan_bist_interest.py \
  --index XUTUM \
  --lookback-days 60 \
  --min-volume-ratio 1.0 \
  --min-turnover-ratio 0 \
  --min-daily-return -100 \
  --include-negative-moves \
  --no-require-above-ma20 \
  --no-require-xu100-relative \
  --no-require-close-position \
  --breakout-mode off \
  --no-require-min-score \
  --max-workers 8
```

Pozitif hacim artışı:

```bash
docker compose run --rm valuation-app python apps/market_radar/scripts/scan_bist_interest.py \
  --index XUTUM \
  --lookback-days 60 \
  --min-volume-ratio 1.5 \
  --min-turnover-ratio 1.2 \
  --min-daily-return 0 \
  --no-require-above-ma20 \
  --no-require-xu100-relative \
  --no-require-close-position \
  --breakout-mode off \
  --min-score 30 \
  --max-workers 8
```

Pozitif para girişi:

```bash
docker compose run --rm valuation-app python apps/market_radar/scripts/scan_bist_interest.py \
  --index XUTUM \
  --lookback-days 60 \
  --scan-mode positive_money_flow \
  --max-workers 8
```

Sessiz toplama:

```bash
docker compose run --rm valuation-app python apps/market_radar/scripts/scan_bist_interest.py \
  --index XUTUM \
  --lookback-days 60 \
  --scan-mode silent_accumulation \
  --max-workers 8
```
