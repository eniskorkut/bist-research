# valuation

BIST otomatik degerleme modulu. Kullanici sadece hisse kodu girer. Tum veri `borsapy` kaynagindan cekilir (`source="borsapy"`).

## Neler Yapar

- Otomatik net kar tahmini: `TTM`, `sezonsallik`, `ciro x marj` ve uygun yontem medyani.
- Ana 5 yontemli degerleme tablosu:
  - `cari_fk`
  - `pd_dd`
  - `odenmis_sermaye`
  - `potansiyel_piyasa_degeri`
  - `ozsermaye_karliligi`
- Sektor bazli ek degerleme:
  - `sektor_fk_hedef`
  - `sektor_pd_dd_hedef`
- SQLite cache:
  - `/data/valuation_cache.sqlite`
  - `company_snapshot`, `sector_metrics`, `valuation_results`

## Cache Kurali

- `company_snapshot` ve `sector_metrics` 24 saatten eskiyse `stale`.
- Streamlit varsayilaninda tum XU100 otomatik refresh edilmez.
- `Cache'i yenile` sadece secili hisse + sektorunu yeniler.
- Toplu yenileme CLI ile yapilir.

## Sektor Hesabi

- Sektor F/K medyan: sadece pozitif F/K.
- Sektor F/K aggregate: toplam piyasa degeri / toplam pozitif tahmini net kar.
- Sektor PD/DD medyan: sadece pozitif PD/DD.
- Sektor PD/DD aggregate: toplam piyasa degeri / toplam pozitif ozkaynak.
- Sektor ROE aggregate: toplam tahmini net kar / toplam ozkaynak.

Negatif kar veya negatif ozkaynakta ilgili carpana dayali hedef fiyat hesaplanmaz.

## Komutlar

```shell
docker compose build valuation-app
docker compose run --rm valuation-app python -m pytest
docker compose run --rm valuation-app python scripts/refresh_bist_cache.py --symbols THYAO ASELS GARAN --db-path /data/valuation_cache.sqlite --force
docker compose run --rm valuation-app python scripts/inspect_cache.py --db-path /data/valuation_cache.sqlite --limit 10
docker compose up valuation-app
```

## Gunluk Refresh Cron Ornegi

```cron
30 8 * * 1-5 cd /path/to/bist-research && docker compose run --rm valuation-refresh
```

Eksik veri veya eski veri durumlari UI'da acik uyarilarla gosterilir. Bu calisma yatirim tavsiyesi degildir.
