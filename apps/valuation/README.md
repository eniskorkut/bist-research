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

## Cache Mimarisi

Cache, degerleme hesaplamasinin **ana veri kaynagidir**.

### Akis

1. Kullanici "Analiz" butonuna tikladiginda ilk olarak cache kontrol edilir.
2. `company_snapshot` ve `sector_metrics` tablolari sorgulanir.
3. Her iki kayit da **fresh** (24 saatten yeni) ise:
   - **borsapy cagirilmaz**.
   - `run_valuation_from_snapshot(cached, sector_metrics, db_path)` ile tamamen cache uzerinden degerleme yapilir.
4. `company_snapshot` stale veya missing ise:
   - Secili hisse **ve sektoru** borsapy'den cekilir, cache guncellenir.
5. `sector_metrics` stale veya missing ama `company_snapshot` fresh ise:
   - Sektor uyelerinden fresh olanlar cache'den alinir (borsapy cagirilmaz).
   - Sadece stale/missing sektor uyeleri borsapy'den cekilir.
   - Sektor metrikleri tum uyeler uzerinden yeniden hesaplanir.

### Cache Suresi

- `company_snapshot`: 24 saat (`is_stale()` fonksiyonu ile kontrol).
- `sector_metrics`: 24 saat (`calculated_at` alani ile kontrol).

### `--force` Flagi

`refresh_bist_cache.py` scriptinde `--force` kullanildiginda:
- Tum semboller icin cache suresi goz ardi edilir.
- Veriler borsapy'den yeniden cekilir.
- `--force` olmadan calistirildiginda fresh semboller **skip** edilir.

### Cache Durumu (UI)

Streamlit arayuzunde "Cache Durumu" bolumunde su bilgiler gosterilir:
- `company_cache_status`: fresh / stale / missing
- `sector_cache_status`: fresh / stale / missing
- `valuation_source`: cache / borsapy_refresh
- `data_quality_status`: usable / partial / unusable

`data_quality_status=unusable` olan kayitlar fresh olsa bile degerleme icin kullanilmaz.

## Sektor Hesabi

- Sektor F/K medyan: sadece pozitif F/K.
- Sektor F/K aggregate: toplam piyasa degeri / toplam tahmini net kar (sadece `estimated_net_income > 0` ve `market_cap > 0` olan sirketler).
- Sektor PD/DD medyan: sadece pozitif PD/DD.
- Sektor PD/DD aggregate: toplam piyasa degeri / toplam ozkaynak (sadece `equity > 0` ve `market_cap > 0` olan sirketler).
- Sektor ROE aggregate: toplam tahmini net kar / toplam ozkaynak (sadece `equity > 0` olan sirketler).

Her aggregate metrikte pay ve payda ayni gecerli sirket setinden hesaplanir.

Negatif kar veya negatif ozkaynakta ilgili carpana dayali hedef fiyat hesaplanmaz.
Negatif net karda sistem hedef fiyat uretmez; kurs modeline gore pozitif gelecek/yil sonu net kar tahmini varsa ayri senaryoda degerleme yapilir.

## Komutlar

```shell
docker compose build valuation-app
docker compose run --rm valuation-app python -m pytest
docker compose run --rm valuation-app python scripts/debug_borsapy_symbol.py THYAO

# Zorla yenile (fresh cache'i goz ardi et)
docker compose run --rm valuation-app python scripts/refresh_bist_cache.py --symbols THYAO ASELS GARAN --db-path /data/valuation_cache.sqlite --force

# Normal calistirma (fresh semboller skip edilir)
docker compose run --rm valuation-app python scripts/refresh_bist_cache.py --symbols THYAO ASELS GARAN --db-path /data/valuation_cache.sqlite

docker compose run --rm valuation-app python scripts/inspect_cache.py --db-path /data/valuation_cache.sqlite --limit 10
docker compose up valuation-app
```

## Gunluk Refresh Cron Ornegi

```cron
30 8 * * 1-5 cd /path/to/bist-research && docker compose run --rm valuation-refresh
```

Eksik veri veya eski veri durumlari UI'da acik uyarilarla gosterilir. Bu calisma yatirim tavsiyesi degildir.
