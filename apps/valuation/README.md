# valuation

BIST icin otomatik degerleme modulu.

Kullanici sadece hisse kodu girer. Manuel net kar/ciro/marj girisi yoktur. Veriler `borsapy` uzerinden cekilir.

## Ozellikler

- Otomatik net kar tahmini (`profit_estimator.py`)
- 5 yontemli hedef fiyat tablosu (`valuation_engine.py`)
- Streamlit arayuz (`streamlit_app.py`)
- Sembol smoke testi (`scripts/test_symbols.py`)
- Unit testler (`tests/`)

## Kurulum

```shell
cd apps/valuation
python -m venv .venv
. .venv/bin/activate
pip install -e .
pip install pytest
```

## Calistirma

```shell
streamlit run src/valuation/streamlit_app.py
```

## Test

```shell
python scripts/test_symbols.py
pytest
```

Her cikti yatirim tavsiyesi degildir. Veri eksikse tahmin uydurulmaz, eksik alan raporlanir.
