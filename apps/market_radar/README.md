# BIST Pozitif Ilgi Radari

Pozitif hacim, TL hacim, fiyat ve trend hareketi ile BIST hisselerinde ilgi artisini tarar.

Bu modul standalone Streamlit servisi olarak calistirilmaz. Unified dashboard icinde import edilir ve `valuation-app` uzerinden acilir.
Tarama evreni manuel hisse listesi degildir; varsayilan olarak borsapy `XUTUM` bilesenlerinden gelen tum Borsa Istanbul hisseleri taranir.

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
