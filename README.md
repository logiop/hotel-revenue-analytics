# Hotel Revenue Analytics — Revenue Management & Guest Retention

Analisi end-to-end su un anno di prenotazioni di un hotel di lusso fittizio (60 camere, 24 mesi di storico): pulizia dati, revenue management (RevPAR/ADR, scomposizione stagionale), un modello di classificazione per prevedere il ritorno degli ospiti, e un test causale corretto per un confondente stagionale (paradosso di Simpson) sull'effetto di una promo.

Il taglio è volutamente hospitality invece che e-commerce: lavoro come Chef de Rang in hotel di lusso (Splendido, A Belmond Hotel — LVMH — e in precedenza La Posta Vecchia, Pellicano Hotels), e questo progetto nasce per applicare l'analisi statistica a un dominio che conosco da dentro, non solo sulla carta.

**Il dataset è sintetico e generato da codice** (vedi sotto), con problemi di data quality e trappole statistiche inserite di proposito per rendere l'esercizio realistico: omonimie tra ospiti, valute miste, date corrotte, e un vero confondente stagionale sull'effetto di una promo.

## Cosa contiene il repo

```
report.md              # report finale: executive summary, 4 grafici, raccomandazioni, limiti
scripts/                # pipeline in 5 step, eseguibili in ordine
  generate_data.py          # step 0 - genera il dataset sintetico
  clean_and_merge.py        # step 1 - pulizia, matching nomi cross-fonte, conversione valuta
  step2_revpar_seasonality.py  # step 2 - RevPAR/ADR mensili + scomposizione STL
  step3_repeat_guest_model.py  # step 3 - segmentazione ospiti + logistic regression vs random forest
  step4_promo_test.py          # step 4 - chi-quadrato + regressione logistica sull'effetto promo
  chart4_promo_effect.py       # grafico di confronto delle stime dell'effetto promo
data/                   # CSV grezzi e puliti prodotti da ogni step
charts/                 # i 4 grafici chiave, referenziati in report.md
requirements.txt
```

## La pipeline, in breve

1. **Generazione dati** — 3.808 prenotazioni sintetiche, 3.300 ospiti, con nomi scritti in modo inconsistente tra fonti, valute miste (EUR/USD), date corrotte, e ~15% di ospiti volutamente omonimi (persone diverse, stesso nome — un vero problema di entity resolution).
2. **Pulizia** — normalizzazione nomi per il join cross-fonte, conversione valuta, validazione/correzione date, gestione cancellazioni. Report di data quality prima/dopo.
3. **RevPAR/ADR** — metriche mensili di revenue management, scomposizione STL (trend/stagionalità/residuo). La stagionalità spiega il ~97% della varianza mese su mese.
4. **Segmentazione e modello di ritorno** — ospiti one-time vs repeat (entro 18 mesi, con correzione esplicita per censura), Logistic Regression vs Random Forest, valutati su precision/recall/AUC (non solo accuracy, viste le classi sbilanciate).
5. **Test causale sulla promo** — chi-quadrato aggregato vs segmentato per stagione vs regressione logistica di controllo, per isolare l'effetto vero della promo booking-diretto da un confondente stagionale.

Risultati completi, grafici e raccomandazioni nel [report finale](report.md).

## Un bug trovato e corretto lungo il percorso

La prima versione del generatore aveva un problema nel meccanismo "repeat guest": sceglieva a ogni prenotazione un ospite a caso tra quelli già visti, e con un pool che si esauriva in fretta questo produceva un tasso di ritorno artificiale del 98,6% — un target completamente degenere, con l'AUC del modello che scendeva sotto 0,5. L'ho notato controllando i numeri invece di fidarmi ciecamente dell'output, e ho corretto il generatore perché il ritorno di ogni ospite fosse deciso una volta sola, con una probabilità legata a segnali realistici (canale di prenotazione, stagione, durata del soggiorno). Il tasso di ritorno finale nel dataset è 25,3%, e il modello ottiene un AUC di 0,49-0,54 — onestamente debole, ma su un target che ha senso.

## Come riprodurlo

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd scripts
python generate_data.py
python clean_and_merge.py
python step2_revpar_seasonality.py
python step3_repeat_guest_model.py
python step4_promo_test.py
python chart4_promo_effect.py
```

Tutti i passaggi usano un seed fisso (`numpy.random.default_rng(42)`), quindi l'output è deterministico e riproducibile.

## Stack

Python, pandas, numpy, scipy, statsmodels (STL, regressione logistica), scikit-learn (Logistic Regression, Random Forest), matplotlib/seaborn.

## Autore

Giorgio Vernarecci — [@logiop](https://github.com/logiop)

## Licenza

MIT — vedi [LICENSE](LICENSE).
