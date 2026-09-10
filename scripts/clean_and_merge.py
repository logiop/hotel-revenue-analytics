"""
Step 1 - Pulizia e unione dei tre dataset in un unico dataframe analitico.

Produce:
  - bookings_clean.csv  (dataframe analitico pronto per gli step successivi)
  - stampa un report di data quality prima/dopo
"""

import numpy as np
import pandas as pd

FX_USD_TO_EUR = 0.92          # tasso fisso plausibile usato per la conversione
HIGH_SEASON_MONTHS = {6, 7, 8, 9, 12}
PROMO_LAUNCH = pd.Timestamp("2026-03-12")

# ---------------------------------------------------------------- LOAD ----
bookings = pd.read_csv("../data/bookings.csv", parse_dates=["check_in", "check_out"])
guests = pd.read_csv("../data/guests.csv", parse_dates=["prima_visita"])
promo = pd.read_csv("../data/promo_log.csv", parse_dates=["data_lancio"])

report = {"prima": {}, "dopo": {}}
report["prima"]["righe_bookings"] = len(bookings)
report["prima"]["righe_guests"] = len(guests)

# ------------------------------------------------------ NORMALIZZA NOMI ---
def normalize_name(name: str) -> str:
    """Case, spazi, ordine nome/cognome -> chiave di join stabile."""
    if pd.isna(name):
        return ""
    tokens = str(name).strip().lower().split()
    return " ".join(sorted(tokens))

bookings["guest_name_norm"] = bookings["guest_name"].apply(normalize_name)
guests["guest_name_norm"] = guests["guest_name"].apply(normalize_name)

# quante varianti di scrittura diverse mappano sullo stesso ospite normalizzato
varianti_bookings = bookings.groupby("guest_name_norm")["guest_name"].nunique()
report["prima"]["nomi_univoci_grezzi_bookings"] = bookings["guest_name"].nunique()
report["dopo"]["nomi_univoci_normalizzati_bookings"] = bookings["guest_name_norm"].nunique()

# ATTENZIONE - problema reale di entity resolution: bookings.csv ha come
# unico attributo per il join il nome, e nel dataset esistono OMONIMI veri
# (persone diverse con stesso nome, email diversa). Senza un ID univoco o
# l'email anche in bookings.csv, il join per nome resta strutturalmente
# ambiguo: lo quantifico invece di nasconderlo.
nomi_ambigui_set = set(
    guests.loc[guests["guest_name_norm"].duplicated(keep=False), "guest_name_norm"]
)
report["prima"]["ospiti_coinvolti_in_omonimia"] = int(
    guests["guest_name_norm"].duplicated(keep=False).sum()
)
report["prima"]["quota_ospiti_in_omonimia"] = round(
    guests["guest_name_norm"].duplicated(keep=False).mean(), 3
)

dup_guests = guests["guest_name_norm"].duplicated().sum()
if dup_guests:
    # scelta esplicita e documentata: in caso di omonimia tengo il profilo
    # con la prima_visita piu' antica. E' un best-effort, non una soluzione:
    # i booking di un omonimo "perso" nel dedup potrebbero finire attribuiti
    # al profilo sbagliato -> per questo aggiungo il flag nome_ambiguo sotto.
    guests = guests.sort_values("prima_visita").drop_duplicates("guest_name_norm", keep="first")

# --------------------------------------------------------------- MERGE ----
merged = bookings.merge(
    guests[["guest_name_norm", "email", "country", "prima_visita", "cibo_preferito"]],
    on="guest_name_norm", how="left", validate="m:1",
)

n_non_matchati = merged["email"].isna().sum()
report["dopo"]["booking_non_matchati_a_un_ospite"] = int(n_non_matchati)
report["dopo"]["booking_matchati_a_un_ospite"] = int(len(merged) - n_non_matchati)

# flag: il booking e' stato attribuito a un nome coinvolto in un'omonimia ->
# l'attribuzione al guest_id corretto NON e' garantita, va trattata con cautela
# negli step successivi (es. segmentazione/repeat-guest allo Step 3)
merged["nome_ambiguo"] = merged["guest_name_norm"].isin(nomi_ambigui_set)
report["dopo"]["booking_con_attribuzione_ambigua"] = int(merged["nome_ambiguo"].sum())
report["dopo"]["quota_booking_con_attribuzione_ambigua"] = round(
    merged["nome_ambiguo"].mean(), 3
)

# ---------------------------------------------------------- VALUTA -> EUR -
report["prima"]["righe_in_USD_da_convertire"] = int((merged["currency"] == "USD").sum())
merged["room_rate_eur"] = np.where(
    merged["currency"] == "USD",
    (merged["room_rate"] * FX_USD_TO_EUR).round(2),
    merged["room_rate"],
)
report["dopo"]["righe_convertite_USD_to_EUR"] = report["prima"]["righe_in_USD_da_convertire"]

# ------------------------------------------------ VALIDAZIONE/CORREZIONE DATE
date_invalide = merged["check_out"] <= merged["check_in"]
report["prima"]["righe_con_checkout_non_valido"] = int(date_invalide.sum())

# tariffa media di soggiorno "sana" per stagione, usata per imputare le date rotte
merged["high_season_tmp"] = merged["check_in"].dt.month.isin(HIGH_SEASON_MONTHS)
nights_valid = (merged.loc[~date_invalide, "check_out"] - merged.loc[~date_invalide, "check_in"]).dt.days
mediana_nights_per_stagione = (
    nights_valid.groupby(merged.loc[~date_invalide, "high_season_tmp"]).median()
)

merged["date_corretta"] = False
for is_high, mediana in mediana_nights_per_stagione.items():
    mask = date_invalide & (merged["high_season_tmp"] == is_high)
    merged.loc[mask, "check_out"] = merged.loc[mask, "check_in"] + pd.to_timedelta(mediana, unit="D")
    merged.loc[mask, "date_corretta"] = True

merged["nights"] = (merged["check_out"] - merged["check_in"]).dt.days
report["dopo"]["righe_con_date_corrette"] = int(merged["date_corretta"].sum())
report["dopo"]["righe_con_checkout_non_valido_residue"] = int((merged["check_out"] <= merged["check_in"]).sum())
merged = merged.drop(columns=["high_season_tmp"])

# -------------------------------------------------------- CANCELLAZIONI ---
report["prima"]["prenotazioni_cancellate"] = int(merged["cancelled"].sum())
# le cancellate restano nel dataframe (servono per l'analisi comportamentale)
# ma non generano ricavo
merged["net_revenue_eur"] = np.where(
    merged["cancelled"], 0.0, (merged["room_rate_eur"] * merged["nights"]).round(2)
)
report["dopo"]["prenotazioni_escluse_dal_ricavo"] = report["prima"]["prenotazioni_cancellate"]
report["dopo"]["prenotazioni_incluse_nel_ricavo"] = int(len(merged) - merged["cancelled"].sum())

# -------------------------------------------------------- FLAG DI COMODO --
merged["high_season"] = merged["check_in"].dt.month.isin(HIGH_SEASON_MONTHS)
merged["post_promo"] = merged["check_in"] >= PROMO_LAUNCH

report["dopo"]["righe_finali"] = len(merged)
report["dopo"]["colonne_finali"] = len(merged.columns)

# ------------------------------------------------------------- SALVATAGGIO
cols_out = [
    "booking_id", "guest_name_norm", "email", "country", "cibo_preferito",
    "prima_visita", "check_in", "check_out", "nights", "channel",
    "currency", "room_rate", "room_rate_eur", "cancelled", "net_revenue_eur",
    "high_season", "post_promo", "date_corretta", "nome_ambiguo",
]
merged[cols_out].to_csv("../data/bookings_clean.csv", index=False)

# ------------------------------------------------------------------ PRINT -
print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
print("\n-- PRIMA --")
for k, v in report["prima"].items():
    print(f"  {k}: {v}")
print("\n-- DOPO --")
for k, v in report["dopo"].items():
    print(f"  {k}: {v}")
print("\nAnteprima dataframe pulito:")
print(merged[cols_out].head(3).to_string())
