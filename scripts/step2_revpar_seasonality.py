"""
Step 2 - RevPAR/ADR mensili + scomposizione stagionale (STL).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL

N_ROOMS = 60
START = pd.Timestamp("2024-09-10")
TODAY = pd.Timestamp("2026-09-10")
HIGH_SEASON_MONTHS = {6, 7, 8, 9, 12}

# palette (dataviz skill - reference palette)
COL_ADR = "#2a78d6"       # categorical slot 1 - blue
COL_REVPAR = "#1baf7a"    # categorical slot 3 - aqua
COL_SEASON_BAND = "#c3c2b7"  # neutral baseline, usato solo come sfondo a bassa opacita'
COL_TEXT = "#0b0b0b"
COL_MUTED = "#898781"
COL_GRID = "#e1e0d9"

# --------------------------------------------------------------- LOAD -----
df = pd.read_csv("../data/bookings_clean.csv", parse_dates=["check_in", "check_out"])
valid = df[(~df["cancelled"]) & (df["nights"] > 0)].copy()

# ------------------------------------------------- ESPANSIONE ROOM-NIGHT ---
# ogni notte di soggiorno diventa una riga: data, ricavo di quella notte
rows = []
for r in valid.itertuples(index=False):
    nights_dates = pd.date_range(r.check_in, periods=r.nights, freq="D")
    rows.append(pd.DataFrame({"date": nights_dates, "revenue": r.room_rate_eur}))
room_nights = pd.concat(rows, ignore_index=True)
room_nights["month"] = room_nights["date"].values.astype("datetime64[M]")

monthly = room_nights.groupby("month").agg(
    revenue_eur=("revenue", "sum"),
    occupied_room_nights=("revenue", "count"),
).reset_index()

# giorni disponibili per mese, clippati al periodo effettivo dei dati [START, TODAY]
def days_available(month_start: pd.Timestamp) -> int:
    month_end = month_start + pd.offsets.MonthEnd(0)
    lo = max(month_start, START)
    hi = min(month_end, TODAY)
    return max(0, (hi - lo).days + 1)

monthly["giorni_disponibili"] = monthly["month"].apply(days_available)
monthly["mese_completo"] = monthly["month"].apply(
    lambda m: days_available(m) == (m + pd.offsets.MonthEnd(0) - m).days + 1
)
monthly["available_room_nights"] = monthly["giorni_disponibili"] * N_ROOMS

monthly["ADR"] = (monthly["revenue_eur"] / monthly["occupied_room_nights"]).round(2)
monthly["RevPAR"] = (monthly["revenue_eur"] / monthly["available_room_nights"]).round(2)
monthly["occupazione"] = (monthly["occupied_room_nights"] / monthly["available_room_nights"]).round(3)
monthly["alta_stagione"] = monthly["month"].dt.month.isin(HIGH_SEASON_MONTHS)

monthly.to_csv("../data/monthly_revpar_adr.csv", index=False)

print("=== ADR / RevPAR mensili (tutti i mesi, inclusi quelli parziali ai bordi) ===")
print(monthly[["month", "mese_completo", "occupazione", "ADR", "RevPAR"]].to_string(index=False))

# --------------------------------------------- SOLO MESI COMPLETI PER STL --
complete = monthly[monthly["mese_completo"]].sort_values("month").reset_index(drop=True)
print(f"\nMesi completi usati per la scomposizione stagionale: {len(complete)} "
      f"({complete.month.min().date()} -> {complete.month.max().date()})")
print(f"Mesi parziali esclusi: "
      f"{sorted(monthly.loc[~monthly.mese_completo, 'month'].dt.date.tolist())}")

ts = complete.set_index("month")["RevPAR"]
ts.index.freq = "MS"

stl = STL(ts, period=12, robust=True)
res = stl.fit()

# ---------------------------------------------------------------- ANOMALIE
resid = res.resid
soglia = 2 * resid.std()
anomalie = resid[resid.abs() > soglia]
print(f"\nDeviazione standard residui: {resid.std():.2f} EUR")
print(f"Soglia anomalia (2 sigma): +/- {soglia:.2f} EUR")
if len(anomalie):
    print("Mesi con residuo anomalo:")
    print(anomalie.round(2).to_string())
else:
    print("Nessun mese supera la soglia di 2 sigma sui residui.")

ampiezza_stagionale = res.seasonal.max() - res.seasonal.min()
trend_start, trend_end = res.trend.dropna().iloc[0], res.trend.dropna().iloc[-1]
var_spiegata_stagionalita = res.seasonal.var() / ts.var()

print(f"\nAmpiezza componente stagionale (picco-valle): {ampiezza_stagionale:.2f} EUR di RevPAR")
print(f"Trend: {trend_start:.2f} -> {trend_end:.2f} EUR ({(trend_end/trend_start-1)*100:+.1f}%)")
print(f"Quota di varianza spiegata dalla sola stagionalita': {var_spiegata_stagionalita:.1%}")

# ------------------------------------------------------------- PLOT 1 -----
# ADR/RevPAR mensili con bande di alta stagione (grafico standalone)
fig0, ax0 = plt.subplots(figsize=(11, 4.5), facecolor="#fcfcfb")
ax0.set_facecolor("#fcfcfb")

for _, row in monthly.iterrows():
    if row["alta_stagione"]:
        m0 = row["month"]
        m1 = m0 + pd.offsets.MonthBegin(1)
        ax0.axvspan(m0, m1, color=COL_SEASON_BAND, alpha=0.35, lw=0)

ax0.plot(monthly["month"], monthly["ADR"], color=COL_ADR, lw=2, marker="o",
         markersize=4, label="ADR (tariffa media giornaliera)")
ax0.plot(monthly["month"], monthly["RevPAR"], color=COL_REVPAR, lw=2, marker="o",
         markersize=4, label="RevPAR (ricavo per camera disponibile)")

ax0.set_title("ADR e RevPAR mensili - Splendido fittizio, 60 camere\n"
              "(sfondo grigio = mesi di alta stagione: giu-set e dic)",
              color=COL_TEXT, fontsize=12, loc="left", pad=12)
ax0.set_ylabel("EUR / camera / notte", color=COL_MUTED, fontsize=9)
ax0.grid(axis="y", color=COL_GRID, lw=0.8)
ax0.spines[["top", "right"]].set_visible(False)
ax0.spines[["left", "bottom"]].set_color(COL_MUTED)
ax0.tick_params(colors=COL_MUTED, labelsize=8)
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
ax0.legend(frameon=False, fontsize=8, loc="upper left")
fig0.tight_layout()
fig0.savefig("../charts/chart1_revpar_adr_stagionale.png", dpi=160, bbox_inches="tight", facecolor=fig0.get_facecolor())
print("Grafico salvato in chart1_revpar_adr_stagionale.png")

# ------------------------------------------------------------- PLOT 2 -----
# Scomposizione STL: observed, trend, seasonal, resid (grafico standalone)
fig2 = plt.figure(figsize=(11, 8), facecolor="#fcfcfb")
gs2 = fig2.add_gridspec(4, 1, hspace=0.35)

labels = ["RevPAR osservato", "Trend", "Stagionalita'", "Residuo"]
series = [ts, res.trend, res.seasonal, res.resid]
for i, (lab, s) in enumerate(zip(labels, series)):
    ax = fig2.add_subplot(gs2[i])
    ax.set_facecolor("#fcfcfb")
    color = COL_ADR if i == 0 else (COL_REVPAR if i in (1, 2) else "#e34948")
    ax.plot(s.index, s.values, color=color, lw=1.6)
    ax.axhline(0, color=COL_MUTED, lw=0.6) if i == 3 else None
    ax.set_ylabel(lab, color=COL_MUTED, fontsize=8)
    ax.grid(axis="y", color=COL_GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COL_MUTED)
    ax.tick_params(colors=COL_MUTED, labelsize=7)
    if i < 3:
        ax.set_xticklabels([])
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

fig2.suptitle("Scomposizione STL del RevPAR mensile (mesi completi, "
              f"{complete.month.min().strftime('%b %Y')} - {complete.month.max().strftime('%b %Y')})",
              color=COL_TEXT, fontsize=11, y=0.995)

fig2.savefig("../charts/chart2_stl_decomposition.png", dpi=160, bbox_inches="tight", facecolor=fig2.get_facecolor())
print("Grafico salvato in chart2_stl_decomposition.png")
