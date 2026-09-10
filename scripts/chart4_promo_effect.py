"""
Chart 4 - confronto visivo delle stime dell'effetto promo: aggregato (distorto
dal mix stagionale) vs segmentato per stagione vs regressione logistica
controllata. Il messaggio del grafico e' proprio la differenza tra la prima
barra (gonfiata) e le altre (corrette).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

COL_NAIVE = "#eb6834"     # arancio - stima distorta, da evidenziare come "attenzione"
COL_CORRECT = "#2a78d6"   # blu - stime corrette
COL_MUTED = "#898781"
COL_GRID = "#e1e0d9"

df = pd.read_csv("../data/bookings_clean.csv", parse_dates=["check_in", "check_out"])
df["post_promo"] = df["post_promo"].astype(bool)
df["high_season"] = df["high_season"].astype(bool)
df["direct"] = (df["channel"] == "direct").astype(int)


def prop_diff_ci(sub):
    ct = pd.crosstab(sub["post_promo"], sub["channel"], normalize="index")
    counts = pd.crosstab(sub["post_promo"], sub["channel"])
    pre, post = ct.loc[False, "direct"], ct.loc[True, "direct"]
    n_pre, n_post = counts.loc[False].sum(), counts.loc[True].sum()
    diff = post - pre
    se = np.sqrt(pre * (1 - pre) / n_pre + post * (1 - post) / n_post)
    return diff * 100, 1.96 * se * 100


agg_diff, agg_ci = prop_diff_ci(df)
high_diff, high_ci = prop_diff_ci(df[df.high_season])
low_diff, low_ci = prop_diff_ci(df[~df.high_season])

model = smf.logit("direct ~ post_promo + high_season", data=df).fit(disp=0)
mfx = model.get_margeff(at="overall")
reg_diff = mfx.margeff[0] * 100
reg_ci = 1.96 * mfx.margeff_se[0] * 100

labels = [
    "Aggregato\n(NON controllato\nper stagione)",
    "Alta stagione\n(pre vs post)",
    "Bassa stagione\n(pre vs post)",
    "Regressione logistica\n(controllato per\nstagione)",
]
values = [agg_diff, high_diff, low_diff, reg_diff]
errors = [agg_ci, high_ci, low_ci, reg_ci]
colors = [COL_NAIVE, COL_CORRECT, COL_CORRECT, COL_CORRECT]

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
x = np.arange(len(labels))
bars = ax.bar(x, values, yerr=errors, color=colors, width=0.55, capsize=4,
              error_kw={"ecolor": "#52514e", "elinewidth": 1})

for xi, v in zip(x, values):
    ax.text(xi, v + 0.9, f"+{v:.1f} pp", ha="center", fontsize=10, color="#0b0b0b")

ax.axhline(0, color=COL_MUTED, lw=0.8)
ax.set_ylabel("Variazione quota booking diretti\npre -> post promo (punti percentuali)",
              color="#52514e", fontsize=9)
ax.set_title("Effetto della promo sulla quota di booking diretti:\n"
              "il confronto aggregato lo sovrastima di circa il 60%",
              loc="left", fontsize=12, color="#0b0b0b", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5, color="#52514e")
ax.grid(axis="y", color=COL_GRID, lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(COL_MUTED)
ax.tick_params(colors=COL_MUTED)

# legenda manuale (due soli ruoli di colore, non serie multiple)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=COL_NAIVE, label="Stima non corretta (mix stagionale confonde l'effetto)"),
    Patch(facecolor=COL_CORRECT, label="Stime corrette per la stagione"),
], frameon=False, fontsize=8, loc="upper right")

fig.tight_layout()
fig.savefig("../charts/chart4_promo_effect.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Grafico salvato in chart4_promo_effect.png")
print(dict(zip(labels, zip(values, errors))))
