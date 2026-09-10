"""
Step 4 - Il canale di prenotazione (direct/OTA) e' un attributo della
prenotazione stessa, non dell'identita' dell'ospite: uso tutte le
prenotazioni pulite (inclusa la quota con nome_ambiguo, che riguarda solo
l'attribuzione all'ospite, non il canale/la data/la stagione).
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf

df = pd.read_csv("../data/bookings_clean.csv", parse_dates=["check_in", "check_out"])
df["post_promo"] = df["post_promo"].astype(bool)
df["high_season"] = df["high_season"].astype(bool)
df["direct"] = (df["channel"] == "direct").astype(int)

print(f"Prenotazioni totali analizzate: {len(df)}")
print(f"Quota diretta complessiva: {df['direct'].mean():.1%}\n")


def run_chi2(sub: pd.DataFrame, label: str):
    ct = pd.crosstab(sub["post_promo"], sub["channel"])
    prop = pd.crosstab(sub["post_promo"], sub["channel"], normalize="index")
    chi2, p, dof, _ = chi2_contingency(ct)
    pre = prop.loc[False, "direct"]
    post = prop.loc[True, "direct"]
    print(f"--- {label} (n={len(sub)}) ---")
    print(f"Quota diretta PRE-promo:  {pre:.1%}  (n={ct.loc[False].sum()})")
    print(f"Quota diretta POST-promo: {post:.1%}  (n={ct.loc[True].sum()})")
    print(f"Differenza: {(post-pre)*100:+.1f} punti percentuali")
    print(f"Chi-quadrato = {chi2:.2f}, p-value = {p:.4f}")
    print()
    return pre, post, p


# ------------------------------------------------------- 1. AGGREGATO -----
print("=" * 70)
print("1. CONFRONTO AGGREGATO (senza controllare per stagione)")
print("=" * 70)
pre_agg, post_agg, p_agg = run_chi2(df, "Tutte le prenotazioni")

# -------------------------------------------------- 2. SEGMENTATO x STAGIONE
print("=" * 70)
print("2. CONFRONTO SEGMENTATO PER STAGIONE")
print("=" * 70)
pre_high, post_high, p_high = run_chi2(df[df.high_season], "Alta stagione")
pre_low, post_low, p_low = run_chi2(df[~df.high_season], "Bassa stagione")

# quanto e' cambiato il MIX stagionale tra pre e post (il confondente)
mix = df.groupby("post_promo")["high_season"].mean()
print("--- Il confondente: quota di prenotazioni in alta stagione ---")
print(f"Pre-promo:  {mix.loc[False]:.1%} delle prenotazioni sono in alta stagione")
print(f"Post-promo: {mix.loc[True]:.1%} delle prenotazioni sono in alta stagione")
print(f"(la finestra post-promo, gli ultimi 6 mesi, contiene strutturalmente "
      f"piu' mesi di alta stagione della finestra pre-promo, 18 mesi)\n")

# ---------------------------------------- 3. REGRESSIONE LOGISTICA (controllo)
print("=" * 70)
print("3. REGRESSIONE LOGISTICA - effetto della promo al netto della stagione")
print("=" * 70)

model = smf.logit("direct ~ post_promo + high_season", data=df).fit(disp=0)
print(model.summary())

# effetto marginale medio (AME) di post_promo, il modo piu' interpretabile
# di leggere "quanti punti percentuali" al netto della stagione
mfx = model.get_margeff(at="overall")
print("\n--- Effetti marginali medi (punti percentuali di probabilita' di booking diretto) ---")
print(mfx.summary())

coef_promo = model.params["post_promo[T.True]"]
odds_ratio = np.exp(coef_promo)
ame_promo = mfx.margeff[list(mfx.margeff_names).index("post_promo[T.True]") if hasattr(mfx, "margeff_names") else 0]

print(f"\nOdds ratio della promo (al netto della stagione): {odds_ratio:.3f}")
print(f"p-value coefficiente post_promo: {model.pvalues['post_promo[T.True]']:.4f}")

# ------------------------------------------------------------- RIEPILOGO --
print("\n" + "=" * 70)
print("RIEPILOGO CONFRONTO STIME DELL'EFFETTO PROMO")
print("=" * 70)
print(f"Aggregato (NON controllato):         {(post_agg-pre_agg)*100:+.1f} pp  (p={p_agg:.4f})")
print(f"Alta stagione, pre vs post:           {(post_high-pre_high)*100:+.1f} pp  (p={p_high:.4f})")
print(f"Bassa stagione, pre vs post:           {(post_low-pre_low)*100:+.1f} pp  (p={p_low:.4f})")
print(f"Regressione logistica (controllato per stagione), effetto marginale medio "
      f"di post_promo su P(diretto): vedi output sopra")
