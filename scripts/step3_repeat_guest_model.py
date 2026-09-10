"""
Step 3 - Segmentazione ospiti one-time vs repeat (entro 18 mesi dalla prima
visita) + modello di classificazione (Logistic Regression vs Random Forest).

Scelte metodologiche esplicite (documentate anche nel report):
  1. Escludo le prenotazioni cancellate: una "visita" e' un soggiorno
     effettivo, non una prenotazione mai avvenuta.
  2. Escludo le prenotazioni con nome_ambiguo=True (omonimie non risolvibili
     - vedi Step 1): non voglio costruire un modello su identita' incerte.
  3. CENSURA: includo nel training solo gli ospiti la cui prima visita e'
     avvenuta almeno 18 mesi prima della fine del periodo osservato
     (10/09/2026). Per chi ha visitato per la prima volta piu' di recente,
     non abbiamo ancora una finestra di osservazione completa: etichettarli
     "non tornati" sarebbe un bias di censura classico (sottostimerebbe
     sistematicamente il tasso di ritorno dei nuovi ospiti).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score, roc_curve,
)
from sklearn.inspection import permutation_importance

RNG_SEED = 42
TODAY = pd.Timestamp("2026-09-10")
CENSOR_CUTOFF = TODAY - pd.DateOffset(months=18)  # 2025-03-10
RETURN_WINDOW = pd.DateOffset(months=18)

COL_1 = "#2a78d6"   # blue
COL_2 = "#eb6834"   # orange

# --------------------------------------------------------------- LOAD -----
df = pd.read_csv("../data/bookings_clean.csv", parse_dates=["check_in", "check_out"])

visits = df[(~df["cancelled"]) & (~df["nome_ambiguo"])].copy()
print(f"Soggiorni effettivi utilizzabili (non cancellati, nome non ambiguo): "
      f"{len(visits)} su {len(df)} righe totali")

visits = visits.sort_values(["guest_name_norm", "check_in"])
first_visit = visits.groupby("guest_name_norm").first().reset_index()
first_visit = first_visit.rename(columns={
    "check_in": "prima_visita_effettiva", "room_rate_eur": "adr_prima_visita",
    "nights": "durata_prima_visita", "channel": "canale_prima_visita",
    "high_season": "alta_stagione_prima_visita", "country": "paese",
})

n_pre_censura = len(first_visit)
first_visit = first_visit[first_visit["prima_visita_effettiva"] <= CENSOR_CUTOFF].copy()
print(f"Ospiti totali (nomi non ambigui): {n_pre_censura}")
print(f"Esclusi per censura (prima visita dopo il {CENSOR_CUTOFF.date()}, "
      f"finestra di 18 mesi non ancora completa): {n_pre_censura - len(first_visit)}")
print(f"Ospiti utilizzati per il modello: {len(first_visit)}")

# ---------------------------------------------------------- TARGET: repeat
visit_counts_after = []
for row in first_visit.itertuples(index=False):
    later = visits[
        (visits["guest_name_norm"] == row.guest_name_norm)
        & (visits["check_in"] > row.prima_visita_effettiva)
        & (visits["check_in"] <= row.prima_visita_effettiva + RETURN_WINDOW)
    ]
    visit_counts_after.append(len(later))
first_visit["repeat"] = (np.array(visit_counts_after) > 0).astype(int)

tasso_repeat = first_visit["repeat"].mean()
print(f"\nTasso di ritorno entro 18 mesi: {tasso_repeat:.1%} "
      f"({first_visit['repeat'].sum()} repeat su {len(first_visit)} ospiti)")

first_visit.to_csv("../data/guest_segments.csv", index=False)

# ------------------------------------------------------------ FEATURES ----
feat_num = ["adr_prima_visita", "durata_prima_visita"]
feat_cat = ["canale_prima_visita", "alta_stagione_prima_visita", "paese"]

X = pd.get_dummies(first_visit[feat_num + feat_cat], columns=feat_cat, drop_first=False)
y = first_visit["repeat"]

# mappa: nome colonna one-hot -> feature originale (per aggregare l'importanza dopo)
feature_group = {}
for c in X.columns:
    if c in feat_num:
        feature_group[c] = c
    else:
        for base in feat_cat:
            if c.startswith(base):
                feature_group[c] = base

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RNG_SEED, stratify=y
)
print(f"\nTrain: {len(X_train)} ospiti ({y_train.mean():.1%} repeat) | "
      f"Test: {len(X_test)} ospiti ({y_test.mean():.1%} repeat)")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# --------------------------------------------------------- LOGISTIC REG ---
logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RNG_SEED)
logreg.fit(X_train_s, y_train)
proba_lr = logreg.predict_proba(X_test_s)[:, 1]
pred_lr = logreg.predict(X_test_s)

results["Logistic Regression"] = {
    "precision": precision_score(y_test, pred_lr),
    "recall": recall_score(y_test, pred_lr),
    "f1": f1_score(y_test, pred_lr),
    "auc": roc_auc_score(y_test, proba_lr),
}

# ------------------------------------------------------------ RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=400, max_depth=5, min_samples_leaf=5,
    class_weight="balanced", random_state=RNG_SEED,
)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]
pred_rf = rf.predict(X_test)

results["Random Forest"] = {
    "precision": precision_score(y_test, pred_rf),
    "recall": recall_score(y_test, pred_rf),
    "f1": f1_score(y_test, pred_rf),
    "auc": roc_auc_score(y_test, proba_rf),
}

print("\n=== Confronto modelli (test set, classi sbilanciate: precision/recall/F1/AUC) ===")
res_df = pd.DataFrame(results).T.round(3)
print(res_df.to_string())

print("\n--- Classification report: Logistic Regression ---")
print(classification_report(y_test, pred_lr, target_names=["one-time", "repeat"]))
print("--- Classification report: Random Forest ---")
print(classification_report(y_test, pred_rf, target_names=["one-time", "repeat"]))

# ------------------------------------------ FEATURE IMPORTANCE (permutation)
def aggregated_permutation_importance(model, X_eval, y_eval, use_scaled=False):
    X_in = scaler.transform(X_eval) if use_scaled else X_eval
    pi = permutation_importance(
        model, X_in, y_eval, scoring="roc_auc",
        n_repeats=30, random_state=RNG_SEED,
    )
    imp = pd.Series(pi.importances_mean, index=X_eval.columns)
    agg = imp.groupby(feature_group).sum().sort_values(ascending=False)
    return agg

imp_lr = aggregated_permutation_importance(logreg, X_test, y_test, use_scaled=True)
imp_rf = aggregated_permutation_importance(rf, X_test, y_test, use_scaled=False)

print("\n=== Importanza delle feature (permutation importance su AUC, aggregata) ===")
print("\nLogistic Regression:")
print(imp_lr.round(4).to_string())
print("\nRandom Forest:")
print(imp_rf.round(4).to_string())

top3_rf = imp_rf.head(3)
print(f"\nTop 3 feature piu' predittive (Random Forest): {list(top3_rf.index)}")

# ------------------------------------------------------------------ PLOT --
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#fcfcfb")

# ROC
for ax_i, (name, proba, color) in enumerate([
    ("Logistic Regression", proba_lr, COL_1), ("Random Forest", proba_rf, COL_2)
]):
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC={results[name]['auc']:.2f})")
axes[0].plot([0, 1], [0, 1], color="#898781", lw=1, ls="--")
axes[0].set_title("Curva ROC - test set", loc="left", fontsize=11)
axes[0].set_xlabel("False Positive Rate", color="#52514e", fontsize=9)
axes[0].set_ylabel("True Positive Rate", color="#52514e", fontsize=9)
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_facecolor("#fcfcfb")
axes[0].spines[["top", "right"]].set_visible(False)

# feature importance RF
imp_plot = imp_rf.sort_values()
axes[1].barh(imp_plot.index, imp_plot.values, color=COL_1)
axes[1].set_title("Importanza feature (Random Forest, permutation su AUC)",
                   loc="left", fontsize=11)
axes[1].set_xlabel("Calo di AUC se la feature viene permutata", color="#52514e", fontsize=9)
axes[1].set_facecolor("#fcfcfb")
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].tick_params(labelsize=8)

fig.tight_layout()
fig.savefig("../charts/chart3_model_comparison.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nGrafico salvato in chart3_model_comparison.png")
