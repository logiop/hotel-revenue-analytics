"""
Generatore di dataset sintetico per l'esercizio "Revenue Management & Guest
Retention" - hotel di lusso da 60 camere, 24 mesi di storico.

v2 - corregge un bug del generatore v1: il meccanismo "repeat guest" di v1
sceglieva a ogni prenotazione, con probabilita' 25%, un ospite a caso tra
quelli gia' visti. Con un pool di ospiti limitato che si esauriva in fretta,
questo produceva un tasso di ritorno artificiale del 98,6% (quasi tutti
"repeat"), un target completamente degenere e inutile per un modello di
classificazione. Qui il ritorno e' deciso UNA VOLTA per ciascun ospite, alla
prima visita, con una probabilita' che dipende da alcune feature (per dare
al modello dello Step 3 un segnale vero ma non banale) e poi eventualmente
schedulato come seconda prenotazione indipendente.

Sporcizia e trappole inserite di proposito:
  - guest_name scritto in modo diverso tra bookings.csv e guests.csv
    (case, spazi extra, ordine invertito) -> richiede normalizzazione per il join
  - room_rate in valuta mista (EUR/USD, non convertita) -> richiede
    conversione in Step 1
  - una piccola percentuale di righe con check_out <= check_in
    -> richiede validazione in Step 1
  - OMONIMIA CONTROLLATA: ~15% degli ospiti condivide il nome con un altro
    ospite realmente diverso (email/paese diversi) -> il join per nome resta
    strutturalmente ambiguo per quella quota, un problema reale di entity
    resolution (vedi Step 1)
  - paradosso di Simpson sull'effetto della promo booking diretto (Step 4):
    l'alta stagione ha naturalmente una quota di booking diretti piu' alta,
    INDIPENDENTE dalla promo, e il periodo post-promo contiene piu' mesi di
    alta stagione del periodo pre-promo -> il mix stagionale si sposta nel
    tempo, mascherando/gonfiando l'effetto vero (piccolo ma reale) della promo
  - CENSURA: i "repeat" schedulati oltre la fine del periodo osservato non
    compaiono nei dati (l'ospite non e' ancora "tornato" nella finestra
    osservabile) -> va gestita esplicitamente allo Step 3
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG = np.random.default_rng(42)

# ---- Parametri generali -----------------------------------------------
N_ROOMS = 60
TODAY = date(2026, 9, 10)
START = TODAY - timedelta(days=365 * 2)          # 24 mesi di storico
PROMO_LAUNCH = TODAY - timedelta(days=182)        # promo lanciata ~6 mesi fa

HIGH_SEASON_MONTHS = {6, 7, 8, 9, 12}             # giu-set + dicembre

COUNTRIES = ["Italy", "Germany", "France", "USA", "UK", "Switzerland",
             "Japan", "UAE", "Spain", "Netherlands"]
FOODS = ["italiana", "vegetariana", "senza glutine", "pesce", "vegana",
         "internazionale", "kosher", "halal"]

FIRST_NAMES = [
    "Marco", "Giulia", "Luca", "Sara", "Andrea", "Elena", "Francesco", "Chiara",
    "Alessandro", "Valentina", "John", "Emma", "Hans", "Sophie", "Yuki", "Fatima",
    "Carlos", "Anna", "Pierre", "Laura", "David", "Maria", "Thomas", "Isabella",
    "Ahmed", "Olga", "Nikolai", "Priya", "Omar", "Charlotte", "Paolo", "Federica",
    "Matteo", "Silvia", "Simone", "Giorgia", "William", "Olivia", "James", "Amelia",
    "Klaus", "Ingrid", "Jean", "Camille", "Kenji", "Aiko", "Layla", "Youssef",
    "Diego", "Lucia", "Henri", "Marie", "Robert", "Patricia", "Erik", "Astrid",
    "Dmitri", "Ekaterina", "Arjun", "Ananya",
]
LAST_NAMES = [
    "Rossi", "Bianchi", "Ferrari", "Esposito", "Romano", "Colombo", "Ricci",
    "Marino", "Greco", "Bruno", "Smith", "Muller", "Dubois", "Tanaka", "Al-Farsi",
    "Garcia", "Kowalski", "Ivanov", "Nakamura", "Schmidt", "Johnson", "Meyer",
    "Leroy", "Conti", "Silva", "Novak", "Petrov", "Sharma", "Al-Rashid", "Fischer",
    "Wagner", "Moreau", "Bernard", "Suzuki", "Watanabe", "Costa", "Ferreira",
    "Kaczmarek", "Sokolov", "Popov", "Nowak", "Weber", "Klein", "Lefevre", "Girard",
    "Kobayashi", "Yamamoto", "Hassan", "Ibrahim", "Fernandez", "Martinez", "Lopez",
    "Andersen", "Nilsson", "Kim", "Park", "Patel", "Kumar", "Singh", "Chen",
]

rng_py = np.random.default_rng(7)
ALL_NAME_COMBOS = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
rng_py.shuffle(ALL_NAME_COMBOS)

N_GUESTS = 3300           # ospiti a "prima visita"
COLLISION_FRACTION = 0.15  # quota di ospiti volutamente coinvolta in un'omonimia

n_collision_guests = int(N_GUESTS * COLLISION_FRACTION)
n_collision_pairs = n_collision_guests // 2
n_unique_guests = N_GUESTS - 2 * n_collision_pairs

assert n_unique_guests + n_collision_pairs <= len(ALL_NAME_COMBOS), \
    "pool di nomi troppo piccolo: aumenta FIRST_NAMES/LAST_NAMES"

pool = iter(ALL_NAME_COMBOS)
guest_canonical_names = []
for _ in range(n_unique_guests):
    fn, ln = next(pool)
    guest_canonical_names.append(f"{fn} {ln}")
for _ in range(n_collision_pairs):
    fn, ln = next(pool)
    name = f"{fn} {ln}"
    guest_canonical_names.extend([name, name])   # due ospiti diversi, stesso nome

RNG.shuffle(guest_canonical_names)  # cosi' l'ordine di generazione non e' correlato al nome

# ---- 1. Genero l'anagrafica ospiti (base) ------------------------------
guests = []
for gid, canonical_name in enumerate(guest_canonical_names):
    fn = canonical_name.split()[0]
    email = f"{canonical_name.lower().replace(' ', '.')}{RNG.integers(1, 999)}@{RNG.choice(['gmail.com','outlook.com','yahoo.com','icloud.com'])}"
    country = RNG.choice(COUNTRIES)
    food = RNG.choice(FOODS)
    guests.append({
        "guest_id": gid,
        "canonical_name": canonical_name,
        "email": email,
        "country": country,
        "cibo_preferito": food,
    })
guests_df = pd.DataFrame(guests)


def messy_name(name: str) -> str:
    """Introduce variazioni di scrittura casuali per simulare fonti diverse."""
    r = RNG.random()
    if r < 0.25:
        return name.upper()
    elif r < 0.45:
        return name.lower()
    elif r < 0.60:
        return f"  {name}  "          # spazi extra
    elif r < 0.70:
        parts = name.split()
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"   # cognome nome invertiti
    return name


# ---- 2. Utility di generazione booking (condivisa tra prima visita e repeat)
n_days = (TODAY - START).days
all_days = [START + timedelta(days=i) for i in range(n_days)]

def season_weight(d: date) -> float:
    return 2.2 if d.month in HIGH_SEASON_MONTHS else 1.0

weights = np.array([season_weight(d) for d in all_days], dtype=float)
weights /= weights.sum()


def sample_checkin_dates(n):
    idx = RNG.choice(len(all_days), size=n, p=weights)
    return [all_days[i] for i in idx]


def make_booking_fields(cin: date):
    """Genera nights/channel/rate/currency/cancelled per una prenotazione con
    check-in a una data data (usata sia per prime visite che per repeat)."""
    is_high = cin.month in HIGH_SEASON_MONTHS
    post_promo = cin >= PROMO_LAUNCH

    if is_high:
        nights = max(1, int(RNG.lognormal(mean=1.3, sigma=0.4)))
    else:
        nights = max(1, int(RNG.lognormal(mean=0.9, sigma=0.5)))
    nights = min(nights, 14)

    base_direct_share = 0.45 if is_high else 0.25
    promo_effect = 0.05 if post_promo else 0.0
    direct_prob = min(0.95, base_direct_share + promo_effect)
    channel = "direct" if RNG.random() < direct_prob else "OTA"

    base_rate = RNG.normal(430, 40) if is_high else RNG.normal(240, 30)
    if channel == "direct":
        base_rate *= RNG.normal(0.96, 0.03)
    room_rate = max(80, round(base_rate, 2))

    currency = "USD" if RNG.random() < 0.18 else "EUR"
    cancelled = RNG.random() < 0.09

    return dict(nights=nights, channel=channel, room_rate=room_rate,
                currency=currency, cancelled=cancelled, is_high=is_high)


LOYAL_COUNTRIES = {"Italy", "Switzerland"}

def p_repeat(first_fields: dict, country: str) -> float:
    """Probabilita' che l'ospite torni entro ~18 mesi. Segnale deliberato:
    canale diretto e alta stagione alla prima visita aumentano la fedelta',
    soggiorni piu' lunghi un po' di meno, l'ADR pagato non ha un effetto
    disegnato (deve restare la feature meno predittiva)."""
    p = 0.15
    if first_fields["channel"] == "direct":
        p += 0.15
    if first_fields["is_high"]:
        p += 0.07
    if country in LOYAL_COUNTRIES:
        p += 0.05
    p += 0.01 * min(first_fields["nights"], 6)
    return float(np.clip(p, 0.03, 0.75))


# ---- 3. Genero le prenotazioni: prima visita + eventuale repeat --------
first_checkins = sample_checkin_dates(N_GUESTS)

bookings = []
guest_first_visit = {}
bid_counter = 0

order = np.argsort([d.toordinal() for d in first_checkins])  # genero in ordine cronologico
for i in order:
    guest_id = int(i)
    cin = first_checkins[i]
    fields = make_booking_fields(cin)
    cout = cin + timedelta(days=fields["nights"])
    guest_first_visit[guest_id] = cin

    canonical = guests_df.loc[guests_df.guest_id == guest_id, "canonical_name"].iloc[0]
    country = guests_df.loc[guests_df.guest_id == guest_id, "country"].iloc[0]

    bookings.append({
        "booking_id": f"BK{bid_counter:05d}",
        "guest_name": messy_name(canonical),
        "guest_id_internal": guest_id,
        "check_in": cin,
        "check_out": cout,
        "room_rate": fields["room_rate"],
        "currency": fields["currency"],
        "channel": fields["channel"],
        "cancelled": fields["cancelled"],
    })
    bid_counter += 1

    # decido UNA VOLTA, alla prima visita, se e quando l'ospite tornera'
    if RNG.random() < p_repeat(fields, country):
        delay_days = int(RNG.uniform(30, 600))
        repeat_cin = cin + timedelta(days=delay_days)
        if repeat_cin <= TODAY:
            rfields = make_booking_fields(repeat_cin)
            rcout = repeat_cin + timedelta(days=rfields["nights"])
            bookings.append({
                "booking_id": f"BK{bid_counter:05d}",
                "guest_name": messy_name(canonical),
                "guest_id_internal": guest_id,
                "check_in": repeat_cin,
                "check_out": rcout,
                "room_rate": rfields["room_rate"],
                "currency": rfields["currency"],
                "channel": rfields["channel"],
                "cancelled": rfields["cancelled"],
            })
            bid_counter += 1
        # se repeat_cin > TODAY: l'ospite "tornera'" ma oltre l'orizzonte
        # osservato -> censura, la seconda prenotazione semplicemente non
        # compare nei dati (realistico)

bookings_df = pd.DataFrame(bookings).sort_values("check_in").reset_index(drop=True)
bookings_df["booking_id"] = [f"BK{i:05d}" for i in range(len(bookings_df))]

# --- sporcizia extra sulle date (per lo step di validazione) ---
bad_idx = RNG.choice(bookings_df.index, size=25, replace=False)
bookings_df.loc[bad_idx, "check_out"] = bookings_df.loc[bad_idx, "check_in"]
bad_idx2 = RNG.choice(bookings_df.index.difference(bad_idx), size=15, replace=False)
bookings_df.loc[bad_idx2, "check_out"] = bookings_df.loc[bad_idx2, "check_in"] - timedelta(days=1)

guest_id_lookup = bookings_df[["booking_id", "guest_id_internal"]].copy()
bookings_out = bookings_df.drop(columns=["guest_id_internal"])

# ---- 4. guests.csv (nomi "sporchi" indipendentemente da bookings.csv) --
guests_out = guests_df.copy()
guests_out["prima_visita"] = guests_out["guest_id"].map(
    lambda g: guest_first_visit.get(g, pd.NaT)
)
guests_out = guests_out.dropna(subset=["prima_visita"]).reset_index(drop=True)
guests_out["guest_name"] = guests_out["canonical_name"].apply(messy_name)
guests_out = guests_out[["guest_name", "email", "country", "prima_visita", "cibo_preferito"]]

# ---- 5. promo_log.csv --------------------------------------------------
promo_log = pd.DataFrame([{
    "promo_id": "PROMO2026-01",
    "nome_promo": "Prenota Diretto, Risparmia di Piu'",
    "data_lancio": PROMO_LAUNCH,
    "canale_target": "direct",
    "descrizione": "Sconto e vantaggi per chi prenota direttamente sul sito "
                    "dell'hotel invece che tramite OTA",
}])

# ---- Salvataggio ---------------------------------------------------------
bookings_out.to_csv("../data/bookings.csv", index=False)
guests_out.to_csv("../data/guests.csv", index=False)
promo_log.to_csv("../data/promo_log.csv", index=False)
guest_id_lookup.to_csv("../data/_internal_guest_id_lookup.csv", index=False)

n_repeat_guests = bookings_out.groupby(
    guest_id_lookup.set_index("booking_id").loc[bookings_out["booking_id"], "guest_id_internal"].values
).size()

print(f"bookings.csv: {len(bookings_out)} righe")
print(f"guests.csv: {len(guests_out)} righe (di cui {n_collision_guests} coinvolti in omonimia)")
print(f"promo_log.csv: {len(promo_log)} riga")
print(f"Periodo dati: {START} -> {TODAY}")
print(f"Lancio promo: {PROMO_LAUNCH}")
print(f"Ospiti con >=2 prenotazioni nel dataset consegnato: {(n_repeat_guests >= 2).sum()} "
      f"su {guests_out.shape[0]} ({(n_repeat_guests >= 2).mean():.1%})")
