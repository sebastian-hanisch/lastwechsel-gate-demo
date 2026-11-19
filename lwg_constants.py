"""Konstanten: Gate-Modell, Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Gate (markov-queue-demo: zwei Spuren, 6 Plätze INSGESAMT, Zeiteinheit = mittlere Abfertigungsdauer = 3 min)
C_SERV, K_CAP, MU = 2, 6, 1.0
LAM0, LAM1 = 1.6, 2.0                      # Lastwechsel von 80 % auf 100 % Last
MIN_PRO_EINHEIT = 3.0

# --- Karten auf der gesehenen Zahl n (ganzzahlige Scores s = Q·n − κ, Schwelle H ganzzahlig)
Q = 8                                      # Skala: Gittereinheit 1/8
KAPPA_OPTIONS = (18, 20, 22, 24, 26, 28)   # k = κ/8 = 2,25 ... 3,5 (k = 2,0 liegt unter dem Nullmittel 2,16: Drift schon ohne Änderung, nicht angeboten)
DEFAULT_KAPPA = 20
H_MIN, H_MAX, DEFAULT_H = 10, 700, 97      # 97: Lehrbuch-Schwelle (unabhängig, ARL₀ = 408,2)
ARL_CAP = 2.0e5                            # größte ARL₀, die der Löser verlässlich liefert (Vorversuch: bei ARL ≈ 10¹⁵ negative Werte)
SURV_STEPS = 2000                          # Schutz: überlebt die Karte 2000 Lkw mit mindestens 99 %, ist ARL₀ > 2·10⁵ (Heuristik, Tests prüfen sie)
SURV_LIMIT = 0.99
H_STEP_FROZEN = 4                          # Raster der gefrorenen Kennlinien (alle Scores sind Vielfache von 2)
H_MAX_FROZEN = 1200
TARGET_OPTIONS = (50, 100, 150, 200, 300, 370, 500, 700, 1000, 1500, 2000)
DEFAULT_TARGET = 370
LAM1_OPTIONS = (1.7, 1.8, 2.0, 2.4, 3.0, 4.0)
DEFAULT_LAM1 = 2.0
ARL0_NENN = 370.0                          # Ziel der Linie (gegen 370,398 der Shewhart-Karte, siehe README)

# --- Experiment 3 (Wartezeit-Karten, vorgerechnet)
BLOCK_OPTIONS = (1, 5, 10, 20, 40)
DEFAULT_BLOCK = 1                          # b = 1 ist die Rohkarte (jeder Lkw)
VERFAHREN = ("Blockmittel", "Ausdünnen (jeder b-te)")
DEFAULT_VERFAHREN = VERFAHREN[0]
GROESSE_OPTIONS = (1.8, 2.0, 2.4)
DEFAULT_GROESSE = 2.0
LIVE_R_MIN, LIVE_R_MAX, DEFAULT_LIVE_R, LIVE_R_STEP = 50, 200, 100, 50
LIVE_T = 2000                              # Lkw je Live-Lauf nach 400 Lkw Einschwingen
LIVE_BURN = 400
V2_H, V2_K = 4.77, 0.5                     # naiver zweiseitiger CUSUM auf der Wartezeit (V2 6.1)
NAIV_NOMINAL = 368.6                       # ARL₀ des naiven zweiseitigen CUSUM bei unabhängigen Werten (Näherung ARL⁺/2; tests/test_claims.py)
LIVE_SEED_MAX, DEFAULT_LIVE_SEED = 999999, 8080

# --- Experiment 4 (Modellkarte)
QS_OPTIONS = (20, 40, 80)
DEFAULT_QS = 20
LAM1_WAHR_OPTIONS = (1.7, 1.8, 2.0, 2.4, 3.0, 4.0)
LAM1_ENTW_OPTIONS = (1.8, 2.0, 2.4, 3.0, 4.0)
LAM0_WAHR_OPTIONS = (1.4, 1.5, 1.55, 1.6, 1.65, 1.7, 1.8)
DEFAULT_LAM0_WAHR = 1.6

# --- Experiment 5 (Residuenkarte)
AENDERUNGEN = ("Ankunftsrate 1,6 → 1,8", "Ankunftsrate 1,6 → 2,0", "Ankunftsrate 1,6 → 2,4", "Abfertigung 1,0 → 0,85")
AENDERUNG_KEY = {AENDERUNGEN[0]: "lam 1,6 -> 1,8", AENDERUNGEN[1]: "lam 1,6 -> 2,0", AENDERUNGEN[2]: "lam 1,6 -> 2,4", AENDERUNGEN[3]: "mu 1,0 -> 0,85"}
DEFAULT_AENDERUNG = AENDERUNGEN[1]

# --- Messreihen (tools/lwg_make_sim.py)
SIM_R_NULL, SIM_R_CHANGE, SIM_T_NULL, SIM_T_CHANGE, SIM_BURN = 10000, 8000, 12000, 6000, 400
SIM_SEEDS = (17001, 17002)                 # erster und zweiter Zufalls-Seed der Messreihen

COLOR_IID = "#4c78a8"
COLOR_KETTE = "#c0504d"
COLOR_ROH = "#2F6B65"
COLOR_BLOCK = "#e8a33d"
COLOR_AUSD = "#8a6bbf"
COLOR_RESID = "#7f7f7f"
COLOR_MODELL = "#17a2b8"
COLOR_N = "#2d3e50"

PRESETS = {
    "Lehrbuch-Schwelle (unabhängig, ARL₀ 408)": {},
    "Schwelle angepasst (370 auf der Kette)": {"h": 309, "target": 370},
    "Blockmittel b = 20": {"verfahren": "Blockmittel", "block": 20},
    "Modellkarte für 2,0": {"lam1_wahr": 2.0, "lam1_entw": 2.0, "qs": 40},
    "Falsches Nullmodell (1,7 statt 1,6)": {"lam0_wahr": 1.7},
    "Größerer Sprung (2,4)": {"lam1": 2.4, "groesse": 2.4, "lam1_wahr": 2.4, "aenderung": "Ankunftsrate 1,6 → 2,4"},
    "Abfertigung langsamer (0,85)": {"aenderung": "Abfertigung 1,0 → 0,85"},
}
PRESET_HELP = {
    "Lehrbuch-Schwelle (unabhängig, ARL₀ 408)": "Karte auf der gesehenen Zahl, Schwelle H = 97: unter der Annahme unabhängiger Lkw ist ARL₀ = 408,2. Zeigt, was die Kette daraus macht.",
    "Schwelle angepasst (370 auf der Kette)": "Schwelle H = 309: auf der tatsächlichen Kette ist ARL₀ dort 378,1 (nahe 370; H = 308: 367,0). Zeigt, was die Verzögerung nach dem Wechsel dann kostet.",
    "Blockmittel b = 20": "Mittel von 20 Lkw als Beobachtung der Karte (Wartezeit, vorgerechnet): fast unabhängig, aber auf der Blockgrenze nicht schneller.",
    "Modellkarte für 2,0": "Likelihood-Quotient-Karte der Übergänge, entworfen für den Wechsel 1,6 → 2,0, verglichen mit der Rohkarte.",
    "Falsches Nullmodell (1,7 statt 1,6)": "Die Karten sind für 1,6 geeicht, die Wahrheit ohne Änderung ist 1,7: ARL₀ der Rohkarte sinkt auf das 0,65-Fache.",
    "Größerer Sprung (2,4)": "Ankunftsrate springt auf 2,4: dieselben Karten, größerer Sprung.",
    "Abfertigung langsamer (0,85)": "Nicht die Ankunftsrate ändert sich, sondern die Abfertigung (1,0 → 0,85): die Residuenkarte sieht diese Änderung gut.",
}
