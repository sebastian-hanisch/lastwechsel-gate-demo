"""Lastwechsel am Gate: was nach dem Wechsel noch funktioniert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 der Reihe „Änderungserkennung“: Karten (CUSUM), deren Schwelle unter „unabhängige Beobachtungen“ eingestellt ist, laufen auf einem Lkw-Tor (M/M/2/6), dessen Beobachtungen abhängig
sind. Exakt gerechnet wird auf der gesehenen Zahl (endliche Markov-Kette), vorgerechnet (Simulation) auf der Wartezeit.

Lauffähig mit: streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st

import lwg_constants as C
import lwg_evaluation as EV
import lwg_reference as REF
import lwg_sim as SIM
import lwg_texts as TX
import lwg_visualization as V
from lwg_evaluation import de, pct
from lwg_layout import dunkles_design, kopf_css
from lwg_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Lastwechsel am Gate – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exp1(h, kappa):
    return EV.exp1(h, kappa)


@st.cache_data(show_spinner=False)
def _exp1_sweep(kappa):
    return {h: EV.exp1(h, kappa, tmax=100) for h in (20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300)}


@st.cache_data(show_spinner=False)
def _exp2(kappa, target, lam1):
    return EV.exp2(kappa, target, lam1)


@st.cache_data(show_spinner=False)
def _exp4(lam1_wahr, lam1_entw, lam0_wahr, qs):
    return EV.exp4(lam1_wahr, lam1_entw, lam0_wahr, qs)


@st.cache_data(show_spinner=False)
def _probe(r, t, seed):
    g0 = EV.gate(C.LAM0)
    return SIM.live_probe(r, t, seed, g0.mean_w, g0.sd_w)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lwg_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🚛 Lastwechsel am Gate: was nach dem Wechsel noch funktioniert")
st.markdown(TX.INTRO)

with st.expander("So funktioniert die Prüfung am Gate", expanded=True):
    st.markdown(TX.SO_FUNKTIONIERT)
st.markdown(TX.FRAGEN)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Experiment 1: was bedeutet 370?")
    h = st.slider("Schwelle H (Gittereinheit 1/8)", *bounds("h_slider"), step=1, key="h_slider",
                  help="Alarm, sobald die Karte H erreicht. Die Voreinstellung 97 ist die Lehrbuch-Schwelle (unabhängige Lkw: ARL₀ = 408,2); auf der Kette ist ARL₀ bei H = 309 gleich 378,1 (nahe 370; H = 308: 367,0).")
    kappa = st.select_slider("Referenzwert κ (k = κ/8)", options=C.KAPPA_OPTIONS, key="kappa_select",
                             help="Score je Lkw = 8·n − κ. Das Nullmittel der gesehenen Zahl ist 2,16: ein k unter diesem Mittel würde schon ohne Änderung nach oben driften und wird nicht angeboten.")
    st.header("⚙️ Experiment 2: Schwelle anpassen")
    target = st.select_slider("Ziel-ARL₀ (Lkw)", options=C.TARGET_OPTIONS, key="target_select", help="Die Schwelle wird so gewählt, dass die tatsächliche ARL₀ dieses Ziel trifft (Zero-state).")
    lam1 = st.select_slider("Größe des Wechsels λ₁ (alt: 1,6)", options=C.LAM1_OPTIONS, key="lam1_select", format_func=lambda x: de(x, 1),
                            help="Neue Ankunftsrate. Die Steady-state-Zahlen sind nur für 2,0 gerechnet.")
    st.header("⚙️ Experiment 3: Blockmittel und Ausdünnen")
    verfahren = st.radio("Verfahren", C.VERFAHREN, key="verfahren_radio", help="Blockmittel: das Mittel von b Lkw ist eine Beobachtung. Ausdünnen: nur jeder b-te Lkw geht in die Karte (gemessen: 5, 10, 20). b = 1 ist die Rohkarte.")
    block = st.select_slider("Blocklänge b", options=C.BLOCK_OPTIONS, key="block_select")
    groesse = st.select_slider("Größe des Wechsels λ₁", options=C.GROESSE_OPTIONS, key="groesse_select", format_func=lambda x: de(x, 1), help="Vorgerechnet sind 1,8 / 2,0 / 2,4.")
    live_r = st.slider("Läufe der Live-Probe", *bounds("live_r_slider"), step=C.LIVE_R_STEP, key="live_r_slider", help=f"Unabhängige Gates, je {C.LIVE_T} Lkw nach {C.LIVE_BURN} Lkw Einschwingen (SplitMix64).")
    seed = st.number_input("Zufalls-Seed der Live-Probe", *bounds("live_seed_input"), key="live_seed_input", step=1)
    st.button("🎲 Neue Live-Probe", width="stretch", on_click=randomize_seed)
    st.header("⚙️ Experiment 4: Modellkarte")
    lam1_wahr = st.select_slider("Wahres λ₁", options=C.LAM1_WAHR_OPTIONS, key="lam1_wahr_select", format_func=lambda x: de(x, 1))
    lam1_entw = st.select_slider("Entwurf der Modellkarte für λ₁", options=C.LAM1_ENTW_OPTIONS, key="lam1_entw_select", format_func=lambda x: de(x, 1))
    lam0_wahr = st.select_slider("Wahres λ₀′ ohne Änderung (die Karten sind für 1,6 geeicht)", options=C.LAM0_WAHR_OPTIONS, key="lam0_wahr_select", format_func=lambda x: de(x, 2))
    qs = st.select_slider("Gitter q_s des Scores (1/q_s)", options=C.QS_OPTIONS, key="qs_select", help="Der Likelihood-Quotient wird auf Vielfache von 1/q_s gerundet (ganzzahlige Scores machen die Kette endlich).")
    st.header("⚙️ Experiment 5: Residuenkarte")
    aenderung = st.radio("Änderung", C.AENDERUNGEN, key="aenderung_radio", help="Drei Änderungen der Ankunftsrate und eine der Abfertigung (vorgerechnet, Simulation).")

sync_query_params({"h_slider": int(h), "kappa_select": int(kappa), "target_select": int(target), "lam1_select": float(lam1), "block_select": int(block), "verfahren_radio": verfahren,
                   "groesse_select": float(groesse), "live_r_slider": int(live_r), "live_seed_input": int(seed), "lam1_wahr_select": float(lam1_wahr),
                   "lam1_entw_select": float(lam1_entw), "lam0_wahr_select": float(lam0_wahr), "qs_select": int(qs), "aenderung_radio": aenderung})
S = EV.Settings(int(h), int(kappa), int(target), float(lam1), int(block), verfahren, float(groesse), int(live_r), int(seed), float(lam1_wahr), float(lam1_entw), float(lam0_wahr), int(qs), aenderung)
g0 = EV.gate(C.LAM0)
satz0, satz1 = REF.satz(0), REF.satz(1)

# =====================================================================================================================================================
# Experiment 1
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 1: Was bedeutet „370“ am Gate?")
st.markdown(TX.EXP1_FRAGE)
with st.spinner("Rechne die Ketten (exakt) …"):
    r1 = _exp1(S.h, S.kappa)
c1, c2, c3, c4 = st.columns(4)
c1.metric("ARL₀ bei unabhängigen Lkw", de(r1["arl0_iid"], 1) if r1["arl0_iid"] is not None else "über 2·10⁵", delta="Lehrbuch-Rechnung", delta_color="off")
c2.metric("ARL₀ auf der Kette", de(r1["arl0_kette"], 1) if r1["arl0_kette"] is not None else "über 2·10⁵", delta="tatsächlich (exakt)", delta_color="off")
c3.metric("Faktor", de(r1["faktor"], 1) if r1["faktor"] is not None else "–", delta="unabhängig / Kette", delta_color="off")
c4.metric("Alarme in den ersten 100 Lkw", pct(r1["p100_kette"]), delta=f"{pct(r1['p100_iid'])} bei unabhängigen Lkw", delta_color="off")
st.plotly_chart(V.build_surv([("unabhängige Lkw (Lehrbuch)", r1["surv_iid"], C.COLOR_IID, "dash"), ("tatsächliche Kette", r1["surv_kette"], C.COLOR_KETTE, "solid")], vlines=[(100, "100 Lkw")]),
                width="stretch", key="exp1_chart")
st.caption("Verteilung der Lauflänge ohne Änderung (exakt, Zero-state); beide Karten haben dieselbe Schwelle und denselben Score, nur die Annahme über die Lkw-Folge unterscheidet sich.")
kind, text = EV.verdict_exp1(r1)
getattr(st, kind)(text)
with st.spinner("Rechne die Tabelle über Schwellen …"):
    sweep = _exp1_sweep(S.kappa)
rows = []
for hh, r in sweep.items():
    rows.append({"Schwelle H": hh, "ARL₀ unabhängig": de(r["arl0_iid"], 1) if r["arl0_iid"] is not None else "über 2·10⁵", "ARL₀ Kette": de(r["arl0_kette"], 1) if r["arl0_kette"] is not None else "über 2·10⁵",
                 "Faktor": de(r["faktor"], 2) if r["faktor"] is not None else "–", "Alarme in 100 Lkw (unabhängig / Kette)": f"{pct(r['p100_iid'])} / {pct(r['p100_kette'])}"})
st.markdown(f"**Der Faktor je Schwelle** (κ = {S.kappa}, exakt):")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown(TX.EXP1_NEGATIV.format(v2_a=de(satz0["v2"]["arl0"], 1), v2_b=de(satz1["v2"]["arl0"], 1), v2_p=de(100 * satz0["v2"]["p100"], 1)))

# =====================================================================================================================================================
# Experiment 2
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 2: Was bringt es, die Schwelle anzupassen?")
st.markdown(TX.EXP2_FRAGE)
r2 = _exp2(S.kappa, S.target, S.lam1)
if r2["iid"] is not None and r2["kette"] is not None:
    (hi, ai), (hk, ak) = r2["iid"], r2["kette"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Schwelle H* (unabhängig → Kette)", f"{de(hi, 1)} → {de(hk, 1)}", delta=f"ARL₀ = {S.target}", delta_color="off")
    c2.metric("Verzögerung, unabhängige Rechnung", de(ai, 1), delta="Lkw nach dem Wechsel (Zero-state)", delta_color="off")
    c3.metric("Verzögerung, Kette", de(ak, 1), delta=f"{de(ak / ai, 1)}-fach der unabhängigen Rechnung", delta_color="inverse")
    c4.metric("Schwelle in Standardabweichungen der gesehenen Zahl", f"{de(hi / (C.Q * g0.sd_n), 1)} → {de(hk / (C.Q * g0.sd_n), 1)}", delta="1 Gittereinheit = 1/8", delta_color="off")
    st.plotly_chart(V.build_kennlinie(S.kappa, S.lam1, S.target), width="stretch", key="exp2_chart")
    st.caption("Kennlinie: Verzögerung nach dem Wechsel gegen die ARL₀, die die Schwelle hat (gefroren, exakt, Zero-state). Beim Ziel ist die Kette um den Faktor oben langsamer; die Kurven fallen nicht zusammen.")
kind, text = EV.verdict_exp2(r2)
getattr(st, kind)(text)
if r2["dist"] is not None:
    d = r2["dist"]
    st.plotly_chart(V.build_surv([("nach dem Wechsel", d["surv1"], C.COLOR_KETTE, "solid"), ("ohne Änderung", d["surv0"], C.COLOR_IID, "dash")], vlines=[(100, "100 Lkw")]),
                    width="stretch", key="exp2_dist_chart")
    st.caption(f"Verteilung der Lauflänge an der ganzzahligen Schwelle H = {d['H']} (live, exakt): durchgezogen nach dem Wechsel, gestrichelt ohne Änderung.")
rows = []
for kap in C.KAPPA_OPTIONS:
    ti, tk = REF.at_target_zero(kap, "iid", S.lam1, float(S.target)), REF.at_target_zero(kap, "kette", S.lam1, float(S.target))
    if ti is None or tk is None:
        continue
    ss = REF.steady_at_target(kap, float(S.target)) if S.lam1 == C.LAM1 else {"gleich": None, "eq": None}
    rows.append({"κ": kap, "k": de(kap / C.Q, 2), "H* unabhängig": de(ti[0], 1), "ARL₁ unabhängig": de(ti[1], 1), "H* Kette": de(tk[0], 1), "ARL₁ Kette (Zero-state)": de(tk[1], 1),
                 "ARL₁ Steady-state, gleiche Schwelle (Steady-ARL₀ dort)": f"{de(ss['gleich'][0], 1)} ({de(ss['gleich'][1], 1)})" if ss["gleich"] else "–",
                 "ARL₁ Steady-state bei Steady-ARL₀ = Ziel": de(ss["eq"], 1) if ss["eq"] else "–", "Kette / unabhängig": de(tk[1] / ti[1], 2)})
st.markdown(f"**Fünf Referenzwerte κ bei ARL₀ = {S.target}** (Wechsel 1,6 → {de(S.lam1, 1)}; exakt, aus den gefrorenen Kennlinien; Steady-state nur für 2,0):")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
kr = REF.k_reihe()
st.markdown("**Die acht gemessenen k bei ARL₀ = 370 (Wechsel 1,6 → 2,0, Zero-state, exakt):**")
st.dataframe(pd.DataFrame([{"k": de(z["k"], 2), "Skala q / κ": f"{z['q']} / {z['kappa']}", "H* unabhängig": de(z["h_iid"], 1), "ARL₁ unabhängig": de(z["arl1_iid"], 1), "H* Kette": de(z["h_kette"], 1),
                            "ARL₁ Kette": de(z["arl1_kette"], 1), "Kette / unabhängig": de(z["verhaeltnis"], 2)} for z in kr]), hide_index=True, width="stretch")
kind, text = EV.verdict_k_reihe(EV.k_summary())
getattr(st, kind)(text)
st.markdown(TX.EXP2_NEGATIV)

# =====================================================================================================================================================
# Experiment 3
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 3: Helfen Blockmittel oder Ausdünnen?")
st.markdown(TX.EXP3_FRAGE)
r3 = EV.exp3(S.verfahren, S.block, S.groesse)
label = r3["label"]
items = []
for nm, shown, color in (("Roh k=0,14", "Roh (jeder Lkw, k = Sprung/2)", C.COLOR_ROH), ("Roh k=0,5", "Roh k = 0,5", C.COLOR_ROH), ("Blockmittel b=5", "Blockmittel b = 5", C.COLOR_BLOCK),
                         ("Blockmittel b=10", "Blockmittel b = 10", C.COLOR_BLOCK), ("Blockmittel b=20", "Blockmittel b = 20", C.COLOR_BLOCK), ("Blockmittel b=40", "Blockmittel b = 40", C.COLOR_BLOCK),
                         ("Ausduennen jeder 5.", "Ausdünnen jeder 5.", C.COLOR_AUSD), ("Ausduennen jeder 10.", "Ausdünnen jeder 10.", C.COLOR_AUSD), ("Ausduennen jeder 20.", "Ausdünnen jeder 20.", C.COLOR_AUSD),
                         ("Karte auf n (sim, kappa 20)", "Karte auf der gesehenen Zahl (Simulation)", C.COLOR_N)):
    v = EV.sim_value(nm, label)
    if v is not None:
        items.append((shown, v[0], v[1], color, nm))
sel_shown = next((i[0] for i in items if i[4] == r3["name"]), None)
c1, c2, c3 = st.columns(3)
if r3["sel"] is not None:
    c1.metric("Gewählte Karte", f"{de(r3['sel'][0], 1)} ± {de(r3['sel'][1], 1)}", delta=sel_shown, delta_color="off")
    c2.metric("Rohkarte", f"{de(r3['roh'][0], 1)} ± {de(r3['roh'][1], 1)}", delta="gleiche ARL₀ = 370", delta_color="off")
    c3.metric("Gewählt minus Roh", EV.de_s(r3["sel"][0] - r3["roh"][0], 1), delta="Lkw (Simulation)", delta_color="off")
st.plotly_chart(V.build_bars([i[:4] for i in items], selected=sel_shown), width="stretch", key="exp3_chart")
st.caption(f"Verzögerung nach dem Wechsel {label.replace('lam', 'λ').replace('->', '→')} bei ARL₀ = 370 Lkw (Wartezeit, Simulation, Zufalls-Seed {satz0['seed']}; {satz0['R0']} Läufe ohne Änderung, {satz0['R1']} nach der Änderung; Zero-state).")
kind, text = EV.verdict_exp3(r3, satz0["ref"]["blocks"])
getattr(st, kind)(text)
left, right = st.columns(2)
with left:
    st.plotly_chart(V.build_block_lag(satz0["ref"]["blocks"]), width="stretch", key="exp3_lag_chart")
    st.caption("Lag-1-Autokorrelation der Beobachtungen der Karte: b = 1 ist die Wartezeit selbst, danach die Blockmittel (Referenzlauf, 1 000 Läufe × 8 000 Lkw).")
with right:
    iw = REF.iid_wartezeit()
    st.markdown("**Bezugsgrößen** (Wechsel 1,6 → 2,0, ARL₀ = 370, Rohkarte k = 0,139, Zero-state):")
    st.dataframe(pd.DataFrame([
        {"Welt": "unabhängig, Normalverteilung", "ARL₁ (Lkw)": de(iw["normal"]["arl1"], 1), "Art": "exakt (Integralgleichung)"},
        {"Welt": "unabhängig, exakte schiefe Randverteilung der Wartezeit", "ARL₁ (Lkw)": de(iw["exakt"]["800"]["arl1"], 1), "Art": "exakt (Zellenkette, 800 Zellen)"},
        {"Welt": "Gate (abhängig), Rohkarte", "ARL₁ (Lkw)": de(r3["roh"][0] if abs(S.groesse - C.LAM1) < 1e-9 else EV.sim_value("Roh k=0,14", "lam 1,6 -> 2,0")[0], 1), "Art": "Simulation"}]),
        hide_index=True, width="stretch")
    st.caption("Schiefe der Randverteilung erklärt nur einen kleinen Teil des Abstands; der größere kommt aus der Abhängigkeit.")
rows = []
for shown, v, se, _, nm in items:
    v2 = EV.sim_value(nm, label, 1)
    rows.append({"Karte": shown, "Schwelle h* (Satz 1)": de(satz0["karten"][nm]["h"], 2), f"ARL₁ ± Standardfehler (Satz 1, Zufalls-Seed {satz0['seed']})": f"{de(v, 1)} ± {de(se, 1)}",
                 f"zweiter Zufalls-Seed ({satz1['seed']})": f"{de(v2[0], 1)} ± {de(v2[1], 1)}" if v2 else "–"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown("**Live-Probe** (stationäres Gate, λ = 1,6, ohne Änderung): Autokorrelation, Blockmittel und der naive zweiseitige CUSUM (k = 0,5, h = 4,77) aus Experiment 1, jetzt auf der Wartezeit.")
with st.spinner(f"Simuliere {S.live_r} Gates à {C.LIVE_T} Lkw …"):
    _t0 = time.perf_counter()
    pr = _probe(S.live_r, C.LIVE_T, S.live_seed)
    _sec = time.perf_counter() - _t0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Naiver CUSUM: Lkw bis zum Fehlalarm", f"{de(pr['arl0'], 1)} ± {de(pr['se'], 1)}", delta=f"{de(C.NAIV_NOMINAL, 1)} bei unabhängigen Werten", delta_color="off")
c2.metric("Fehlalarm in den ersten 100 Lkw", pct(pr["p100"]), delta=f"{pr['R']} Läufe", delta_color="off")
c3.metric("Autokorrelation der Wartezeit, Lag 1 / 2 / 5", f"{de(pr['acf'][1], 2)} / {de(pr['acf'][2], 2)} / {de(pr['acf'][5], 2)}", delta="0,78 / 0,59 / 0,18 im Referenzlauf", delta_color="off")
c4.metric("Standardabweichung der Blockmittel b = 20, Faktor", de(pr["blocks"][20]["faktor"], 2), delta="gegen σ/√b, Lag-1 " + de(pr["blocks"][20]["lag1"], 2), delta_color="off")
st.caption(f"Live gerechnet in {de(_sec, 2)} s (Rechenzeit dieser Maschine, kein Befund; im Cache bei gleichen Reglern 0 s). {pr['R']} Läufe × {C.LIVE_T} Lkw, Zufalls-Seed {S.live_seed}; "
           f"Läufe ohne Alarm in {pr['T']} Lkw: {pct(pr['cens'])}. Bei wenigen Läufen streuen alle Zahlen (Standardfehler am Mittel).")
st.markdown(TX.EXP3_NEGATIV)

# =====================================================================================================================================================
# Experiment 4
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 4: Was bringt ein Modell der Schlange?")
st.markdown(TX.EXP4_FRAGE)
with st.spinner("Rechne die Karten bei ARL₀ = 370 (exakt) …"):
    r4 = _exp4(S.lam1_wahr, S.lam1_entw, S.lam0_wahr, S.qs)
lw = S.lam1_wahr
c1, c2, c3, c4 = st.columns(4)
c1.metric("ARL₁ Rohkarte (κ = 20)", de(r4["raw"]["arl1"][lw], 1), delta=f"Schwelle H = {de(r4['raw']['h'], 1)}", delta_color="off")
c2.metric("ARL₁ Modellkarte", de(r4["mod"]["arl1"][lw], 1), delta=f"entworfen für {de(S.lam1_entw, 1)}, H = {de(r4['mod']['h'], 1)}", delta_color="off")
c3.metric("Information je Lkw (Übergänge)", de(r4["i_trans"], 5), delta=f"Randverteilung: {de(r4['i_marg'], 5)}", delta_color="off")
c4.metric("ln(370)/Information", de(r4["lorden"], 0) if r4["lorden"] else "–", delta="Näherung, keine Schranke", delta_color="off")
st.plotly_chart(V.build_exp4_arl1(r4["rows"], lw), width="stretch", key="exp4_chart")
st.caption("ARL₁ nach dem Wechsel 1,6 → λ₁ bei gleicher ARL₀ = 370 (exakt, Zero-state; Schwellen zwischen ganzzahligen Werten log-linear auf 370 interpoliert).")
kind, text = EV.verdict_exp4(r4)
getattr(st, kind)(text)
left, right = st.columns(2)
with left:
    st.dataframe(pd.DataFrame([{"wahres λ₁": de(z["lam"], 1), "Sprung (Sd der gesehenen Zahl)": de(z["sprung_sd"], 2), "Rohkarte": de(z["roh"], 1), "Modellkarte (Entwurf gewählt)": de(z["modell"], 1),
                                "Modellkarte für das wahre λ₁": de(z["eigen"], 1)} for z in r4["rows"]]), hide_index=True, width="stretch")
with right:
    st.plotly_chart(V.build_null(r4["null"], S.lam0_wahr), width="stretch", key="exp4_null_chart")
st.caption("Rechts: beide Karten fest für λ₀ = 1,6; die Wahrheit ohne Änderung ist λ₀′. Links: ARL₁ in Lkw je wahrem λ₁.")
st.markdown(TX.EXP4_NEGATIV)

# =====================================================================================================================================================
# Experiment 5
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🔬 Experiment 5: Welche Änderung sieht eine Residuenkarte?")
st.markdown(TX.EXP5_FRAGE)
r5 = EV.exp5(S.aenderung)
c1, c2, c3 = st.columns(3)
c1.metric("Residuenkarte", f"{de(r5['resid'][0], 1)} ± {de(r5['resid'][1], 1)}", delta=f"zweiter Zufalls-Seed: {de(r5['resid2'][0], 1)}", delta_color="off")
c2.metric("Rohkarte (Wartezeit)", f"{de(r5['roh'][0], 1)} ± {de(r5['roh'][1], 1)}", delta=f"zweiter Zufalls-Seed: {de(r5['roh2'][0], 1)}", delta_color="off")
c3.metric("Karte auf der gesehenen Zahl", f"{de(r5['n'][0], 1)} ± {de(r5['n'][1], 1)}", delta="Simulation", delta_color="off")
st.plotly_chart(V.build_bars([("Residuenkarte (W gegeben n)", r5["resid"][0], r5["resid"][1], C.COLOR_RESID), ("Rohkarte auf der Wartezeit", r5["roh"][0], r5["roh"][1], C.COLOR_ROH),
                              ("Karte auf der gesehenen Zahl", r5["n"][0], r5["n"][1], C.COLOR_N)]), width="stretch", key="exp5_chart")
st.caption(f"Verzögerung nach der Änderung ({S.aenderung}) bei ARL₀ = 370 Lkw (Simulation, Zufalls-Seed {satz0['seed']}, Zero-state).")
kind, text = EV.verdict_exp5(r5)
getattr(st, kind)(text)
st.markdown(TX.EXP5_NEGATIV)

# =====================================================================================================================================================
# Grenzen und Formeln
# =====================================================================================================================================================
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_MARKDOWN)
st.markdown(TX.VEHIKEL)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_INTRO)
    st.markdown(TX.CONVENTIONS_MARKDOWN)
    st.markdown(TX.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
