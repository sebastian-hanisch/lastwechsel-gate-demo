"""Kennzahlen und Verdikte der fünf Experimente. Verdikte sind reine Funktionen (Art, Text): die App ruft st.<Art>(Text), `tests/test_verdicts.py` prüft jede Zahl und
jede Schwelle an handgesetzten Werten. Alle Zahlen: Zähleinheit angenommene Lkw; ARL₀ ohne, ARL₁ mit Änderung (Zero-state), wenn nicht anders gesagt."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import lwg_cards as CA
import lwg_chain as CH
import lwg_constants as C
import lwg_reference as REF


# ------------------------------------------------------------------------------------------------------------------------------ Format
def de(x: float, nd: int = 1) -> str:
    """Deutsche Schreibweise: Komma als Dezimalzeichen, Leerzeichen als Tausendertrenner."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")


def pct(p: float, nd: int = 1) -> str:
    return de(100 * p, nd) + " %"


def de_s(x: float, nd: int = 1) -> str:
    """Mit Vorzeichen."""
    return ("+" if x >= 0 else "−") + de(abs(x), nd)


def lam_s(x: float) -> str:
    return de(x, 2 if round(x * 10) != x * 10 else 1)


@dataclass(frozen=True)
class Settings:
    h: int = C.DEFAULT_H
    kappa: int = C.DEFAULT_KAPPA
    target: int = C.DEFAULT_TARGET
    lam1: float = C.DEFAULT_LAM1
    block: int = C.DEFAULT_BLOCK
    verfahren: str = C.DEFAULT_VERFAHREN
    groesse: float = C.DEFAULT_GROESSE
    live_r: int = C.DEFAULT_LIVE_R
    live_seed: int = C.DEFAULT_LIVE_SEED
    lam1_wahr: float = C.DEFAULT_LAM1
    lam1_entw: float = C.DEFAULT_LAM1
    lam0_wahr: float = C.DEFAULT_LAM0_WAHR
    qs: int = C.DEFAULT_QS
    aenderung: str = C.DEFAULT_AENDERUNG


@lru_cache(maxsize=None)
def gate(lam: float) -> CH.Gate:
    return CH.Gate(lam)


@lru_cache(maxsize=None)
def iid_chain(lam_pre: float, lam_post: float) -> tuple:
    """Die 'unabhängige' Welt: jede Zeile = Randverteilung der gesehenen Zahl (Lehrbuchrechnung), vor und nach der Änderung."""
    K = C.K_CAP
    return np.tile(gate(lam_pre).pi, (K, 1)), np.tile(gate(lam_post).pi, (K, 1))


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 1
def exp1(h: int, kappa: int, tmax: int = 300) -> dict:
    """Karte auf der gesehenen Zahl (Skala 8, Referenzwert κ, Schwelle H): ARL₀ unter der Annahme 'unabhängig' und auf der tatsächlichen Kette, Überlebenskurven.
    ARL₀ None, wenn sie über der Reglergrenze liegt (2·10⁵) oder der Löser keine verlässliche Zahl liefert."""
    g0, g1 = gate(C.LAM0), gate(C.LAM1)
    S = CA.raw_score(C.Q, kappa)
    Riid0, Riid1 = iid_chain(C.LAM0, C.LAM1)
    tmax = max(tmax, 100)       # die Anzeige „Alarme in den ersten 100 Lkw“ braucht die Kurve bis 100
    out = {"h": h, "kappa": kappa}
    for key, R0, R1 in (("iid", Riid0, Riid1), ("kette", g0.R, g1.R)):
        try:
            out["arl0_" + key] = CA.exact_card(R0, R0, S, h, g0.pi, guard=True)["arl0"]
        except CA.ArlTooLarge:
            out["arl0_" + key] = None
        Q = CA.build_Q(R0, S, h)
        s = CA.surv_curve(Q, CA.start_vec(g0.pi, h), tmax)
        out["surv_" + key] = s
        out["p100_" + key] = float(1 - s[100])
    a, b = out["arl0_iid"], out["arl0_kette"]
    out["faktor"] = a / b if a is not None and b is not None else None
    return out


def verdict_exp1(r: dict) -> tuple[str, str]:
    h, a, b, f = r["h"], r["arl0_iid"], r["arl0_kette"], r["faktor"]
    kopf = f"Karte auf der gesehenen Zahl (κ = {r['kappa']}, k = {de(r['kappa'] / C.Q, 2)}), Schwelle H = {h}: "
    if b is None:
        return "warning", kopf + "Auch auf der tatsächlichen Kette liegt ARL₀ über 2·10⁵ Lkw (keine Zahl: der Löser ist dort nicht verlässlich). Die Schwelle ist für diese Rechnung zu hoch."
    if a is None:
        return "info", (kopf + f"Mit der Randverteilung der gesehenen Zahl und unabhängigen Lkw läge ARL₀ über 2·10⁵ Lkw (keine Zahl). Auf der tatsächlichen Kette sind es "
                        f"{de(b, 1)} Lkw (Alarme in den ersten 100 Lkw: {pct(r['p100_kette'])}; exakt, Zero-state).")
    text = (kopf + f"Mit der Randverteilung der gesehenen Zahl und unabhängigen Lkw hat die Schwelle ARL₀ = {de(a, 1)} Lkw (Alarme in den ersten 100 Lkw: {pct(r['p100_iid'])}). "
            f"Auf der tatsächlichen Kette sind es {de(b, 1)} Lkw ({pct(r['p100_kette'])}), Faktor {de(f, 1)} (exakt, Zero-state).")
    if f >= 1.5:
        return "warning", text
    if f <= 1 / 1.5:
        return "info", text + " Hier kehrt sich der Faktor um: bei kleinen Schwellen meldet die Kette seltener als die Rechnung für unabhängige Lkw."
    return "info", text + " Hier liegen beide Rechnungen nahe beieinander (Faktor zwischen zwei Dritteln und 1,5)."


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 2
def exp2(kappa: int, target: int, lam1: float) -> dict:
    """Schwelle auf ARL₀ = target eingestellt: Verzögerung nach dem Wechsel 1,6 → lam1 für die Annahme 'unabhängig' und für die Kette (aus den gefrorenen Kennlinien),
    Steady-state (nur 1,6 → 2,0), Verteilung der Verzögerung an der ganzzahligen Schwelle (live, exakt)."""
    g0 = gate(C.LAM0)
    iid = REF.at_target_zero(kappa, "iid", lam1, float(target))
    kette = REF.at_target_zero(kappa, "kette", lam1, float(target))
    out = {"kappa": kappa, "target": target, "lam1": lam1, "iid": iid, "kette": kette, "steady": REF.steady_at_target(kappa, float(target)) if lam1 == C.LAM1 else None,
           "sd_n": g0.sd_n, "dist": None}
    if kette is not None:
        S = CA.raw_score(C.Q, kappa)
        Hn = int(np.ceil(kette[0]))
        Hn = max(Hn, 3)
        e = CA.exact_card(g0.R, gate(lam1).R, S, Hn, g0.pi)
        tmax = int(min(8000, max(300, 6 * e["arl1"])))
        e = CA.exact_card(g0.R, gate(lam1).R, S, Hn, g0.pi, tmax=tmax)
        s0, s1 = e["surv0"], e["surv1"]
        q95 = int(np.argmax(s1 <= 0.05)) if (s1 <= 0.05).any() else None
        out["dist"] = {"H": Hn, "arl0": e["arl0"], "arl1": e["arl1"], "median": int(np.argmax(s1 <= 0.5)), "q95": q95, "tmax": tmax,
                       "p50": float(1 - s1[50]), "p100": float(1 - s1[100]), "p200": float(1 - s1[200]), "p100_ohne": float(1 - s0[100]), "surv1": s1, "surv0": s0}
    return out


def k_summary() -> dict:
    """Zusammenfassung der acht gemessenen k bei ARL₀ = 370 (Zero-state)."""
    z = REF.k_reihe()
    best_k = min(z, key=lambda r: r["arl1_kette"])
    best_i = min(z, key=lambda r: r["arl1_iid"])
    return {"n": len(z), "k_min": z[0]["k"], "k_max": z[-1]["k"], "v_min": min(r["verhaeltnis"] for r in z), "v_max": max(r["verhaeltnis"] for r in z),
            "best_kette": best_k["arl1_kette"], "best_kette_k": best_k["k"], "best_iid": best_i["arl1_iid"], "best_iid_k": best_i["k"],
            "faktor_beste": best_k["arl1_kette"] / best_i["arl1_iid"]}


def verdict_exp2(r: dict) -> tuple[str, str]:
    if r["iid"] is None or r["kette"] is None:
        return "warning", (f"Für ARL₀ = {r['target']} liegt die Schwelle (κ = {r['kappa']}) außerhalb des gerechneten Bereichs der gefrorenen Kennlinien "
                           "(Schwelle bis 1 200 bzw. ARL₀ bis 2·10⁵): keine Zahl.")
    (hi, ai), (hk, ak) = r["iid"], r["kette"]
    sd = r["sd_n"] * C.Q
    t = (f"Mit angepasster Schwelle (H = {de(hk, 1)} = {de(hk / sd, 1)} Standardabweichungen der gesehenen Zahl statt {de(hi, 1)} = {de(hi / sd, 1)}) hat die Karte wieder "
         f"ARL₀ = {r['target']} (zwischen den Rasterschwellen interpoliert), aber nach dem Wechsel 1,6 → {lam_s(r['lam1'])} im Mittel {de(ak, 1)} Lkw statt {de(ai, 1)} (Faktor {de(ak / ai, 1)}; exakt, Zero-state")
    if r["steady"] is not None and r["steady"]["gleich"] is not None:
        t += f"; Steady-state {de(r['steady']['gleich'][0], 1)} an derselben Schwelle" + (f" bzw. {de(r['steady']['eq'], 1)} bei Steady-state-ARL₀ = {r['target']}" if r["steady"]["eq"] else "")
    t += ")."
    d = r["dist"]
    if d is not None:
        t += (f" An der ganzzahligen Schwelle H = {d['H']} (ARL₀ = {de(d['arl0'], 1)}) schlagen in 50 / 100 / 200 Lkw nach dem Wechsel {de(100 * d['p50'], 1)} / "
              f"{de(100 * d['p100'], 1)} / {de(100 * d['p200'], 1)} % der Läufe an, im Mittel nach {de(d['arl1'], 1)} Lkw (Median {d['median']}, 95-%-Quantil {d['q95'] if d['q95'] is not None else 'über ' + str(d['tmax'])}).")
    t += " Die angepasste Schwelle stellt das Versprechen wieder her, nicht die Empfindlichkeit."
    return ("warning" if ak / ai >= 1.5 else "info"), t


def verdict_k_reihe(s: dict) -> tuple[str, str]:
    return "info", (f"Bei keinem der {s['n']} gemessenen k von {de(s['k_min'], 2)} bis {de(s['k_max'], 2)} holt die Kette den Wert der unabhängigen Rechnung: sie ist "
                    f"{de(s['v_min'], 1)}- bis {de(s['v_max'], 1)}-fach langsamer als die Rechnung für unabhängige Lkw desselben k (bei k ≈ 4 und 4,5 nur 1,1- bis 1,3-fach, dort sind aber beide Karten "
                    f"langsam; die Verhältnisse fallen mit k Richtung 1), und die beste der gemessenen Karten der Kette (k = {de(s['best_kette_k'], 2)}: {de(s['best_kette'], 1)} Lkw) ist {de(s['faktor_beste'], 1)}-mal langsamer als die beste "
                    f"gemessene für unabhängige Lkw (k = {de(s['best_iid_k'], 2)}: {de(s['best_iid'], 1)}). Gilt für die Rohkarte auf der gesehenen Zahl, ARL₀ = 370, Zero-state, Wechsel 1,6 → 2,0.")


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 3
def sim_value(name: str, label: str, i: int = 0):
    """[ARL₁, Standardfehler] der Karte `name` im Szenario `label` aus Satz i oder None."""
    rec = REF.satz(i)["karten"].get(name)
    return None if rec is None else rec["arl1"].get(label)


def card_name(verfahren: str, b: int) -> str | None:
    """Name der gemessenen Karte zu (Verfahren, b); b = 1 ist die Rohkarte. None, wenn die Kombination nicht gemessen ist (Ausdünnen jeder 40.)."""
    if b == 1:
        return "Roh k=0,14"
    if verfahren == C.VERFAHREN[0]:
        return f"Blockmittel b={b}"
    return f"Ausduennen jeder {b}." if b in (5, 10, 20) else None


def exp3(verfahren: str, b: int, groesse: float) -> dict:
    label = f"lam 1,6 -> {de(groesse, 1)}"
    name = card_name(verfahren, b)
    return {"name": name, "label": label, "verfahren": verfahren, "b": b, "groesse": groesse,
            "sel": sim_value(name, label) if name else None, "sel2": sim_value(name, label, 1) if name else None,
            "roh": sim_value("Roh k=0,14", label), "roh2": sim_value("Roh k=0,14", label, 1),
            "n": sim_value("Karte auf n (sim, kappa 20)", label),
            "phase": (REF.satz(0)["phase"].get(name) if (name and verfahren == C.VERFAHREN[0] and b > 1 and abs(groesse - C.LAM1) < 1e-9) else None),
            "phase2": (REF.satz(1)["phase"].get(name) if (name and verfahren == C.VERFAHREN[0] and b > 1 and abs(groesse - C.LAM1) < 1e-9) else None)}


def verdict_exp3(r: dict, block_stats: dict | None = None) -> tuple[str, str]:
    if r["name"] is None:
        return "info", "Ausdünnen jeder 40. Beobachtung ist nicht gemessen (gemessen: jeder 5., 10. und 20.)."
    sel, roh = r["sel"], r["roh"]
    lab = f"1,6 → {de(r['groesse'], 1)}"
    if r["b"] == 1:
        return "info", f"Rohkarte (jeder Lkw, k = Sprung/2) bei ARL₀ = 370 Lkw, Wechsel {lab}: {de(roh[0], 1)} Lkw (± {de(roh[1], 1)}; Wartezeit, Simulation)."
    diff = sel[0] - roh[0]
    se = float(np.hypot(sel[1], roh[1]))
    wort = "Blockmittel" if r["verfahren"] == C.VERFAHREN[0] else "Ausdünnen jeder"
    tail = ""
    if block_stats and r["verfahren"] == C.VERFAHREN[0] and str(r["b"]) in block_stats:
        bs = block_stats[str(r["b"])]
        tail = f" Die Blockmittel sind fast unabhängig (Lag-1-Autokorrelation {de(bs['lag1'], 2)}), aber die Verzögerung bleibt."
        if r.get("phase"):
            tail += (f" Liegt die Änderung nicht auf einer Blockgrenze (b/2 = {r['phase']['phase']} Lkw des ersten Blocks vor dem Wechsel), sind es {de(r['phase']['arl1'][0], 1)} Lkw "
                     f"(± {de(r['phase']['arl1'][1], 1)}); gemessen sind nur diese beiden Phasen (Blockgrenze und Blockmitte), weitere nicht.")
    if abs(diff) <= 2 * se:
        return "info", (f"{wort} b = {r['b']}: {de(sel[0], 1)} Lkw (± {de(sel[1], 1)}), Rohkarte {de(roh[0], 1)} (± {de(roh[1], 1)}) bei ARL₀ = 370 Lkw, Wechsel {lab}: der Unterschied von "
                        f"{de_s(diff, 1)} Lkw ist nicht aufgelöst (höchstens 2 Standardfehler der Differenz, {de(2 * se, 1)}).{tail}")
    return "warning", (f"{wort} b = {r['b']}: {de(sel[0], 1)} Lkw (± {de(sel[1], 1)}) gegen Rohkarte {de(roh[0], 1)} (± {de(roh[1], 1)}) bei ARL₀ = 370 Lkw, Wechsel {lab}: "
                       f"{de_s(diff, 1)} Lkw ({de(abs(diff) / se, 1)} Standardfehler der Differenz), also {'langsamer' if diff > 0 else 'schneller'}.{tail}")


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 4
@lru_cache(maxsize=None)
def raw_card_370() -> dict:
    """Rohkarte auf n (κ = 20) bei ARL₀ = 370, ARL₁ für alle λ₁ (interpoliert)."""
    g0 = gate(C.LAM0)
    S = CA.raw_score(C.Q, 20)
    return CA.at_target_exact(g0.R, {lam: gate(lam).R for lam in C.LAM1_WAHR_OPTIONS}, S, g0.pi) | {"S": S}


@lru_cache(maxsize=None)
def model_card_370(lam_design: float, qs: int) -> dict:
    """Likelihood-Quotient-Karte der Übergänge (Score ln R_entw/R₀ auf dem Gitter 1/qs) bei ARL₀ = 370, ARL₁ für alle λ₁."""
    g0 = gate(C.LAM0)
    S, llr = CA.lr_score(gate(lam_design).R, g0.R, qs)
    return CA.at_target_exact(g0.R, {lam: gate(lam).R for lam in C.LAM1_WAHR_OPTIONS}, S, g0.pi, hi=600) | {"S": S, "llr": llr}


def information(lam_true: float, lam_design: float) -> tuple[float, float]:
    """Kullback-Information je Lkw unter der wahren Rate: (Likelihood-Quotient der Übergänge, Randverteilung der gesehenen Zahl)."""
    g0, gt, gd = gate(C.LAM0), gate(lam_true), gate(lam_design)
    llr = CA.lr_score(gd.R, g0.R, 1)[1]  # ln R_entw/R₀ dort, wo R₀ > 0 (von n sind nur Zahlen bis n + 1 erreichbar), sonst 0
    i_trans = float((gt.pi[:, None] * gt.R * llr).sum())
    i_marg = float((gt.pi * np.log(gt.pi / g0.pi)).sum())
    return i_trans, i_marg


def exp4(lam1_wahr: float, lam1_entw: float, lam0_wahr: float, qs: int) -> dict:
    raw = raw_card_370()
    mod = model_card_370(lam1_entw, qs)
    own = model_card_370(lam1_wahr, qs)
    i_trans, i_marg = information(lam1_wahr, lam1_entw)
    g0 = gate(C.LAM0)
    rows = []
    for lam in C.LAM1_WAHR_OPTIONS:
        sh = (gate(lam).mean_n - g0.mean_n) / g0.sd_n
        rows.append({"lam": lam, "sprung_sd": sh, "roh": raw["arl1"][lam], "modell": model_card_370(lam1_entw, qs)["arl1"][lam], "eigen": model_card_370(lam, qs)["arl1"][lam]})
    # falsches Nullmodell: Karten fest (ganzzahlige Schwelle H), die Wahrheit ohne Änderung ist lam0'
    null = []
    for lam0 in C.LAM0_WAHR_OPTIONS:
        gt = gate(lam0)
        a_raw = CA.exact_card(gt.R, gt.R, raw["S"], raw["h_int"], gt.pi)["arl0"]
        a_mod = CA.exact_card(gt.R, gt.R, mod["S"], mod["h_int"], gt.pi)["arl0"]
        null.append({"lam0": lam0, "roh": a_raw, "faktor_roh": a_raw / raw["arl0_hi"], "modell": a_mod, "faktor_modell": a_mod / mod["arl0_hi"]})
    return {"raw": raw, "mod": mod, "own": own, "rows": rows, "null": null, "i_trans": i_trans, "i_marg": i_marg,
            "lorden": float(np.log(C.ARL0_NENN) / i_trans) if i_trans > 0 else None, "lam1_wahr": lam1_wahr, "lam1_entw": lam1_entw, "lam0_wahr": lam0_wahr, "qs": qs}


def verdict_exp4(r: dict) -> tuple[str, str]:
    lw, le, q = r["lam1_wahr"], r["lam1_entw"], r["qs"]
    a_raw, a_mod = r["raw"]["arl1"][lw], r["mod"]["arl1"][lw]
    a_own = r["own"]["arl1"][lw]
    null = next(z for z in r["null"] if z["lam0"] == r["lam0_wahr"])
    t = (f"Bei gleicher ARL₀ = 370 Lkw braucht die Rohkarte auf der gesehenen Zahl nach dem Wechsel 1,6 → {lam_s(lw)} {de(a_raw, 1)} Lkw, die Likelihood-Quotient-Karte der Übergänge, "
         f"entworfen für {lam_s(le)} (Gitter 1/{q}), {de(a_mod, 1)}; für das wahre {lam_s(lw)} entworfen {de(a_own, 1)}. ")
    if r["lorden"] is not None:
        t += (f"Die Näherung ln(370)/I mit der Information I = {de(r['i_trans'], 5)} je Lkw (Randverteilung der gesehenen Zahl: {de(r['i_marg'], 5)}) gibt {de(r['lorden'], 0)} Lkw, "
              f"das {de(r['lorden'] / a_own, 1)}-Fache des exakten Werts der für {lam_s(lw)} entworfenen Karte: keine Schranke. ")
    if abs(r["lam0_wahr"] - C.LAM0) < 1e-9:
        t += "Das Nullmodell stimmt (λ₀ = 1,6): ARL₀ der beiden Karten ist die nominelle."
    else:
        t += (f"Weicht die wahre Rate ohne Änderung ab (λ₀ = {lam_s(r['lam0_wahr'])} statt 1,6), liegt ARL₀ der Rohkarte bei {de(null['roh'], 1)} (Faktor {de(null['faktor_roh'], 2)}), "
              f"die der Modellkarte bei {de(null['modell'], 1)} (Faktor {de(null['faktor_modell'], 2)}).")
    kind = "warning" if (abs(r["lam0_wahr"] - C.LAM0) > 1e-9 and (null["faktor_roh"] < 0.7 or null["faktor_roh"] > 1.4)) else "info"
    return kind, t


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 5
def exp5(aenderung: str) -> dict:
    label = C.AENDERUNG_KEY[aenderung]
    return {"aenderung": aenderung, "label": label, "roh": sim_value("Roh k=0,14", label), "n": sim_value("Karte auf n (sim, kappa 20)", label),
            "resid": sim_value("Residuen W|n", label), "resid2": sim_value("Residuen W|n", label, 1), "roh2": sim_value("Roh k=0,14", label, 1)}


def verdict_exp5(r: dict) -> tuple[str, str]:
    roh, res = r["roh"], r["resid"]
    faktor = res[0] / roh[0]
    t = (f"{r['aenderung']}: Residuenkarte (Wartezeit gegeben die gesehene Zahl) {de(res[0], 1)} Lkw (± {de(res[1], 1)}), Rohkarte {de(roh[0], 1)} (± {de(roh[1], 1)}), "
         f"Karte auf der gesehenen Zahl {de(r['n'][0], 1)} (± {de(r['n'][1], 1)}), jeweils bei ARL₀ = 370 Lkw (Simulation, Zero-state). ")
    if faktor > 1.5:
        return "warning", t + (f"Die Residuenkarte braucht das {de(faktor, 1)}-Fache der Rohkarte: sie rechnet das Modell heraus, in dem diese Änderung steckt "
                               "(Deutung; gemessen ist nur das Verhalten dieser Regel).")
    if faktor < 1.15:
        wort = (f"etwa {de(100 * (faktor - 1), 0)} % langsamer als die Rohkarte" if faktor >= 1.005 else "nicht langsamer als die Rohkarte")
        return "success", t + f"Die Residuenkarte ist hier {wort} (Faktor {de(faktor, 2)}): die Änderung liegt nicht im herausgerechneten Modellteil (Deutung)."
    return "info", t + f"Die Residuenkarte braucht das {de(faktor, 2)}-Fache der Rohkarte."
