"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds; so prüft der Test die Grenzen (Faktor 1,5 und 2/3, zwei Standardfehler, Residuenfaktor 1,15 und 1,5)
und jede genannte Zahl direkt."""

import pytest

import lwg_evaluation as ev


def exp1(h=97, kappa=20, a=408.2, b=51.2, p_iid=0.196, p_kette=0.872):
    return {"h": h, "kappa": kappa, "arl0_iid": a, "arl0_kette": b, "p100_iid": p_iid, "p100_kette": p_kette, "faktor": (a / b if a is not None and b is not None else None)}


# ----------------------------------------------------------------------------------------------------------------- Format
def test_deutsche_schreibweise():
    assert ev.de(408.17, 1) == "408,2" and ev.de(1282.84, 1) == "1 282,8" and ev.de(0.01058, 5) == "0,01058" and ev.de(2.0, 0) == "2"
    assert ev.pct(0.8717) == "87,2 %" and ev.de_s(2.5) == "+2,5" and ev.de_s(0.0) == "+0,0"
    assert ev.de_s(-1.26, 1) == "−1,3" and ev.lam_s(1.6) == "1,6" and ev.lam_s(1.55) == "1,55" and ev.lam_s(4.0) == "4,0"


# ----------------------------------------------------------------------------------------------------------------- Experiment 1
def test_exp1_zusammenbruch_nennt_alle_zahlen():
    kind, text = ev.verdict_exp1(exp1())
    assert kind == "warning"
    for teil in ("κ = 20, k = 2,50", "Schwelle H = 97", "ARL₀ = 408,2 Lkw", "in den ersten 100 Lkw: 19,6 %", "sind es 51,2 Lkw (87,2 %)", "Faktor 8,0 (exakt, Zero-state)"):
        assert teil in text, teil


@pytest.mark.parametrize("a, b, art", [(15.0, 10.0, "warning"), (14.9, 10.0, "info"), (10.0, 10.0, "info"), (10.0, 15.0, "info"), (10.0, 15.1, "info")])
def test_exp1_schwellen_des_faktors(a, b, art):
    """Faktor ≥ 1,5: Zusammenbruch (warning); sonst info; Faktor ≤ 2/3 nennt die Umkehr."""
    kind, text = ev.verdict_exp1(exp1(a=a, b=b))
    assert kind == art
    assert ("kehrt sich der Faktor um" in text) == (a / b <= 1 / 1.5)
    assert ("nahe beieinander" in text) == (1 / 1.5 < a / b < 1.5)


def test_exp1_ohne_unabhaengige_zahl_und_ohne_jede_zahl():
    kind, text = ev.verdict_exp1(exp1(h=300, a=None, b=345.6, p_iid=0.0, p_kette=0.206))
    assert kind == "info" and "über 2·10⁵ Lkw (keine Zahl)" in text and "345,6 Lkw" in text and "20,6 %" in text and "Faktor" not in text
    kind, text = ev.verdict_exp1(exp1(h=700, kappa=28, a=None, b=None))
    assert kind == "warning" and "Auch auf der tatsächlichen Kette liegt ARL₀ über 2·10⁵ Lkw" in text


# ----------------------------------------------------------------------------------------------------------------- Experiment 2
def exp2(ai=37.2, ak=101.2, lam1=2.0, steady=True, dist=True, target=370):
    r = {"kappa": 20, "target": target, "lam1": lam1, "iid": (97.4, ai), "kette": (309.1, ak), "sd_n": 1.5655,
         "steady": {"gleich": (89.2, 342.6), "eq": 93.0} if steady else None,
         "dist": {"H": 310, "arl0": 378.1, "arl1": 102.4, "median": 80, "q95": 254, "tmax": 1000, "p50": 0.267, "p100": 0.619, "p200": 0.899} if dist else None}
    return r


def test_exp2_nennt_alle_zahlen():
    kind, text = ev.verdict_exp2(exp2())
    assert kind == "warning"
    for teil in ("H = 309,1 = 24,7 Standardabweichungen der gesehenen Zahl statt 97,4 = 7,8", "ARL₀ = 370", "1,6 → 2,0", "im Mittel 101,2 Lkw statt 37,2 (Faktor 2,7",
                 "Steady-state 89,2 an derselben Schwelle bzw. 93,0 bei Steady-state-ARL₀ = 370", "H = 310 (ARL₀ = 378,1)", "26,7 / 61,9 / 89,9 % der Läufe", "im Mittel nach 102,4 Lkw (Median 80, 95-%-Quantil 254)",
                 "stellt das Versprechen wieder her, nicht die Empfindlichkeit"):
        assert teil in text, teil


def test_exp2_schwelle_faktor_und_fehlende_teile():
    assert ev.verdict_exp2(exp2(ak=55.5, ai=37.0))[0] == "warning"           # Faktor 1,5 genau: warning
    assert ev.verdict_exp2(exp2(ak=55.4, ai=37.0))[0] == "info"
    kind, text = ev.verdict_exp2(exp2(steady=False, dist=False, lam1=2.4))
    assert "Steady-state" not in text and "ganzzahligen Schwelle" not in text and "1,6 → 2,4" in text
    r = exp2()
    r["dist"]["q95"] = None
    assert "95-%-Quantil über 1000" in ev.verdict_exp2(r)[1]
    r["steady"]["eq"] = None
    assert "bzw." not in ev.verdict_exp2(r)[1]


def test_exp2_ziel_ausserhalb_des_bereichs():
    r = exp2()
    r["kette"] = None
    kind, text = ev.verdict_exp2(r)
    assert kind == "warning" and "außerhalb des gerechneten Bereichs" in text and "ARL₀ = 370" in text


def test_k_reihe_verdikt_nennt_die_zahlen():
    s = {"n": 8, "k_min": 2.25, "k_max": 4.47, "v_min": 1.128, "v_max": 2.719, "best_kette": 101.16, "best_kette_k": 2.5, "best_iid": 37.21, "best_iid_k": 2.5, "faktor_beste": 2.72}
    kind, text = ev.verdict_k_reihe(s)
    assert kind == "info"
    for teil in ("Bei keinem der 8 gemessenen k von 2,25 bis 4,47 holt die Kette den Wert", "1,1- bis 2,7-fach", "(k = 2,50: 101,2 Lkw)", "2,7-mal langsamer", "(k = 2,50: 37,2)", "ARL₀ = 370, Zero-state"):
        assert teil in text, teil


# ----------------------------------------------------------------------------------------------------------------- Experiment 3
def exp3(sel, roh=(134.7, 1.4), verfahren="Blockmittel", b=20, name="Blockmittel b=20", phase=None, groesse=2.0):
    return {"name": name, "label": "x", "verfahren": verfahren, "b": b, "groesse": groesse, "sel": sel, "roh": roh, "sel2": None, "roh2": None, "n": None, "phase": phase, "phase2": None}


def test_exp3_nicht_aufgeloest_bis_zwei_standardfehler_der_differenz():
    se = (1.4 ** 2 + 1.4 ** 2) ** 0.5            # 1,98
    kind, text = ev.verdict_exp3(exp3((134.7 + 2 * se, 1.4)))
    assert kind == "info" and "nicht aufgelöst" in text and "+4,0 Lkw" in text and "b = 20: 138,7 Lkw (± 1,4), Rohkarte 134,7 (± 1,4)" in text
    kind, text = ev.verdict_exp3(exp3((134.7 + 2 * se + 0.01, 1.4)))
    assert kind == "warning" and "langsamer" in text and "Standardfehler der Differenz" in text and "nicht aufgelöst" not in text
    kind, text = ev.verdict_exp3(exp3((134.7 - 2 * se - 0.01, 1.4)))
    assert kind == "warning" and "schneller" in text and "−4,0 Lkw" in text


def test_exp3_rohkarte_ausduennen_und_nicht_gemessen():
    kind, text = ev.verdict_exp3(exp3((134.7, 1.4), b=1, name="Roh k=0,14"))
    assert kind == "info" and text.startswith("Rohkarte (jeder Lkw, k = Sprung/2) bei ARL₀ = 370 Lkw, Wechsel 1,6 → 2,0: 134,7 Lkw (± 1,4;")
    kind, text = ev.verdict_exp3(exp3((166.7, 1.8), verfahren="Ausdünnen (jeder b-te)", b=10, name="Ausduennen jeder 10."))
    assert kind == "warning" and text.startswith("Ausdünnen jeder b = 10: 166,7 Lkw")
    kind, text = ev.verdict_exp3(exp3(None, name=None, verfahren="Ausdünnen (jeder b-te)", b=40))
    assert kind == "info" and "jeder 40. Beobachtung ist nicht gemessen (gemessen: jeder 5., 10. und 20.)" in text


def test_exp3_blockstatistik_und_phase_stehen_nur_bei_blockmitteln_und_passenden_daten():
    bs = {"20": {"lag1": 0.0799}}
    text = ev.verdict_exp3(exp3((135.4, 1.4)), bs)[1]
    assert "Lag-1-Autokorrelation 0,08" in text and "Liegt die Änderung nicht" not in text
    text = ev.verdict_exp3(exp3((135.4, 1.4), phase={"phase": 10, "arl1": [130.4, 1.46]}), bs)[1]
    assert "b/2 = 10 Lkw des ersten Blocks" in text and "130,4 Lkw (± 1,5)" in text and "nur diese beiden Phasen" in text
    text = ev.verdict_exp3(exp3((166.7, 1.8), verfahren="Ausdünnen (jeder b-te)", b=10, name="Ausduennen jeder 10."), bs)[1]
    assert "Lag-1" not in text


def test_exp3_kartennamen():
    assert ev.card_name("Blockmittel", 1) == "Roh k=0,14" and ev.card_name("Ausdünnen (jeder b-te)", 1) == "Roh k=0,14"
    assert ev.card_name("Blockmittel", 40) == "Blockmittel b=40" and ev.card_name("Ausdünnen (jeder b-te)", 20) == "Ausduennen jeder 20."
    assert ev.card_name("Ausdünnen (jeder b-te)", 40) is None


# ----------------------------------------------------------------------------------------------------------------- Experiment 4
def test_exp4_verdikt_an_echten_werten_nennt_jede_zahl_und_die_art():
    r = ev.exp4(2.0, 2.0, 1.6, 40)
    kind, text = ev.verdict_exp4(r)
    assert kind == "info"
    for teil in ("1,6 → 2,0 101,2 Lkw", "entworfen für 2,0 (Gitter 1/40), 102,3", "für das wahre 2,0 entworfen 102,3", "Information I = 0,01058 je Lkw (Randverteilung der gesehenen Zahl: 0,06362) gibt 559 Lkw",
                 "das 5,5-Fache des exakten Werts", "keine Schranke", "Das Nullmodell stimmt (λ₀ = 1,6)"):
        assert teil in text, teil
    kind, text = ev.verdict_exp4(ev.exp4(2.0, 2.0, 1.7, 40))
    assert kind == "warning" and "λ₀ = 1,7 statt 1,6" in text and "Rohkarte bei 244,9 (Faktor 0,65)" in text and "Modellkarte bei 244,1 (Faktor 0,64)" in text
    kind, text = ev.verdict_exp4(ev.exp4(2.0, 2.0, 1.65, 40))
    assert kind == "info" and "Faktor 0,80" in text                          # 0,80 liegt zwischen 0,7 und 1,4: nur info
    kind, text = ev.verdict_exp4(ev.exp4(2.0, 2.0, 1.5, 40))
    assert kind == "warning" and "Faktor 1,70" in text                       # über 1,4: warning (ARL₀ steigt)


def test_exp4_andere_entwuerfe():
    r = ev.exp4(4.0, 2.0, 1.6, 40)
    text = ev.verdict_exp4(r)[1]
    assert "1,6 → 4,0 26,6 Lkw" in text and "entworfen für 2,0 (Gitter 1/40), 28,1" in text and "für das wahre 4,0 entworfen 22,4" in text


# ----------------------------------------------------------------------------------------------------------------- Experiment 5
def exp5(resid, roh=(134.7, 1.4), n=(101.3, 1.0), name="Ankunftsrate 1,6 → 2,0"):
    return {"aenderung": name, "label": "x", "roh": roh, "n": n, "resid": resid, "resid2": None, "roh2": None}


@pytest.mark.parametrize("resid, art, satz", [(134.7 * 1.5 + 0.01, "warning", "braucht das 1,5-Fache"), (134.7 * 1.5, "info", "braucht das 1,50-Fache"), (134.7 * 1.15, "info", "braucht das 1,15-Fache"),
                                              (134.7 * 1.15 - 0.01, "success", "15 % langsamer als die Rohkarte"), (134.7 * 0.9, "success", "nicht langsamer als die Rohkarte (Faktor 0,90)")])
def test_exp5_schwellen_1_15_und_1_5(resid, art, satz):
    kind, text = ev.verdict_exp5(exp5((resid, 3.5)))
    assert kind == art and satz in text


def test_exp5_nennt_alle_zahlen_und_die_deutung_als_deutung():
    kind, text = ev.verdict_exp5(exp5((269.6, 3.5)))
    for teil in ("Ankunftsrate 1,6 → 2,0:", "269,6 Lkw (± 3,5)", "Rohkarte 134,7 (± 1,4)", "gesehenen Zahl 101,3 (± 1,0)", "bei ARL₀ = 370 Lkw (Simulation, Zero-state)", "das 2,0-Fache", "(Deutung; gemessen ist nur das Verhalten dieser Regel)"):
        assert teil in text, teil
    assert "blind" not in text
