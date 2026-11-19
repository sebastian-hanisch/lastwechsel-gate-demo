"""Claims: jede Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Aufbau: `claim("408,2", wert, tol)` prüft zweierlei: die Zeichenkette steht in der README, und der gerechnete Wert liegt höchstens `tol` neben der dort genannten Zahl.
Exakte Zahlen (Ketten) kommen aus der Rechnung der Demo (`lwg_evaluation`, `lwg_reference`); die Orakel-Tests in `test_oracle_lastwechsel_gate.py` und die Handbeispiele prüfen
dieselben Ketten auf anderen Wegen. Zahlen der Messreihen (gelb) kommen aus `lwg_sim.json`; `test_tables.py` prüft sie gegen die exakte Kette und gegen eine Neu-Messung."""

import math
import re
from pathlib import Path

import pytest

import lwg_chain as CH
import lwg_constants as C
import lwg_evaluation as EV
import lwg_iid as IID
import lwg_reference as REF
import lwg_texts as TX

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
G0, G1 = CH.Gate(1.6), CH.Gate(2.0)


def num(text: str, nth: int = 0) -> float:
    """Die nth Zahl im Text (deutsche Schreibweise, Leerzeichen als Tausendertrenner)."""
    m = re.findall(r"\d+(?:[  ]\d{3})*(?:,\d+)?", text)[nth]
    return float(m.replace(" ", "").replace(" ", "").replace(",", "."))


def claim(text: str, value: float, tol: float, nth: int = 0):
    assert text in README, f"nicht in der README: {text!r}"
    assert abs(num(text, nth) - value) <= tol, (text, value)


# ----------------------------------------------------------------------------------------------------------------- Zusammenbruch (exakt)
def test_zusammenbruch_h97():
    r = EV.exp1(97, 20)
    claim("ARL₀ = 408,2", r["arl0_iid"], 0.05)
    claim("auf der Kette **51,2**", r["arl0_kette"], 0.05)
    claim("(Faktor 8,0)", r["faktor"], 0.05)
    claim("19,6 % gegen **87,2 %**", 100 * r["p100_iid"], 0.05)
    claim("**87,2 %**", 100 * r["p100_kette"], 0.05)


def test_zusammenbruch_faktor_waechst_ab_h30_und_kehrt_sich_bei_h20_um():
    f = {h: EV.exp1(h, 20)["faktor"] for h in (20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250)}
    assert f[20] < 1 < f[30] and [f[h] for h in (30, 40, 50, 60, 80, 100, 130, 160, 200, 250)] == sorted(f[h] for h in (30, 40, 50, 60, 80, 100, 130, 160, 200, 250))
    claim("kehrt er sich um (0,57)", f[20], 0.005)


def test_tabelle_je_schwelle():
    """Jede Zeile der README-Tabelle „Der Faktor je Schwelle“ gleich der Rechnung (auf die angezeigten Stellen)."""
    for h in (20, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300):
        r = EV.exp1(h, 20)
        iid = EV.de(r["arl0_iid"], 1) if r["arl0_iid"] is not None else "über 2·10⁵"
        fak = EV.de(r["faktor"], 2) if r["faktor"] is not None else "–"
        zeile = f"| {h} | {iid} | {EV.de(r['arl0_kette'], 1)} | {fak} | {EV.pct(r['p100_iid'])} / {EV.pct(r['p100_kette'])} |"
        assert zeile in README, zeile


def test_naive_nenngroesse_und_schwellen_der_shewhart_linie():
    """Naiver zweiseitiger CUSUM k = 0,5, h = 4,77: Näherung ARL⁺/2 = 368,6 bei unabhängigen Werten; einseitig h = 4,0954 für 370 und 4,0965 für 370,398."""
    claim("nominell 368,6", IID.normal_arl(0.5, C.V2_H) / 2, 0.05)
    assert C.NAIV_NOMINAL == pytest.approx(IID.normal_arl(0.5, C.V2_H) / 2, abs=0.05)
    claim("die Schwelle 4,0954", IID.normal_h_for_arl(0.5, 370.0), 5e-5)
    claim("(für 370,398: 4,0965)", IID.normal_h_for_arl(0.5, 370.398), 5e-5, nth=1)


# ----------------------------------------------------------------------------------------------------------------- Schwelle angepasst (exakt)
def test_schwelle_angepasst():
    r = EV.exp2(20, 370, 2.0)
    (hi, ai), (hk, ak) = r["iid"], r["kette"]
    claim("**101,2** statt 37,2", ak, 0.05)
    claim("statt 37,2 Lkw", ai, 0.05)
    claim("(Faktor 2,7)", ak / ai, 0.05)
    claim("89,2 an derselben Schwelle", r["steady"]["gleich"][0], 0.05)
    claim("(Steady-ARL₀ dort 342,6)", r["steady"]["gleich"][1], 0.05)
    claim("bzw. 93,0 bei", r["steady"]["eq"], 0.05)
    sd = C.Q * G0.sd_n
    claim("Schwelle 24,7 statt", hk / sd, 0.05)
    claim("statt 7,8 Standardabweichungen", hi / sd, 0.05)
    d = r["dist"]
    assert d["H"] == 310                                                    # ganzzahlige Schwelle zum Ziel (309 ≡ 310 auf dem Raster der Scores)
    claim("ARL₀ = 378,1", d["arl0"], 0.05)
    claim("26,7 / 61,9 / 89,9 %", 100 * d["p50"], 0.05)
    assert abs(100 * d["p100"] - 61.9) < 0.05 and abs(100 * d["p200"] - 89.9) < 0.05
    claim("Mittel 102,4", d["arl1"], 0.05)
    assert d["median"] == 80 and d["q95"] == 254 and "Median 80, 95-%-Quantil 254" in README


def test_referenzwerte_fuenf_kappa_und_acht_k():
    zeilen = {20: "| 20 (2,50) | 97,4 | 37,2 | 309,1 | 101,2 | 89,2 (342,6) | 93,0 | 2,72 |", 22: "| 22 (2,75) | 68,1 | 41,7 | 231,4 | 104,2 | 95,8 (351,0) | 98,7 | 2,50 |",
              24: "| 24 (3,00) | 51,0 | 49,5 | 177,6 | 108,4 | 102,0 (355,6) | 104,5 | 2,19 |", 26: "| 26 (3,25) | 40,8 | 58,0 | 135,6 | 112,7 | 107,5 (358,3) | 109,8 | 1,94 |",
              28: "| 28 (3,50) | 32,1 | 64,2 | 103,0 | 116,9 | 112,5 (360,3) | 114,5 | 1,82 |"}
    for kappa, zeile in zeilen.items():
        hi, ai = REF.at_target_zero(kappa, "iid", 2.0, 370.0)
        hk, ak = REF.at_target_zero(kappa, "kette", 2.0, 370.0)
        ss = REF.steady_at_target(kappa, 370.0)
        erwartet = (f"| {kappa} ({EV.de(kappa / 8, 2)}) | {EV.de(hi, 1)} | {EV.de(ai, 1)} | {EV.de(hk, 1)} | {EV.de(ak, 1)} | {EV.de(ss['gleich'][0], 1)} ({EV.de(ss['gleich'][1], 1)}) | "
                    f"{EV.de(ss['eq'], 1)} | {EV.de(ak / ai, 2)} |")
        assert erwartet == zeile and zeile in README
    assert "342,6 bis 360,3" in README
    kr = {round(z["k"], 2): z for z in REF.k_reihe()}
    claim("k = 2,25 → 2,50 (103,2 gegen 41,3)", kr[2.25]["verhaeltnis"], 0.005, nth=1)
    assert abs(kr[2.25]["arl1_kette"] - 103.2) < 0.05 and abs(kr[2.25]["arl1_iid"] - 41.3) < 0.05
    claim("k ≈ 3,97 → 1,31 (125,5 gegen 96,1)", kr[3.97]["verhaeltnis"], 0.005, nth=1)
    assert abs(kr[3.97]["arl1_kette"] - 125.5) < 0.05 and abs(kr[3.97]["arl1_iid"] - 96.1) < 0.05
    claim("k ≈ 4,47 → 1,13 (141,2 gegen 125,1)", kr[4.47]["verhaeltnis"], 0.005, nth=1)
    assert abs(kr[4.47]["arl1_kette"] - 141.2) < 0.05 and abs(kr[4.47]["arl1_iid"] - 125.1) < 0.05
    ko = {z["k"]: z for z in REF.k_kontrolle()}
    claim("(k = 4,0: 1,29; k = 4,5: 1,14)", ko[4.0]["verhaeltnis"], 0.005, nth=1)
    assert abs(ko[4.5]["verhaeltnis"] - 1.14) < 0.005


def test_holt_ein_anderes_k_den_wert_der_unabhaengigen_rechnung():
    s = EV.k_summary()
    assert s["n"] == 8 and s["v_min"] > 1.1 and s["v_max"] < 2.75 and s["v_min"] == pytest.approx(1.128, abs=0.005) and s["v_max"] == pytest.approx(2.719, abs=0.005)
    assert "Bei keinem der acht gemessenen k (2,25 bis 4,47) holt die Kette den Wert der unabhängigen Rechnung: sie ist 1,1- bis 2,7-fach langsamer" in README
    claim("(k = 2,5: 101,2) ist 2,7-mal langsamer", s["best_kette"], 0.05, nth=1)
    assert s["faktor_beste"] == pytest.approx(2.72, abs=0.005) and s["best_iid"] == pytest.approx(37.2, abs=0.05) and s["best_kette_k"] == s["best_iid_k"] == 2.5


# ----------------------------------------------------------------------------------------------------------------- Bezugsgrößen, Wartezeit (exakt / gelb)
def test_bezugsgroessen_unabhaengig():
    iw = REF.iid_wartezeit()
    claim("unabhängig Normal 50,5", iw["normal"]["arl1"], 0.05)
    claim("exakten schiefen Randverteilung 56,0", iw["exakt"]["800"]["arl1"], 0.05)
    assert "50,5 → unabhängig mit der exakten schiefen Randverteilung 56,0 → Gate **134,7**" in README


def test_wartezeit_naiver_cusum_beide_zufalls_seeds():
    a, b = REF.satz(0)["v2"], REF.satz(1)["v2"]
    claim("**42,5 ± 0,4** (Zufalls-Seed 17001)", a["arl0"], 0.05)
    assert abs(a["se"] - 0.4) < 0.05 and abs(b["se"] - 0.4) < 0.05
    claim("**42,2 ± 0,4** (17002)", b["arl0"], 0.05)
    assert a["median"] == b["median"] == 32 and "Median 32" in README
    assert "91,8 % beziehungsweise 92,2 %" in README and abs(100 * a["p100"] - 91.8) < 0.05 and abs(100 * b["p100"] - 92.2) < 0.05
    # im Band der Vormessung (42,28 ± 0,38) und der älteren V2-Messung (42,9 ± 0,7)
    for v in (a, b):
        assert abs(v["arl0"] - 42.28) < 2 * math.hypot(v["se"], 0.38) and abs(v["arl0"] - 42.9) < 2 * math.hypot(v["se"], 0.7)
    assert "42,28 ± 0,38" in README and "42,9 ± 0,7" in README


def test_wartezeit_rohkarte_blockmittel_ausduennen():
    k = REF.satz(0)["karten"]
    lab = "lam 1,6 -> 2,0"
    roh = k["Roh k=0,14"]["arl1"][lab]
    claim("**134,7**", roh[0], 0.05)
    claim("gegen Rohkarte 134,7 ± 1,4", roh[0], 0.05)
    assert abs(roh[1] - 1.4) < 0.05
    blocks = [k[f"Blockmittel b={b}"]["arl1"][lab] for b in (5, 10, 20, 40)]
    assert "**135,2 / 135,9 / 135,4 / 136,2** (je ± 1,4)" in README
    assert [round(v[0], 1) for v in blocks] == [135.2, 135.9, 135.4, 136.2] and all(abs(v[1] - 1.4) < 0.05 for v in blocks)
    # nicht aufgelöst: jeder Unterschied unter 2 Standardfehlern der Differenz, in beiden Zufalls-Seeds
    for i in (0, 1):
        r0 = REF.satz(i)["karten"]["Roh k=0,14"]["arl1"][lab]
        for b in (5, 10, 20, 40):
            v = REF.satz(i)["karten"][f"Blockmittel b={b}"]["arl1"][lab]
            assert abs(v[0] - r0[0]) < 2 * math.hypot(v[1], r0[1]), (i, b)
    ausd = [k[f"Ausduennen jeder {b}."]["arl1"][lab][0] for b in (5, 10, 20)]
    assert "141,7 / 166,7 / 202,2 (Rohkarte 134,7)" in README and [round(x, 1) for x in ausd] == [141.7, 166.7, 202.2]
    assert all(x > roh[0] + 4 for x in ausd)


def test_wartezeit_blockmittel_lag1_und_faktoren():
    bl = REF.satz(0)["ref"]["blocks"]
    assert [round(bl[b]["lag1"], 2) for b in ("5", "10", "20", "40")] == [0.36, 0.18, 0.08, 0.04] and "0,36 / 0,18 / 0,08 / 0,04" in README


def test_wartezeit_versetzte_blockgrenze():
    k = REF.satz(0)
    roh = k["karten"]["Roh k=0,14"]["arl1"]["lam 1,6 -> 2,0"][0]
    ph = [k["phase"][f"Blockmittel b={b}"]["arl1"][0] for b in (5, 10, 20, 40)]
    assert "133,0 / 131,5 / 130,4 / 129,0" in README and [round(x, 1) for x in ph] == [133.0, 131.5, 130.4, 129.0]
    assert "1,7 bis 5,6 Lkw" in README and min(roh - x for x in ph) == pytest.approx(1.7, abs=0.05) and max(roh - x for x in ph) == pytest.approx(5.6, abs=0.05)
    k2 = REF.satz(1)
    ph2 = [k2["phase"][f"Blockmittel b={b}"]["arl1"][0] for b in (5, 10, 20, 40)]
    assert "(130,3 / 129,6 / 128,8 / 127,2 gegen 134,9)" in README
    assert [round(x, 1) for x in ph2] == [130.3, 129.6, 128.8, 127.2] and round(k2["karten"]["Roh k=0,14"]["arl1"]["lam 1,6 -> 2,0"][0], 1) == 134.9
    # die versetzte Grenze ist in jeder Blocklänge und beiden Sätzen schneller als die Blockkarte auf der Grenze
    for i, p in ((0, ph), (1, ph2)):
        for b, x in zip((5, 10, 20, 40), p):
            assert x < REF.satz(i)["karten"][f"Blockmittel b={b}"]["arl1"]["lam 1,6 -> 2,0"][0] - 1.5


def test_wartezeit_karte_auf_n_und_ein_viertel_schneller():
    k = REF.satz(0)["karten"]
    n = k["Karte auf n (sim, kappa 20)"]["arl1"]["lam 1,6 -> 2,0"]
    roh = k["Roh k=0,14"]["arl1"]["lam 1,6 -> 2,0"]
    assert "101,3 ± 1,0 gegen 134,7 ± 1,4: rund ein Viertel schneller" in README
    assert round(n[0], 1) == 101.3 and round(n[1], 1) == 1.0 and 0.23 < 1 - n[0] / roh[0] < 0.27


def test_wartezeit_residuenkarte():
    a = REF.satz(0)["karten"]
    r = [a["Residuen W|n"]["arl1"][f"lam 1,6 -> {x}"][0] for x in ("1,8", "2,0", "2,4")]
    ro = [a["Roh k=0,14"]["arl1"][f"lam 1,6 -> {x}"][0] for x in ("1,8", "2,0", "2,4")]
    assert "**312,0 / 269,6 / 215,5** (Rohkarte 203,9 / 134,7 / 78,9)" in README
    assert [round(x, 1) for x in r] == [312.0, 269.6, 215.5] and [round(x, 1) for x in ro] == [203.9, 134.7, 78.9]
    faktoren = [x / y for x, y in zip(r, ro)]
    assert 1.5 <= min(faktoren) < 1.55 and 2.7 <= max(faktoren) < 2.75 and "1,5- bis 2,7-mal langsamer, aber unter ARL₀ = 370" in README
    assert all(x < 370 for x in r)                                            # nicht blind
    for i in (0, 1):                                                          # auch im zweiten Zufalls-Seed 1,5- bis 2,7-mal
        ri = [REF.satz(i)["karten"]["Residuen W|n"]["arl1"][f"lam 1,6 -> {x}"][0] / REF.satz(i)["karten"]["Roh k=0,14"]["arl1"][f"lam 1,6 -> {x}"][0] for x in ("1,8", "2,0", "2,4")]
        assert 1.45 < min(ri) and max(ri) < 2.8
    ab = a["Residuen W|n"]["arl1"]["mu 1,0 -> 0,85"]
    abr = a["Roh k=0,14"]["arl1"]["mu 1,0 -> 0,85"]
    assert "107,8 ± 1,2 gegen Rohkarte 102,2 ± 1,0" in README
    assert (round(ab[0], 1), round(ab[1], 1), round(abr[0], 1), round(abr[1], 1)) == (107.8, 1.2, 102.2, 1.0)
    assert a["Karte auf n (sim, kappa 20)"]["arl1"]["mu 1,0 -> 0,85"][0] == pytest.approx(133.6, abs=0.05)
    assert "Karte auf n 133,6 (135,2)" in README


def test_wartezeit_zweiter_zufalls_seed_alle_44_zellen_im_band():
    a, b = REF.satz(0)["karten"], REF.satz(1)["karten"]
    z = [abs(a[n]["arl1"][lab][0] - b[n]["arl1"][lab][0]) / math.hypot(a[n]["arl1"][lab][1], b[n]["arl1"][lab][1]) for n in a for lab in a[n]["arl1"]]
    assert len(z) == 44 and max(z) < 4 and "alle 44 Zellen beider Sätze liegen im 4-Standardfehler-Band der Differenz" in README


def test_wartezeit_autokorrelation_lag_1_bis_5():
    acf = REF.satz(0)["ref"]["acf"]
    assert "(0,78 / 0,59 / 0,41 / 0,26 / 0,18)" in README
    assert [round(acf[k], 2) for k in "12345"] == [0.78, 0.59, 0.41, 0.26, 0.18]
    assert "Lag-1-Autokorrelation der Wartezeit 0,78" in README


# ----------------------------------------------------------------------------------------------------------------- Modellkarte (exakt)
def test_modellkarte_bei_gleicher_arl0():
    q = {qs: EV.exp4(2.0, 2.0, 1.6, qs) for qs in (20, 40, 80)}
    claim("102,3 gegen 101,2 der Rohkarte (Gitter 1/40)", q[40]["mod"]["arl1"][2.0], 0.05)
    assert q[40]["raw"]["arl1"][2.0] == pytest.approx(101.2, abs=0.05)
    assert "107,4 bei Gitter 1/20, 102,2 bei 1/80" in README
    assert round(q[20]["mod"]["arl1"][2.0], 1) == 107.4 and round(q[80]["mod"]["arl1"][2.0], 1) == 102.2


def test_modellkarte_information_und_naeherung():
    r = EV.exp4(2.0, 2.0, 1.6, 40)
    assert "0,01058 (Übergänge) gegen 0,06362" in README
    assert r["i_trans"] == pytest.approx(0.01058, abs=5e-6) and r["i_marg"] == pytest.approx(0.06362, abs=5e-6)
    assert "ln(370)/0,01058 = 559, das 5,2- bis 5,5-Fache" in README
    assert r["lorden"] == pytest.approx(559, abs=0.5)
    fakt = [EV.exp4(2.0, 2.0, 1.6, qs) for qs in (20, 40, 80)]
    assert [round(x["lorden"] / x["mod"]["arl1"][2.0], 1) for x in fakt] == [5.2, 5.5, 5.5]
    assert "des exakten Werts 102,3" in README


def test_modellkarte_groesse_der_aenderung_und_der_gitterartefakt():
    r = EV.exp4(2.0, 2.0, 1.6, 40)
    soll = {1.7: (240.9, 239.4, 248.1), 1.8: (169.9, 169.1, 173.0), 2.0: (101.2, 102.3, 102.3), 2.4: (55.3, 57.8, 55.7), 3.0: (36.0, 38.3, 34.2), 4.0: (26.6, 28.1, 22.4)}
    for z in r["rows"]:
        roh, mod, eigen = soll[z["lam"]]
        zeile = f"| {EV.de(z['lam'], 1)} | {EV.de(roh, 1)} | {EV.de(mod, 1)} | {EV.de(eigen, 1)} |"
        assert zeile in README, zeile
        assert (round(z["roh"], 1), round(z["modell"], 1), round(z["eigen"], 1)) == (roh, mod, eigen)
    assert "λ₁ = 4,0: 22,4 gegen 26,6" in README
    # das Wort „erst bei großen Änderungen“: die für das wahre λ₁ entworfene Karte ist nur ab λ₁ ≥ 3,0 besser als die Rohkarte
    assert [z["eigen"] < z["roh"] for z in r["rows"]] == [False, False, False, False, True, True]
    assert "fein gerechnet sind es 248,1 (Rohkarte 240,9)" in README


def test_falsches_nullmodell():
    r = EV.exp4(2.0, 2.0, 1.6, 40)
    null = {z["lam0"]: z for z in r["null"]}
    assert (r["raw"]["h_int"], r["mod"]["h_int"]) == (309, 74) and "Rohkarte H = 309, Modellkarte H = 74" in README
    assert "| Faktor Rohkarte | 3,23 | 1,70 | 1,29 | 1,00 | 0,80 | 0,65 | 0,45 |" in README
    assert [round(null[x]["faktor_roh"], 2) for x in (1.4, 1.5, 1.55, 1.6, 1.65, 1.7, 1.8)] == [3.23, 1.70, 1.29, 1.00, 0.80, 0.65, 0.45]
    assert round(null[1.7]["faktor_modell"], 2) == 0.64 and round(null[1.5]["faktor_modell"], 2) == 1.73
    assert "das 0,65-Fache (Rohkarte) bzw. 0,64-Fache (Modellkarte, Gitter 1/40; bei Gitter 1/20: 0,67); λ₀′ = 1,5: das 1,70- bzw. 1,73-Fache (Gitter 1/40)" in README
    assert 1.7 / 1.6 - 1 == pytest.approx(0.0625) and "um 6 % (1,7 gegen 1,6)" in README and 2.0 / 1.6 - 1 == pytest.approx(0.25)


def test_sprung_in_standardabweichungen_und_nullmittel():
    assert (G1.mean_n - G0.mean_n) / G0.sd_n == pytest.approx(0.362, abs=5e-4)
    assert (G1.mean_w - G0.mean_w) / G0.sd_w == pytest.approx(0.279, abs=5e-4)
    assert G0.mean_n == pytest.approx(2.16, abs=0.005) and "Nullmittel der gesehenen Zahl (2,16)" in README
    assert 8 * G0.mean_n < 18 + 1e-9 and 8 * G0.mean_n > 16        # κ = 16 (k = 2,0) liegt unter dem Nullmittel, κ = 18 darüber: der Regler beginnt bei 18
    assert C.KAPPA_OPTIONS[0] == 18 and 18 / 8 > G0.mean_n > 16 / 8


# ----------------------------------------------------------------------------------------------------------------- Texte
def test_annahmen_tabelle_steht_wortgleich_in_app_und_readme():
    for zeile in TX.ASSUMPTIONS_MARKDOWN.strip().splitlines():
        assert zeile in README, zeile


def test_vehikel_satz_steht_wortgleich_in_app_und_readme():
    assert TX.VEHIKEL in README.replace("**", "").replace("\n", " ")


def test_zahlen_der_texte_der_app():
    """Zahlen, die lwg_texts.py nennt und die nicht aus einem Verdikt kommen."""
    assert "bei H = 20 liegt ARL₀ der Kette mit 13,4 Lkw über den 7,7" in TX.EXP1_NEGATIV
    r = EV.exp1(20, 20)
    assert round(r["arl0_kette"], 1) == 13.4 and round(r["arl0_iid"], 1) == 7.7
    assert "Vormessung 42,28 ± 0,38" in TX.EXP1_NEGATIV
    assert "Bei keinem der" in EV.verdict_k_reihe(EV.k_summary())[1]
    assert "Bei λ₁ = 1,7 ist die für das wahre λ₁ entworfene Karte nicht besser als die Rohkarte" in TX.EXP4_NEGATIV and "270,7" in TX.EXP4_NEGATIV
    r4 = EV.exp4(1.7, 1.7, 1.6, 40)
    assert r4["own"]["arl1"][1.7] > r4["raw"]["arl1"][1.7]


def test_steady_state_ist_in_jeder_zeile_schneller_als_zero_state():
    for kappa in C.KAPPA_OPTIONS:
        z = REF.at_target_zero(kappa, "kette", 2.0, 370.0)[1]
        ss = REF.steady_at_target(kappa, 370.0)
        assert ss["gleich"][0] < z and ss["eq"] < z
    assert abs(REF.steady_at_target(18, 370.0)["gleich"][0] - 83.2) < 0.05            # κ = 18 (nicht in der README-Tabelle)
