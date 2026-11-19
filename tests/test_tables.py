"""Gefrorene Daten: Aufbau, Neurechnung im Band (`--check` der beiden Skripte), Abgleich mit den exakten Rechnungen und der Stabilität über zwei Zufalls-Seeds.

Exakte Zahlen (Kennlinien) werden nicht auf Gleichheit, sondern mit relativer Toleranz verglichen (die Letztstelle des Gleichungslösers hängt von der Plattform ab);
Simulationen (Messreihen) im Band aus Standardfehlern."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import lwg_chain as CH
import lwg_constants as C
import lwg_evaluation as EV
import lwg_reference as REF

ROOT = Path(__file__).resolve().parent.parent
G0 = CH.Gate(1.6)


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------- Kennlinien
def test_kennlinien_aufbau_und_kopf():
    k = REF.kennlinien()
    assert set(k) == {"kappa", "k_reihe", "iid_wartezeit", "meta"}
    assert set(k["kappa"]) == {str(x) for x in C.KAPPA_OPTIONS}
    for kappa in C.KAPPA_OPTIONS:
        for var in ("iid", "kette"):
            cv = REF.curve(kappa, var)
            assert cv["H"][0] == C.H_STEP_FROZEN and np.all(np.diff(cv["H"]) == C.H_STEP_FROZEN)
            assert len(cv["arl0"]) == len(cv["H"]) and set(cv["arl1"]) == {str(x) for x in C.LAM1_OPTIONS}
            assert all(len(v) == len(cv["H"]) for v in cv["arl1"].values())
            assert all(b >= a for a, b in zip(cv["arl0"], cv["arl0"][1:]))
    m = REF.meta_kenn()
    assert "lwg_make_kennlinien.py" in m["quelle"] and m["stand"] and m["python"]


def test_kennlinien_check_des_skripts_ist_gruen():
    assert _tool("lwg_make_kennlinien").check() == 0


@pytest.mark.parametrize("kappa, h_k, a1_i, a1_k, ratio", [(20, 309.1, 37.21, 101.16, 2.72), (22, 231.4, 41.72, 104.15, 2.50), (24, 177.6, 49.48, 108.39, 2.19),
                                                           (26, 135.6, 57.97, 112.67, 1.94), (28, 103.0, 64.23, 116.87, 1.82)])
def test_tabelle_der_fuenf_kappa_bei_370_gleich_den_exakten_werten(kappa, h_k, a1_i, a1_k, ratio):
    hi, ai = REF.at_target_zero(kappa, "iid", 2.0, 370.0)
    hk, ak = REF.at_target_zero(kappa, "kette", 2.0, 370.0)
    assert hk == pytest.approx(h_k, abs=0.05) and ai == pytest.approx(a1_i, abs=0.005) and ak == pytest.approx(a1_k, abs=0.005)
    assert ak / ai == pytest.approx(ratio, abs=0.005)


@pytest.mark.parametrize("kappa, gleich, arl0_ss, eq", [(20, 89.18, 342.6, 92.95), (22, 95.76, 351.0, 98.72), (24, 102.01, 355.6, 104.54), (26, 107.45, 358.3, 109.76), (28, 112.52, 360.3, 114.53)])
def test_steady_state_spalten_gleich_den_exakten_werten(kappa, gleich, arl0_ss, eq):
    ss = REF.steady_at_target(kappa, 370.0)
    assert ss["gleich"][0] == pytest.approx(gleich, abs=0.01) and ss["gleich"][1] == pytest.approx(arl0_ss, abs=0.05)
    assert ss["eq"] == pytest.approx(eq, abs=0.01)


def test_k_reihe_der_acht_k_und_kontrollzeilen():
    z = REF.k_reihe()
    assert [r["k"] for r in z] == [2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.96875, 4.46875]
    assert [r["verhaeltnis"] for r in z] == pytest.approx([2.50, 2.72, 2.50, 2.19, 1.94, 1.82, 1.31, 1.13], abs=0.006)
    assert z[6]["arl1_kette"] == pytest.approx(125.49, abs=0.01) and z[7]["arl1_kette"] == pytest.approx(141.18, abs=0.01)
    kk = REF.k_kontrolle()
    assert [r["k"] for r in kk] == [4.0, 4.5] and [r["verhaeltnis"] for r in kk] == pytest.approx([1.29, 1.14], abs=0.006)


def test_kennlinien_stimmen_mit_der_live_rechnung_ueberein():
    """Die gefrorene Kurve bei einer ganzzahligen Schwelle gleich der Live-Rechnung der App (exp1)."""
    for kappa, H in ((20, 308), (22, 200), (26, 100)):
        r = EV.exp1(H, kappa, tmax=10)
        cv = REF.curve(kappa, "kette")
        assert r["arl0_kette"] == pytest.approx(cv["arl0"][cv["H"].index(H)], rel=1e-7)
        cv = REF.curve(kappa, "iid")
        if H in cv["H"] and r["arl0_iid"] is not None:
            assert r["arl0_iid"] == pytest.approx(cv["arl0"][cv["H"].index(H)], rel=1e-7)


def test_iid_wartezeit_bezugsgroessen():
    iw = REF.iid_wartezeit()
    assert iw["shift"] == pytest.approx(0.27858, abs=5e-6) and iw["k"] == pytest.approx(iw["shift"] / 2, rel=1e-9)
    assert iw["normal"]["h"] == pytest.approx(9.2627, abs=5e-5) and iw["normal"]["arl0"] == pytest.approx(370.0, abs=1e-6) and iw["normal"]["arl1"] == pytest.approx(50.52, abs=0.005)
    assert [iw["exakt"][m]["arl1"] for m in ("200", "400", "800")] == pytest.approx([56.02, 56.01, 56.01], abs=0.006)     # Gitter 200 / 400 / 800 gleich auf zwei Stellen
    assert iw["exakt"]["800"]["h"] == pytest.approx(10.250, abs=5e-4)


def test_iid_wartezeit_neu_gerechnet_im_band():
    import lwg_iid as IID
    ref = IID.iid_wait_reference(G0, CH.Gate(2.0), ms=(200,))
    iw = REF.iid_wartezeit()
    assert ref["normal"]["arl1"] == pytest.approx(iw["normal"]["arl1"], rel=1e-8)
    assert ref["exakt"]["200"]["arl1"] == pytest.approx(iw["exakt"]["200"]["arl1"], rel=1e-7)


# ----------------------------------------------------------------------------------------------------------------- Messreihen
def test_messreihen_aufbau_und_kopf():
    s = REF.sim()
    assert len(s["saetze"]) == 2 and [x["seed"] for x in s["saetze"]] == list(C.SIM_SEEDS)
    assert "lwg_make_sim.py" in s["meta"]["quelle"] and "SplitMix64" in s["meta"]["zufall"]
    for st in s["saetze"]:
        assert (st["R0"], st["R1"], st["T0"], st["T1"]) == (C.SIM_R_NULL, C.SIM_R_CHANGE, C.SIM_T_NULL, C.SIM_T_CHANGE)
        assert set(st["karten"]) == {"Roh k=0,14", "Roh k=0,5", "Ausduennen jeder 5.", "Ausduennen jeder 10.", "Ausduennen jeder 20.", "Blockmittel b=5", "Blockmittel b=10",
                                     "Blockmittel b=20", "Blockmittel b=40", "Karte auf n (sim, kappa 20)", "Residuen W|n"}
        for rec in st["karten"].values():
            assert set(rec["arl1"]) == set(C.AENDERUNG_KEY.values()) and all(v is not None and v[0] > 0 and v[1] > 0 for v in rec["arl1"].values())
        assert set(st["phase"]) == {"Blockmittel b=5", "Blockmittel b=10", "Blockmittel b=20", "Blockmittel b=40"}


def test_messreihen_check_des_skripts_ist_gruen():
    assert _tool("lwg_make_sim").check() == 0


def test_beide_zufalls_seeds_liegen_im_band_der_standardfehler():
    """Je Karte und Szenario: |Satz 1 − Satz 2| ≤ 4 Standardfehler der Differenz; und höchstens 4 der 44 Zellen über 2 (bei Normalverteilung wären es im Mittel 2)."""
    a, b = REF.satz(0), REF.satz(1)
    z = []
    for nm, rec in a["karten"].items():
        for lab, v in rec["arl1"].items():
            w = b["karten"][nm]["arl1"][lab]
            z.append(abs(v[0] - w[0]) / math.hypot(v[1], w[1]))
    assert len(z) == 44 and max(z) < 4 and sum(1 for x in z if x > 2) <= 4, sorted(z)[-5:]
    # naiver CUSUM: beide Sätze innerhalb von 4 Standardfehlern der Differenz
    assert abs(a["v2"]["arl0"] - b["v2"]["arl0"]) < 4 * math.hypot(a["v2"]["se"], b["v2"]["se"])


def test_karte_auf_n_simuliert_gleich_der_exakten_kette_bei_370():
    """Die simulierte Karte auf n (κ = 20) bei ARL₀ = 370 gegen die exakte Kette (interpoliert auf 370), alle drei Ratenänderungen, beide Sätze."""
    raw = EV.raw_card_370()["arl1"]
    for i in (0, 1):
        for lab, lam in (("lam 1,6 -> 1,8", 1.8), ("lam 1,6 -> 2,0", 2.0), ("lam 1,6 -> 2,4", 2.4)):
            v = EV.sim_value("Karte auf n (sim, kappa 20)", lab, i)
            assert abs(v[0] - raw[lam]) < 4 * v[1] + 0.3, (i, lab, v, raw[lam])      # +0,3: Start nach 400 Lkw statt stationär, Raster der Schwelle


def test_referenzlauf_autokorrelation_und_blockfaktoren():
    for st in REF.sim()["saetze"]:
        acf = st["ref"]["acf"]
        for lag, soll in zip(("1", "2", "3", "4", "5"), (0.781, 0.586, 0.412, 0.255, 0.178)):
            assert abs(acf[lag] - soll) < 0.02, (lag, acf[lag])
        assert abs(acf["10"] - 0.032) < 0.02 and abs(acf["20"]) < 0.02
        bl = st["ref"]["blocks"]
        assert [bl[b]["faktor"] for b in ("5", "10", "20", "40")] == pytest.approx([1.84, 2.14, 2.32, 2.41], abs=0.03)
        assert [bl[b]["lag1"] for b in ("5", "10", "20", "40")] == pytest.approx([0.357, 0.178, 0.081, 0.037], abs=0.015)
        assert st["ref"]["mean_w"] == pytest.approx(G0.mean_w, abs=0.004) and st["ref"]["sd_w"] == pytest.approx(G0.sd_w, abs=0.004)


def test_gefrorene_dateien_sind_gueltiges_json_mit_deutschen_namen():
    for p in (REF.PATH_KENN, REF.PATH_SIM):
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["meta"]["stand"] >= "2026-11-19" or d["meta"]["stand"].startswith("2026-11")
