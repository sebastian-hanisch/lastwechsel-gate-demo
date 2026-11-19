"""Erzeugt lwg_kennlinien.json (gefrorene Kennlinien der exakten Ketten) und prüft sie.

    python tools/lwg_make_kennlinien.py --voll    rechnet alle Kennlinien neu (exakt, Windows-Python 3.14 rund 1 bis 3 Minuten) und schreibt lwg_kennlinien.json
    python tools/lwg_make_kennlinien.py --check   rechnet 12 Stützstellen neu und vergleicht sie mit der Datei (relative Toleranz 1e-7; Rückgabewert 1 bei Abweichung)

Inhalt: je Referenzwert κ (k = κ/8) die Kurven ARL₀(H) und ARL₁(H) für sechs Lastwechsel λ₁ (Zero-state), für unabhängige Beobachtungen ('iid', Randverteilung der
gesehenen Zahl) und für die tatsächliche Kette; bei der Kette zusätzlich Steady-state-Spalten (quasistationär) für 1,6 → 2,0 im Bereich ARL₀ 40 bis 4 000;
die Reihe über acht k (Tabelle der Schwelle und Verzögerung bei ARL₀ = 370) mit zwei Kontrollzeilen auf anderem Raster; die iid-Bezugsgrößen der Wartezeit-Karte.
Die Schwellen-Raster haben die Schrittweite 4 (alle Scores sind Vielfache von 2, ARL₀(H) ist eine Treppe); ARL₀ über 2·10⁵ wird nicht mehr gerechnet."""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import lwg_cards as CA  # noqa: E402
import lwg_chain as CH  # noqa: E402
import lwg_constants as C  # noqa: E402
import lwg_iid as IID  # noqa: E402

PATH = ROOT / "lwg_kennlinien.json"
SS_WINDOW = (40.0, 4000.0)


def _r(x, k=9):
    return None if x is None else float(f"{x:.{k}g}")


def curves(R0, Rl, pi0, S, Hs, lams, steady):
    """Kurven für eine Karte: arl0(H) und je λ₁ arl1(H); bei steady die Steady-state-Spalten für 2,0."""
    out = {"H": [], "arl0": [], "arl1": {str(lam): [] for lam in lams}, "arl0_ss": [], "arl1_ss": []}
    for H in Hs:
        e0 = CA.exact_card(R0, R0, S, H, pi0, steady=False)
        out["H"].append(H)
        out["arl0"].append(_r(e0["arl0"]))
        for lam in lams:
            out["arl1"][str(lam)].append(_r(CA.exact_card(R0, Rl[lam], S, H, pi0)["arl1"]))
        if steady and SS_WINDOW[0] <= e0["arl0"] <= SS_WINDOW[1]:
            e = CA.exact_card(R0, Rl[C.LAM1], S, H, pi0, steady=True)
            out["arl0_ss"].append(_r(e["arl0_ss"]))
            out["arl1_ss"].append(_r(e["arl1_ss"]))
        else:
            out["arl0_ss"].append(None)
            out["arl1_ss"].append(None)
        if e0["arl0"] > C.ARL_CAP:
            break
    return out


def k_row(q, kappa, hstep, g0, g1):
    S = CA.raw_score(q, kappa)
    Hs = list(range(hstep, (1401 if q == 32 else C.H_MAX_FROZEN + 1), hstep))
    Riid0, Riid1 = np.tile(g0.pi, (6, 1)), np.tile(g1.pi, (6, 1))
    si = CA.scan(Riid0, Riid1, S, Hs, g0.pi)
    sd = CA.scan(g0.R, g1.R, S, Hs, g0.pi)
    ti = CA.at_target(si["H"], si["arl0"], si["arl1"], C.ARL0_NENN)
    td = CA.at_target(sd["H"], sd["arl0"], sd["arl1"], C.ARL0_NENN)
    return {"q": q, "kappa": kappa, "k": kappa / q, "h_iid": _r(ti[0], 6), "arl1_iid": _r(ti[1], 6), "h_kette": _r(td[0], 6), "arl1_kette": _r(td[1], 6),
            "verhaeltnis": _r(td[1] / ti[1], 6)}


def build(log=print) -> dict:
    t0 = time.time()
    g0, g1 = CH.Gate(C.LAM0), CH.Gate(C.LAM1)
    gl = {lam: CH.Gate(lam) for lam in C.LAM1_OPTIONS}
    Rl = {lam: gl[lam].R for lam in C.LAM1_OPTIONS}
    Riid0 = np.tile(g0.pi, (6, 1))
    Riid_l = {lam: np.tile(gl[lam].pi, (6, 1)) for lam in C.LAM1_OPTIONS}
    data = {"kappa": {}}
    Hs = list(range(C.H_STEP_FROZEN, C.H_MAX_FROZEN + 1, C.H_STEP_FROZEN))
    for kappa in C.KAPPA_OPTIONS:
        S = CA.raw_score(C.Q, kappa)
        # iid: die Randverteilung der neuen Rate bestimmt die Verzögerung; die Karte ist dieselbe
        iid = {"H": [], "arl0": [], "arl1": {str(lam): [] for lam in C.LAM1_OPTIONS}}
        for H in Hs:
            a0 = CA.exact_card(Riid0, Riid0, S, H, g0.pi)["arl0"]
            iid["H"].append(H)
            iid["arl0"].append(_r(a0))
            for lam in C.LAM1_OPTIONS:
                iid["arl1"][str(lam)].append(_r(CA.exact_card(Riid0, Riid_l[lam], S, H, g0.pi)["arl1"]))
            if a0 > C.ARL_CAP:
                break
        kette = curves(g0.R, Rl, g0.pi, S, Hs, C.LAM1_OPTIONS, steady=True)
        data["kappa"][str(kappa)] = {"iid": iid, "kette": kette}
        log(f"   [{time.time() - t0:6.1f} s] kappa = {kappa}")
    rows = [k_row(C.Q, kap, 4, g0, g1) for kap in (18, 20, 22, 24, 26, 28)] + [k_row(32, 127, 8, g0, g1), k_row(32, 143, 8, g0, g1)]
    controls = [k_row(C.Q, 32, 4, g0, g1), k_row(C.Q, 36, 4, g0, g1)]
    data["k_reihe"] = {"zeilen": rows, "kontrolle_raster_q8": controls}
    log(f"   [{time.time() - t0:6.1f} s] k-Reihe")
    ref = IID.iid_wait_reference(g0, g1)
    data["iid_wartezeit"] = {"shift": _r(ref["shift"]), "k": _r(ref["k"]), "normal": {k: _r(v) for k, v in ref["normal"].items()},
                             "exakt": {m: {k: _r(v) for k, v in rec.items()} for m, rec in ref["exakt"].items()}}
    data["meta"] = {"quelle": "tools/lwg_make_kennlinien.py --voll", "stand": time.strftime("%Y-%m-%d"), "python": platform.python_version(), "numpy": np.__version__,
                    "plattform": platform.platform(), "raster": f"H in Schritten von {C.H_STEP_FROZEN}, bis ARL0 > {C.ARL_CAP:.0e} oder H = {C.H_MAX_FROZEN}",
                    "steady": f"quasistationär, nur für ARL0 zwischen {SS_WINDOW[0]:.0f} und {SS_WINDOW[1]:.0f}", "sekunden": round(time.time() - t0)}
    return data


def _close(a, b, rel=1e-7):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def check() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    g0 = CH.Gate(C.LAM0)
    gl = {lam: CH.Gate(lam) for lam in C.LAM1_OPTIONS}
    bad = 0
    n = 0
    for kappa, Hs in (("20", (4, 100, 308, 312)), ("24", (60, 180)), ("28", (40, 100))):
        S = CA.raw_score(C.Q, int(kappa))
        rec = data["kappa"][kappa]
        for H in Hs:
            i = rec["kette"]["H"].index(H)
            e0 = CA.exact_card(g0.R, g0.R, S, H, g0.pi)
            e1 = CA.exact_card(g0.R, gl[2.4].R, S, H, g0.pi)
            for a, b in ((e0["arl0"], rec["kette"]["arl0"][i]), (e1["arl1"], rec["kette"]["arl1"]["2.4"][i])):
                n += 1
                if not _close(a, b):
                    bad += 1
                    print(f"ABWEICHUNG kappa {kappa} H {H}: neu {a} gespeichert {b}")
    print(f"{n} Werte verglichen, {bad} Abweichungen")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voll", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.voll:
        d = build()
        PATH.write_text(json.dumps(d, ensure_ascii=False, indent=0, separators=(",", ":")), encoding="utf-8")
        print(f"geschrieben: {PATH} ({PATH.stat().st_size} Bytes), {d['meta']['sekunden']} s")
    elif a.check:
        sys.exit(check())
    else:
        ap.print_help()
