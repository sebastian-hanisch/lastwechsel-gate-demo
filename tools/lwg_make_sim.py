"""Erzeugt lwg_sim.json (gefrorene Messreihen der Karten auf der Wartezeit) und prüft sie.

    python tools/lwg_make_sim.py --voll     zwei vollständige Sätze (zwei Zufalls-Seeds), schreibt lwg_sim.json (rund 10 bis 20 Minuten, ein Thread)
    python tools/lwg_make_sim.py --check    kleine Neu-Messung (andere Zufalls-Seeds, 1 500 Läufe) und Vergleich im Band mit der gespeicherten Datei (Rückgabewert 1 bei Abweichung)

Die Messreihe: Zufall nur aus SplitMix64 (`lwg_normal.Stream`). Je Satz: ein Referenzlauf (Autokorrelation, Blockmittel; 1 000 Läufe × 8 000 Lkw), ein Lauf ohne Änderung
(10 000 Läufe × 12 000 Lkw: Schwelle für ARL₀ = 370 je Karte und die ARL₀ des naiven zweiseitigen CUSUM) und je ein Lauf nach der Änderung (8 000 Läufe × 6 000 Lkw,
400 Lkw Einschwingen, Karten starten bei ν mit 0). Die Karten: Rohkarte (k = Sprung/2 und k = 0,5), Ausdünnen, Blockmittel, Karte auf n, Residuenkarte. Zusätzlich je Satz
ein Lauf, in dem die Änderung NICHT auf einer Blockgrenze liegt (Phase b/2; die Schwelle bleibt die des Laufs ohne Änderung)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import lwg_chain as CH  # noqa: E402
import lwg_constants as C  # noqa: E402
import lwg_normal as N  # noqa: E402
import lwg_sim as S  # noqa: E402

PATH = ROOT / "lwg_sim.json"
SCENARIOS = (("lam 1,6 -> 1,8", 1.8, 1.0), ("lam 1,6 -> 2,0", 2.0, 1.0), ("lam 1,6 -> 2,4", 2.4, 1.0), ("mu 1,0 -> 0,85", 1.6, 0.85))
HG_Z = sorted(set(list(np.round(np.arange(0.5, 60.0, 0.25), 4)) + [4.0965, 4.77]))
HG_B = np.round(np.arange(0.25, 30.0, 0.125), 4)
HG_N = np.arange(100, 480, 4)
BLOCKS = (5, 10, 20, 40)


def _sig(x, k=6):
    return float(f"{x:.{k}g}")


def make_cards(R, g0, g1, sdb, resid_stream):
    """Alle Karten eines Laufs (gleiche Bauart vor und nach der Änderung)."""
    shift = (g1.mean_w - g0.mean_w) / g0.sd_w
    k_raw = shift / 2
    cards = [S.RawCard(R, HG_Z, k_raw, g0.mean_w, g0.sd_w, "Roh k=0,14"), S.RawCard(R, HG_Z, 0.5, g0.mean_w, g0.sd_w, "Roh k=0,5")]
    for b in (5, 10, 20):
        cards.append(S.ThinCard(R, HG_Z, k_raw, g0.mean_w, g0.sd_w, b, f"Ausduennen jeder {b}."))
    for b in BLOCKS:
        cards.append(S.BlockCard(R, HG_B, shift / sdb[b] / 2, g0.mean_w, sdb[b], b, f"Blockmittel b={b}"))
    cards.append(S.NCard(R, HG_N, C.Q, 20, "Karte auf n (sim, kappa 20)"))
    cards.append(S.ResidCard(R, HG_Z, 0.25, resid_stream, "Residuen W|n"))
    return cards


def measure(seed, R0=C.SIM_R_NULL, R1=C.SIM_R_CHANGE, T0=C.SIM_T_NULL, T1=C.SIM_T_CHANGE, Rref=1000, Tref=8000, log=print) -> dict:
    t_start = time.time()
    g0, g1 = CH.Gate(C.LAM0), CH.Gate(C.LAM1)
    # --- Referenzlauf
    W = S.collect_waits(seed + 1, Rref, C.SIM_BURN, Tref)
    acf = S.acf_lags(W)
    bst = S.block_stats(W, g0.sd_w)
    sdb = {b: bst[b]["sd"] for b in BLOCKS}
    ref = {"R": Rref, "T": Tref, "mean_w": _sig(float(W.mean())), "sd_w": _sig(float(W.std())),
           "acf": {str(k): _sig(v, 4) for k, v in acf.items()}, "blocks": {str(b): {k: _sig(v, 4) for k, v in bst[b].items()} for b in BLOCKS}}
    del W
    log(f"   [{time.time() - t_start:6.1f} s] Referenzlauf")
    # --- ohne Änderung
    cards0 = make_cards(R0, g0, g1, sdb, N.Stream(seed + 10))
    naiv = S.TwoSidedFixed(R0, g0.mean_w, g0.sd_w)
    S.run_pass(seed + 2, R0, C.SIM_BURN, T0, C.LAM0, C.LAM0, C.MU, C.MU, cards0 + [naiv])
    rl = np.where(naiv.rl == 0, T0, naiv.rl).astype(float)
    v2 = {"R": R0, "arl0": _sig(rl.mean()), "se": _sig(rl.std(ddof=1) / np.sqrt(R0), 3), "median": float(np.median(rl)), "p100": _sig(float(np.mean(rl <= 100)), 5)}
    pre = {c.name: c for c in cards0}
    log(f"   [{time.time() - t_start:6.1f} s] Lauf ohne Änderung; naiver CUSUM: ARL0 {v2['arl0']} +- {v2['se']}")
    # --- nach der Änderung
    scen = {}
    for i, (label, lam1, mu1) in enumerate(SCENARIOS):
        cs = make_cards(R1, g0, g1, sdb, N.Stream(seed + 100 + i))
        S.run_pass(seed + 20 + i, R1, C.SIM_BURN, T1, C.LAM0, lam1, C.MU, mu1, cs)
        scen[label] = {c.name: c for c in cs}
        log(f"   [{time.time() - t_start:6.1f} s] Lauf nach der Änderung {label}")
    karten = {}
    for name, pc in pre.items():
        r0 = card_h(pc, scen[SCENARIOS[1][0]][name], T0, T1)
        if r0 is None:
            karten[name] = None
            continue
        rec = {"h": _sig(r0["h"]), "arl1": {}}
        for label, _, _ in SCENARIOS:
            r = S.card_at_target(pc, scen[label][name], T0, T1)
            rec["arl1"][label] = None if r is None or not np.isfinite(r["arl1"]) else [_sig(r["arl1"]), _sig(r["se1"], 3)]
        karten[name] = rec
    # --- Phase der Änderung gegen den Block (nur 1,6 -> 2,0)
    shift = (g1.mean_w - g0.mean_w) / g0.sd_w
    phase = {}
    csp = [S.BlockCard(R1, HG_B, shift / sdb[b] / 2, g0.mean_w, sdb[b], b, f"Blockmittel b={b}", phase=b // 2) for b in BLOCKS]
    S.run_pass(seed + 30, R1, C.SIM_BURN, T1, C.LAM0, C.LAM1, C.MU, C.MU, csp)
    for c in csp:
        r = S.card_at_target(pre[c.name], c, T0, T1)
        phase[c.name] = {"phase": c.phase, "arl1": [_sig(r["arl1"]), _sig(r["se1"], 3)]}
    log(f"   [{time.time() - t_start:6.1f} s] Lauf mit versetzter Blockgrenze")
    return {"seed": seed, "R0": R0, "R1": R1, "T0": T0, "T1": T1, "ref": ref, "v2": v2, "karten": karten, "phase": phase, "sekunden": round(time.time() - t_start)}


def card_h(pc, post, T0, T1):
    return S.card_at_target(pc, post, T0, T1)


def voll() -> int:
    import platform
    import scipy
    saetze = [measure(s) for s in C.SIM_SEEDS]
    data = {"meta": {"quelle": "tools/lwg_make_sim.py --voll", "stand": time.strftime("%Y-%m-%d"), "python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "plattform": platform.platform(),
                     "zufall": "SplitMix64 (lwg_normal.Stream), Zufalls-Seeds " + ", ".join(map(str, C.SIM_SEEDS))},
            "saetze": saetze}
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"geschrieben: {PATH} ({PATH.stat().st_size} Bytes)")
    return 0


def check() -> int:
    """Kleine Neu-Messung mit anderem Zufalls-Seed; ARL₁ je Karte im Band (|z| <= 4 mit beiden Standardfehlern, dazu 15 % Zuschlag wegen der kleineren Zahl Läufe)."""
    data = json.loads(PATH.read_text(encoding="utf-8"))
    ref = data["saetze"][0]
    new = measure(91000, R0=2000, R1=1500, T0=4000, T1=3000, Rref=200, Tref=3000, log=lambda *_: None)
    bad = 0
    label = "lam 1,6 -> 2,0"
    for name in ("Roh k=0,14", "Blockmittel b=20", "Karte auf n (sim, kappa 20)", "Residuen W|n"):
        a, b = ref["karten"][name]["arl1"][label], new["karten"][name]["arl1"][label]
        if a is None or b is None:
            print(f"FEHLT {name}")
            bad += 1
            continue
        z = abs(a[0] - b[0]) / np.hypot(a[1], 3 * b[1])
        status = "ok" if z <= 4 else "ABWEICHUNG"
        bad += status != "ok"
        print(f"{status:11s} {name:30s} gespeichert {a[0]:7.1f}  neu {b[0]:7.1f} +- {b[1]:.1f}  z = {z:.2f}")
    print(f"naiver CUSUM ARL0: gespeichert {ref['v2']['arl0']}, neu {new['v2']['arl0']} +- {new['v2']['se']}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voll", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--klein", action="store_true", help="nur Laufzeit messen (kleine Fassung, schreibt nichts)")
    a = ap.parse_args()
    if a.klein:
        t = time.time()
        r = measure(5, R0=1000, R1=800, T0=3000, T1=2000, Rref=100, Tref=2000)
        print(json.dumps(r["karten"], ensure_ascii=False)[:1500], r["v2"], f"{time.time() - t:.0f} s")
    elif a.voll:
        sys.exit(voll())
    elif a.check:
        sys.exit(check())
    else:
        ap.print_help()
