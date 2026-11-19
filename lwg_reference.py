"""Gefrorene Daten: `lwg_kennlinien.json` (exakte Ketten, tools/lwg_make_kennlinien.py) und `lwg_sim.json` (Messreihen der Karten auf der Wartezeit, tools/lwg_make_sim.py).
Jede Datei nennt Quelle und Stand im Kopf (`meta`); die Messreihen nennen je Satz den Zufalls-Seed. Hier nur Lesen und Interpolieren."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import lwg_cards as CA

ROOT = Path(__file__).resolve().parent
PATH_KENN = ROOT / "lwg_kennlinien.json"
PATH_SIM = ROOT / "lwg_sim.json"


@lru_cache(maxsize=1)
def kennlinien() -> dict:
    return json.loads(PATH_KENN.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def sim() -> dict:
    return json.loads(PATH_SIM.read_text(encoding="utf-8"))


def meta_kenn() -> dict:
    return kennlinien()["meta"]


def meta_sim() -> dict:
    return sim()["meta"]


def satz(i: int = 0) -> dict:
    """Der i-te Satz der Messreihen (0: erster Zufalls-Seed, 1: zweiter)."""
    return sim()["saetze"][i]


def curve(kappa: int, variante: str) -> dict:
    """Kurven ARL₀(H), ARL₁(H) je λ₁ (Zero-state) für 'iid' oder 'kette'; Schwellen in Schritten von 4."""
    return kennlinien()["kappa"][str(kappa)][variante]


def at_target_zero(kappa: int, variante: str, lam1: float, target: float):
    """(H*, ARL₁) bei Zero-state-ARL₀ = target aus den gefrorenen Kurven (log-linear zwischen den Rasterpunkten); None, wenn target nicht im gerechneten Bereich liegt."""
    cv = curve(kappa, variante)
    r = CA.at_target(cv["H"], cv["arl0"], cv["arl1"][str(lam1)], target)
    return None if r is None else (float(r[0]), float(r[1]))


def steady_at_target(kappa: int, target: float):
    """Steady-state-Zahlen der Kette für 1,6 → 2,0 (nur Kette; quasistationär):
    'gleich': an derselben Schwelle wie die Zero-state-Spalte (dort Zero-state-ARL₀ = target): (ARL₁ steady, ARL₀ steady dort);
    'eq': an der Schwelle, bei der die Steady-state-ARL₀ selbst target ist: ARL₁ steady. Je None, wenn außerhalb des gerechneten Bereichs."""
    cv = curve(kappa, "kette")
    A0 = np.asarray(cv["arl0"], float)
    i = int(np.searchsorted(A0, target))
    out = {"gleich": None, "eq": None}
    if 0 < i < len(A0):
        t = (np.log(target) - np.log(A0[i - 1])) / (np.log(A0[i]) - np.log(A0[i - 1]))
        lo0, hi0, lo1, hi1 = cv["arl0_ss"][i - 1], cv["arl0_ss"][i], cv["arl1_ss"][i - 1], cv["arl1_ss"][i]
        if None not in (lo0, hi0, lo1, hi1):
            out["gleich"] = (lo1 + t * (hi1 - lo1), float(np.exp(np.log(lo0) + t * (np.log(hi0) - np.log(lo0)))))
    idx = [j for j, v in enumerate(cv["arl0_ss"]) if v is not None]
    for j in idx[:-1]:
        a, b = cv["arl0_ss"][j], cv["arl0_ss"][j + 1]
        if b is None:
            continue
        if a < target <= b:
            u = (np.log(target) - np.log(a)) / (np.log(b) - np.log(a))
            out["eq"] = cv["arl1_ss"][j] + u * (cv["arl1_ss"][j + 1] - cv["arl1_ss"][j])
            break
    return out


def k_reihe() -> list[dict]:
    """Die acht gemessenen k (2,25 bis 4,47) bei Zero-state-ARL₀ = 370: Schwelle und Verzögerung, unabhängig gegen Kette."""
    return sorted(kennlinien()["k_reihe"]["zeilen"], key=lambda z: z["k"])


def k_kontrolle() -> list[dict]:
    return kennlinien()["k_reihe"]["kontrolle_raster_q8"]


def iid_wartezeit() -> dict:
    return kennlinien()["iid_wartezeit"]
