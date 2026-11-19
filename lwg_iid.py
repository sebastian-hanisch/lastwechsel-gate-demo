"""Bezugsgrößen der Lehrbuchrechnung: einseitiger CUSUM bei UNABHÄNGIGEN Beobachtungen (Normalverteilung; exakte schiefe Randverteilung der Gate-Wartezeit).

Normalverteilung: Integralgleichung (Page; Nyström-Verfahren mit Gauß-Legendre-Knoten). Wartezeit: Brook-Evans-Kette (Zellen) mit der exakten Randverteilung der Wartezeit eines
angenommenen Lkw (Atom bei 0 plus Erlang-Anteile, Gewichte aus der Kette der gesehenen Zahl). Zähleinheit: Beobachtungen = angenommene Lkw (iid: ein Lkw je Beobachtung)."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import gammainc, ndtr

import lwg_constants as C


def _phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / np.sqrt(2 * np.pi)


def normal_arl(k: float, h: float, delta: float = 0.0, N: int = 60) -> float:
    """ARL (Zero-state, Start 0) des einseitigen CUSUM C' = max(0, C + x − k), Alarm bei C ≥ h, x ~ N(delta, 1); Nyström mit N Gauß-Legendre-Knoten."""
    x, wq = leggauss(N)
    y = 0.5 * h * (x + 1)
    wq = 0.5 * h * wq
    u = np.concatenate([[0.0], y])
    A = np.zeros((N + 1, N + 1))
    A[:, 0] = ndtr(k - u - delta)
    A[:, 1:] = _phi(y[None, :] - u[:, None] + k - delta) * wq[None, :]
    L = np.linalg.solve(np.eye(N + 1) - A, np.ones(N + 1))
    return float(L[0])


def normal_h_for_arl(k: float, target: float, N: int = 60) -> float:
    """Schwelle h mit ARL₀ = target (Normalverteilung, einseitig)."""
    hi = 1.0
    while normal_arl(k, hi, 0.0, N) < target:      # obere Klammer von unten suchen: bei großem h ist das Nyström-System schlecht konditioniert
        hi *= 1.5
    return brentq(lambda h: normal_arl(k, h, 0.0, N) - target, 0.1, hi, xtol=1e-12)


def wait_cdf_factory(pi: np.ndarray, mean0: float, sd0: float, c: int = C.C_SERV, mu: float = C.MU):
    """Verteilungsfunktion der auf (mean0, sd0) standardisierten Wartezeit bei der gesehenen-Zahl-Verteilung pi (Atom bei 0: Anteil der n < c)."""
    p0 = float(pi[:c].sum())
    rate = c * mu
    parts = [(float(pi[n]), n - c + 1) for n in range(c, len(pi))]

    def cdf(z):
        w = np.asarray(z, float) * sd0 + mean0
        res = np.where(w >= 0, p0, 0.0)
        wp = np.maximum(w, 0.0)
        for wk, a in parts:
            res = res + np.where(w >= 0, wk * gammainc(a, rate * wp), 0.0)
        return res
    return cdf


def brook_evans_q(cdf, k: float, h: float, n: int) -> np.ndarray:
    """Übergangsmatrix der Brook-Evans-Kette für C' = max(0, C + x − k): Zustand 0 und n Zellen gleicher Breite über (0, h]."""
    wdt = h / n
    reps = np.concatenate([[0.0], (np.arange(n) + 0.5) * wdt])
    edges = np.arange(n + 1) * wdt
    F = cdf(k - reps[:, None] + edges[None, :])
    Q = np.empty((n + 1, n + 1))
    Q[:, 0] = cdf(k - reps)
    Q[:, 1:] = F[:, 1:] - F[:, :-1]
    return np.maximum(Q, 0.0)


def chain_arl0(Q: np.ndarray) -> float:
    """ARL vom Zustand 0 aus: (I − Q) L = 1."""
    return float(np.linalg.solve(np.eye(Q.shape[0]) - Q, np.ones(Q.shape[0]))[0])


def iid_wait_reference(g0, g1, target: float = C.ARL0_NENN, ms=(200, 400, 800)) -> dict:
    """Die iid-Welt für dieselbe Sprunggröße wie am Gate (Karte auf der Wartezeit, k = Sprung/2 in Standardabweichungen der alten Wartezeit):
    Normalverteilung und exakte Randverteilung der Wartezeit, je Schwelle für ARL₀ = target und ARL₁ nach dem Sprung (Zero-state)."""
    shift = (g1.mean_w - g0.mean_w) / g0.sd_w
    k = shift / 2
    h = normal_h_for_arl(k, target)
    out = {"shift": shift, "k": k, "normal": {"h": h, "arl0": normal_arl(k, h), "arl1": normal_arl(k, h, shift)}, "exakt": {}}
    cdf0 = wait_cdf_factory(g0.pi, g0.mean_w, g0.sd_w)
    cdf1 = wait_cdf_factory(g1.pi, g0.mean_w, g0.sd_w)
    for m in ms:
        hs = brentq(lambda hh: chain_arl0(brook_evans_q(cdf0, k, hh, m)) - target, 0.5, 40.0, xtol=1e-9)
        out["exakt"][str(m)] = {"h": hs, "arl1": chain_arl0(brook_evans_q(cdf1, k, hs, m))}
    return out
