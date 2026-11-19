"""CUSUM-Karten auf der gesehenen Zahl n als endliche Ketten (exakt).

Score s(n, m) ganzzahlig (Übergang n → m), C' = max(0, C + s), Alarm bei C ≥ H. Dann ist (n, C) mit C = 0 … H − 1 eine endliche Kette, und ARL, Lauflängenverteilung und
Verzögerung sind lineare Gleichungssysteme. Zähleinheit überall: angenommene Lkw (Abgewiesene zählen nicht); der Alarm-Lkw zählt mit.
ARL₀ = mittlere Zahl Lkw bis zum Alarm ohne Änderung (Zero-state: Karte 0, n ~ pi der alten Rate); ARL₁ = mittlere Zahl Lkw von der Änderung bis zum Alarm
(Zero-state: Karte startet beim letzten Lkw ν vor der Änderung mit 0, das Gate ist im alten Zustand stationär). Steady-state: Karte lange ohne Alarm gelaufen
(quasistationäre Verteilung von (n, C)). Das ist weder das Lorden- noch das Pollak-Maß."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import lwg_constants as C


class ArlTooLarge(Exception):
    """Die ARL₀ der Karte liegt über der Reglergrenze (2·10⁵): der Löser liefert dort keine verlässliche Zahl."""


def raw_score(q: int, kappa: int, K: int = C.K_CAP) -> np.ndarray:
    """Score der Rohkarte auf n: s = q·m − κ (nur vom neuen n = m abhängig); K × K für den Übergang n → m."""
    return np.array([[q * m - kappa for m in range(K)] for _ in range(K)], dtype=np.int64)


def lr_score(R1: np.ndarray, R0: np.ndarray, qs: int) -> tuple[np.ndarray, np.ndarray]:
    """Score der Likelihood-Quotient-Karte: ln R1(n, m)/R0(n, m), auf das Gitter 1/qs gerundet; zweiter Rückgabewert: der ungerundete Logarithmus."""
    mask = R0 > 1e-15
    llr = np.zeros_like(R0)
    llr[mask] = np.log(R1[mask] / R0[mask])
    return np.rint(qs * llr).astype(np.int64), llr


def build_Q(R: np.ndarray, S: np.ndarray, H: int) -> sp.csr_matrix:
    """Übergangsmatrix der Karte auf den Zuständen (n, C), Index n·H + C, C = 0 … H − 1; die Alarm-Masse fällt heraus (substochastisch)."""
    K = R.shape[0]
    C_ = np.arange(H)
    rows, cols, data = [], [], []
    for n in range(K):
        for m in range(K):
            p = R[n, m]
            if p <= 0:
                continue
            C2 = np.maximum(0, C_ + int(S[n, m]))
            ok = C2 < H
            rows.append(n * H + C_[ok])
            cols.append(m * H + C2[ok])
            data.append(np.full(int(ok.sum()), p))
    if not rows:
        return sp.csr_matrix((K * H, K * H))
    return sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(K * H, K * H))


def arl_vec(Q: sp.spmatrix) -> np.ndarray:
    """Erwartete Zahl Beobachtungen bis zum Alarm je Startzustand: (I − Q) L = 1."""
    n = Q.shape[0]
    return spl.spsolve(sp.identity(n, format="csc") - Q.tocsc(), np.ones(n))


def start_vec(pi: np.ndarray, H: int) -> np.ndarray:
    """Startverteilung der Zero-state-Rechnung: C = 0, n ~ pi."""
    K = len(pi)
    v = np.zeros(K * H)
    v[np.arange(K) * H] = pi
    return v


def surv_curve(Q: sp.spmatrix, v0: np.ndarray, tmax: int) -> np.ndarray:
    """P(RL > t) für t = 0 … tmax (die Karte hat nach t Lkw noch nicht angeschlagen)."""
    out = np.empty(tmax + 1)
    v = v0.copy()
    out[0] = v.sum()
    QT = Q.T.tocsr()
    for t in range(1, tmax + 1):
        v = QT @ v
        out[t] = v.sum()
    return out


def quasi_stat(Q: sp.spmatrix, iters: int = 20000, tol: float = 1e-13) -> np.ndarray:
    """Quasistationäre Verteilung (Karte läuft lange ohne Alarm): Potenzmethode auf dem bedingten Zustandsvektor."""
    QT = Q.T.tocsr()
    v = np.ones(Q.shape[0])
    v /= v.sum()
    for _ in range(iters):
        w = QT @ v
        w /= w.sum()
        if np.abs(w - v).max() < tol:
            return w
        v = w
    return v


def too_large(Q: sp.spmatrix, v0: np.ndarray) -> bool:
    """Schutz vor unbrauchbaren Lösungen: überlebt die Karte SURV_STEPS Lkw mit mindestens SURV_LIMIT, ist ARL₀ > 2·10⁵ (Heuristik: geometrischer Tail)."""
    v = v0.copy()
    QT = Q.T.tocsr()
    for _ in range(C.SURV_STEPS):
        v = QT @ v
    return float(v.sum()) >= C.SURV_LIMIT


def exact_card(Rpre: np.ndarray, Rpost: np.ndarray, S: np.ndarray, H: int, pi_pre: np.ndarray, steady: bool = False, tmax: int = 0, guard: bool = False) -> dict:
    """ARL₀ (Rate vor der Änderung), ARL₁ (Rate nach der Änderung), beide Zero-state; optional Steady-state und Überlebenskurven.

    guard=True: ArlTooLarge, wenn ARL₀ über 2·10⁵ liegt oder der Löser keine endliche Zahl ≥ 1 liefert (Anzeige: Meldung statt Zahl)."""
    Q0 = build_Q(Rpre, S, H)
    v0 = start_vec(pi_pre, H)
    if guard and too_large(Q0, v0):
        raise ArlTooLarge(H)
    L0 = arl_vec(Q0)
    out = {"arl0": float(v0 @ L0)}
    if guard and not (np.isfinite(out["arl0"]) and 1.0 <= out["arl0"] <= C.ARL_CAP * 1.5):
        raise ArlTooLarge(H)
    same = Rpost is Rpre
    Q1 = Q0 if same else build_Q(Rpost, S, H)
    L1 = L0 if same else arl_vec(Q1)
    out["arl1"] = float(v0 @ L1)
    if steady:
        qs = quasi_stat(Q0)
        out["arl0_ss"] = float(qs @ L0)
        out["arl1_ss"] = float(qs @ L1)
    if tmax:
        out["surv0"] = surv_curve(Q0, v0, tmax)
        out["surv1"] = out["surv0"] if same else surv_curve(Q1, v0, tmax)
    return out


def scan(Rpre, Rpost, S, Hs, pi_pre, cap: float = C.ARL_CAP, steady_window=None) -> dict:
    """ARL₀(H) und ARL₁(H) auf einem Raster von Schwellen; bricht ab, sobald ARL₀ über cap liegt. steady_window = (lo, hi): Steady-state-Spalten, solange lo ≤ ARL₀ ≤ hi."""
    out = {"H": [], "arl0": [], "arl1": [], "arl0_ss": [], "arl1_ss": []}
    for H in Hs:
        e = exact_card(Rpre, Rpost, S, int(H), pi_pre)
        want = steady_window is not None and steady_window[0] <= e["arl0"] <= steady_window[1]
        if want:
            e = exact_card(Rpre, Rpost, S, int(H), pi_pre, steady=True)
        out["H"].append(int(H))
        out["arl0"].append(e["arl0"])
        out["arl1"].append(e["arl1"])
        out["arl0_ss"].append(e.get("arl0_ss"))
        out["arl1_ss"].append(e.get("arl1_ss"))
        if e["arl0"] > cap:
            break
    return out


def at_target(Hs, A0, A1, target: float):
    """Log-lineare Interpolation von ARL₀(H) auf das Ziel; gleiche Gewichte für ARL₁. Rückgabe (H, ARL₁, t) oder None, wenn das Ziel nicht im Raster liegt."""
    A0 = np.asarray(A0, float)
    i = int(np.searchsorted(A0, target))
    if i == 0 or i >= len(Hs):
        return None
    t = (np.log(target) - np.log(A0[i - 1])) / (np.log(A0[i]) - np.log(A0[i - 1]))
    return Hs[i - 1] + t * (Hs[i] - Hs[i - 1]), A1[i - 1] + t * (A1[i] - A1[i - 1]), t


def at_target_exact(Rpre, Rposts: dict, S, pi_pre, target: float = C.ARL0_NENN, hi: int = 700) -> dict | None:
    """Wie `at_target`, aber ohne Raster: die beiden Schwellen H − 1 und H um das Ziel (H = kleinste Schwelle mit ARL₀ ≥ target) werden gerechnet und log-linear
    interpoliert (ARL₁ mit denselben Gewichten). Rposts: {Name: Matrix nach der Änderung}. Rückgabe: h (interpoliert), h_int (= H), arl0_lo/hi und je Name arl1
    (interpoliert) und arl1_hi (an der ganzzahligen Schwelle H); None, wenn auch hi das Ziel nicht erreicht."""
    H = smallest_h(Rpre, S, pi_pre, target, hi=hi)
    if H is None or H <= 2:
        return None
    lo, hi_ = H - 1, H
    e_lo = {k: exact_card(Rpre, R1, S, lo, pi_pre) for k, R1 in Rposts.items()}
    e_hi = {k: exact_card(Rpre, R1, S, hi_, pi_pre) for k, R1 in Rposts.items()}
    k0 = next(iter(Rposts))
    a_lo, a_hi = e_lo[k0]["arl0"], e_hi[k0]["arl0"]
    t = (np.log(target) - np.log(a_lo)) / (np.log(a_hi) - np.log(a_lo))
    return {"h": lo + t * (hi_ - lo), "h_int": H, "arl0_lo": a_lo, "arl0_hi": a_hi, "t": float(t),
            "arl1": {k: e_lo[k]["arl1"] + t * (e_hi[k]["arl1"] - e_lo[k]["arl1"]) for k in Rposts}, "arl1_hi": {k: e_hi[k]["arl1"] for k in Rposts}}


def smallest_h(Rpre, S, pi_pre, target: float = C.ARL0_NENN, lo: int = 2, hi: int = 1400) -> int | None:
    """Kleinste ganzzahlige Schwelle mit ARL₀ ≥ target (Bisektion; ARL₀(H) wächst mit H). None, wenn auch hi das Ziel nicht erreicht."""
    def a0(H):
        try:
            return exact_card(Rpre, Rpre, S, H, pi_pre, guard=True)["arl0"]
        except ArlTooLarge:
            return float("inf")   # über der Reglergrenze: sicher größer als jedes Ziel, der Löser liefert dort keine verlässliche Zahl
    if a0(hi) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if a0(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo
