"""Gate-Simulator und Karten auf der Wartezeit (Simulation, gelb).

R unabhängige M/M/2/6-Gates laufen im Takt der angenommenen Lkw nebeneinander (vektorisiert); jeder Schritt liefert je Lauf den NÄCHSTEN ANGENOMMENEN Lkw: die gesehene
Zahl n im Gate und seine Wartezeit W (FIFO). Abgewiesene Lkw werden übersprungen. Zufall nur aus `lwg_normal.Stream` (SplitMix64, Stapel-Fassung).
Karten: Rohkarte, Ausdünnen, Blockmittel, Karte auf n, Residuenkarte (W gegeben n), naiver zweiseitiger CUSUM (V2). Jede Karte läuft mit einem Raster von Schwellen mit
(`Grid`: erste Durchgangszeit je Schwelle), so dass aus EINEM Lauf ohne Änderung die Schwelle für ARL₀ = 370 und aus den Läufen nach der Änderung die ARL₁ folgen.
Zähleinheit: angenommene Lkw; Verzögerung = Zahl der Lkw von der Änderung bis zum Alarm (ν = letzter Lkw vor der Änderung, der Alarm-Lkw zählt mit), Zero-state."""

import numpy as np
from scipy.special import gammainc, ndtri

import lwg_constants as C
import lwg_normal as N


class GateSim:
    """R unabhängige Gates; `step(lam, mu)` liefert (n, w) des nächsten angenommenen Lkw je Lauf."""

    def __init__(self, stream: N.Stream, R: int, c: int = C.C_SERV, K: int = C.K_CAP):
        self.st, self.R, self.c, self.K = stream, R, c, K
        self.t = np.zeros(R)
        self.ends = np.zeros((R, K))
        self.free = np.zeros((R, c))

    def step(self, lam: float, mu: float):
        st, K = self.st, self.K
        n_out = np.empty(self.R, dtype=np.int64)
        w_out = np.empty(self.R)
        pending = np.arange(self.R)
        while pending.size:
            self.t[pending] += st.exponential(pending.size, 1.0 / lam)
            tp = self.t[pending]
            N_ = (self.ends[pending] > tp[:, None]).sum(axis=1)
            acc = N_ < K
            rows = pending[acc]
            if rows.size:
                tr = self.t[rows]
                j = np.argmin(self.free[rows], axis=1)
                start = np.maximum(tr, self.free[rows, j])
                endt = start + st.exponential(rows.size, 1.0 / mu)
                self.free[rows, j] = endt
                slot = np.argmin(self.ends[rows], axis=1)
                self.ends[rows, slot] = endt
                n_out[rows] = N_[acc]
                w_out[rows] = start - tr
            pending = pending[~acc]
        return n_out, w_out


class Grid:
    """Erste Durchgangszeit der freilaufenden Statistik über ein Raster aufsteigender Schwellen hg. rl[r, g] = Zeit (in Lkw), 0 = nicht erreicht."""

    def __init__(self, R: int, hg):
        self.hg = np.asarray(hg, float)
        self.G = len(self.hg)
        self.rl = np.zeros((R, self.G), dtype=np.int32)
        self.idx = np.zeros(R, dtype=np.int64)

    def update(self, t: int, Cs: np.ndarray):
        G = self.G
        while True:
            cur = self.hg[np.minimum(self.idx, G - 1)]
            m = (Cs >= cur) & (self.idx < G)
            if not m.any():
                return
            r = np.nonzero(m)[0]
            self.rl[r, self.idx[r]] = t
            self.idx[r] += 1


class Card:
    name = "?"
    needs_pre = False

    def __init__(self, R: int, hg):
        self.R = R
        self.C = np.zeros(R)
        self.grid = Grid(R, hg)


class RawCard(Card):
    """Rohkarte auf der standardisierten Wartezeit: C' = max(0, C + (w − mu0)/sd0 − k)."""

    def __init__(self, R, hg, k, mu0, sd0, name):
        super().__init__(R, hg)
        self.k, self.mu0, self.sd0, self.name = k, mu0, sd0, name

    def update(self, t, n, w):
        z = (w - self.mu0) / self.sd0
        self.C = np.maximum(0.0, self.C + z - self.k)
        self.grid.update(t, self.C)


class ThinCard(Card):
    """Ausdünnen: nur jeder b-te Lkw geht in die Karte (t = b, 2b, …)."""

    def __init__(self, R, hg, k, mu0, sd0, b, name):
        super().__init__(R, hg)
        self.k, self.mu0, self.sd0, self.b, self.name = k, mu0, sd0, b, name

    def update(self, t, n, w):
        if t % self.b:
            return
        z = (w - self.mu0) / self.sd0
        self.C = np.maximum(0.0, self.C + z - self.k)
        self.grid.update(t, self.C)


class BlockCard(Card):
    """Blockmittel: das Mittel von b Lkw ist die Beobachtung (standardisiert mit der Standardabweichung der Blockmittel sdb).

    phase p (0 ≤ p < b): die Änderung liegt p Lkw vor dem Ende des ersten Blocks, der erste Block enthält also p Lkw vor ν (phase 0: Änderung auf der Blockgrenze).
    Blöcke schließen bei t + p ≡ 0 (mod b)."""
    needs_pre = True

    def __init__(self, R, hg, k, mu0, sdb, b, name, phase=0):
        super().__init__(R, hg)
        self.k, self.mu0, self.sdb, self.b, self.name, self.phase = k, mu0, sdb, b, name, phase
        self.acc = np.zeros(R)

    def start(self, last_w):
        """last_w: die W-Vektoren der letzten Lkw vor ν (neueste zuletzt); die letzten `phase` gehören in den ersten Block."""
        self.acc[:] = 0.0
        if self.phase:
            for w in last_w[-self.phase:]:
                self.acc += w

    def update(self, t, n, w):
        self.acc += w
        if (t + self.phase) % self.b:
            return
        z = (self.acc / self.b - self.mu0) / self.sdb
        self.acc[:] = 0.0
        self.C = np.maximum(0.0, self.C + z - self.k)
        self.grid.update(t, self.C)


class NCard(Card):
    """Karte auf der gesehenen Zahl mit ganzzahligem Score q·n − κ (Simulation derselben Karte, die lwg_cards exakt rechnet)."""

    def __init__(self, R, hg, q, kappa, name):
        super().__init__(R, hg)
        self.q, self.kappa, self.name = q, kappa, name

    def update(self, t, n, w):
        self.C = np.maximum(0.0, self.C + self.q * n - self.kappa)
        self.grid.update(t, self.C)


class ResidCard(Card):
    """Residuenkarte: U = F(W | n) (Erlang(n − c + 1, c·mu); bei n < c ist W = 0 und U zufällig gleichverteilt: randomisierte PIT), z = Φ⁻¹(U); entworfen mit mu = 1."""

    def __init__(self, R, hg, k, stream, name, c=C.C_SERV, mu=C.MU):
        super().__init__(R, hg)
        self.k, self.st, self.name, self.c, self.mu = k, stream, name, c, mu

    def update(self, t, n, w):
        a = np.maximum(n - self.c + 1, 1).astype(float)
        u = gammainc(a, w * self.c * self.mu)
        u = np.where(n < self.c, self.st.uniform(self.R), u)
        z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        self.C = np.maximum(0.0, self.C + z - self.k)
        self.grid.update(t, self.C)


class TwoSidedFixed:
    """V2 6.1: zweiseitiger CUSUM k = 0,5, h = 4,77 auf der standardisierten Wartezeit; erste Durchgangszeit (Alarm bei C > h)."""
    name = "naiv zweiseitig k=0,5 h=4,77"
    needs_pre = False

    def __init__(self, R, mu0, sd0, k=C.V2_K, h=C.V2_H):
        self.cp = np.zeros(R)
        self.cn = np.zeros(R)
        self.rl = np.zeros(R, dtype=np.int64)
        self.k, self.h, self.mu0, self.sd0 = k, h, mu0, sd0

    def update(self, t, n, w):
        z = (w - self.mu0) / self.sd0
        self.cp = np.maximum(0.0, self.cp + z - self.k)
        self.cn = np.maximum(0.0, self.cn - z - self.k)
        hit = ((self.cp > self.h) | (self.cn > self.h)) & (self.rl == 0)
        self.rl[hit] = t


def run_pass(seed: int, R: int, burn: int, T: int, lam0: float, lam1: float, mu0: float, mu1: float, cards) -> GateSim:
    """burn Lkw mit (lam0, mu0), dann T Lkw mit (lam1, mu1); die Karten starten bei ν = burn mit 0 (Zero-state) und bekommen t = 1 … T."""
    g = GateSim(N.Stream(seed), R)
    ring = []
    for _ in range(burn):
        _, w = g.step(lam0, mu0)
        ring.append(w)
        if len(ring) > 64:
            ring.pop(0)
    for c in cards:
        if getattr(c, "needs_pre", False):
            c.start(ring)
    for t in range(1, T + 1):
        n, w = g.step(lam1, mu1)
        for c in cards:
            c.update(t, n, w)
    return g


def arl_from_grid(grid: Grid, T: int):
    """Mittel über Läufe je Schwelle (nicht erreicht: T gesetzt = untere Schranke), Standardfehler, Anteil nicht erreicht."""
    rl = grid.rl.astype(float)
    cens = rl == 0
    rl2 = np.where(cens, float(T), rl)
    R = rl.shape[0]
    return rl2.mean(axis=0), rl2.std(axis=0, ddof=1) / np.sqrt(R), cens.mean(axis=0)


def card_at_target(pre: Card, post: Card, T0: int, T1: int, target: float = C.ARL0_NENN, max_cens: float = 0.002):
    """Schwelle h* mit ARL₀ = target (log-linear zwischen Rasterpunkten); ARL₁ dort mit Standardfehler. Der Standardfehler enthält die Unschärfe der Schwelle
    (aus dem Lauf ohne Änderung): se² = s1² + (Steigung · se(ARL₀)/ARL₀)². None, wenn das Ziel nicht im Raster liegt."""
    a0, s0, c0 = arl_from_grid(pre.grid, T0)
    a1, s1, c1 = arl_from_grid(post.grid, T1)
    hg = pre.grid.hg
    ok = c0 <= max_cens
    i = None
    for g in range(len(hg) - 1):
        if ok[g] and ok[g + 1] and a0[g] < target <= a0[g + 1]:
            i = g
            break
    if i is None:
        return None
    t = (np.log(target) - np.log(a0[i])) / (np.log(a0[i + 1]) - np.log(a0[i]))
    hs = hg[i] + t * (hg[i + 1] - hg[i])
    A1 = a1[i] + t * (a1[i + 1] - a1[i])
    if max(c1[i], c1[i + 1]) > max_cens:
        return dict(h=float(hs), arl1=float("nan"), se1=float("nan"))
    slope = (a1[i + 1] - a1[i]) / (np.log(a0[i + 1]) - np.log(a0[i]))
    se = float(np.sqrt(s1[i] ** 2 + (slope * s0[i] / a0[i]) ** 2))
    return dict(h=float(hs), arl1=float(A1), se1=se)


def acf_lags(W: np.ndarray, lags=(1, 2, 3, 4, 5, 10, 20, 40)) -> dict:
    """Autokorrelation der Wartezeit je Lag (Mittel über die Läufe als gemeinsame Reihe: Zeilen sind unabhängige Läufe)."""
    X = W - W.mean()
    v = (X * X).mean()
    return {lag: float((X[:, :-lag] * X[:, lag:]).mean() / v) for lag in lags}


def block_stats(W: np.ndarray, sd_w: float, blocks=(5, 10, 20, 40)) -> dict:
    """Standardabweichung der Blockmittel, Faktor gegen sd_w/√b und Lag-1-Autokorrelation der Blockmittel je Blocklänge."""
    out = {}
    R, T = W.shape
    for b in blocks:
        nb = T // b
        B = W[:, : nb * b].reshape(R, nb, b).mean(axis=2)
        sd = float(B.std()) if nb >= 1 else float("nan")
        Xb = B - B.mean() if nb >= 1 else B
        den = float((Xb * Xb).mean()) if nb >= 1 else 0.0
        if nb < 2:
            l1 = float("nan")                                                      # weniger als zwei Blöcke je Lauf: kein Lag zu messen
        else:
            l1 = float((Xb[:, :-1] * Xb[:, 1:]).mean() / den) if den > 0 else 0.0   # konstante Blockmittel: keine Abhängigkeit zu messen
        out[b] = {"sd": sd, "faktor": sd / (sd_w / np.sqrt(b)), "lag1": l1}
    return out


def collect_waits(seed: int, R: int, burn: int, T: int, lam: float = C.LAM0, mu: float = C.MU) -> np.ndarray:
    """Wartezeiten (R × T) der angenommenen Lkw nach dem Einschwingen (burn Lkw verworfen)."""
    g = GateSim(N.Stream(seed), R)
    for _ in range(burn):
        g.step(lam, mu)
    W = np.empty((R, T))
    for t in range(T):
        _, W[:, t] = g.step(lam, mu)
    return W


def live_probe(R: int, T: int, seed: int, mean_w: float, sd_w: float, burn: int = C.LIVE_BURN) -> dict:
    """Kleine Live-Probe (stationäres Gate, λ = 1,6, ohne Änderung): Autokorrelation, Blockmittel und der naive zweiseitige CUSUM aus V2 auf der Wartezeit.
    Rückgabe: acf (Lag 1 bis 5), blocks (je b: Faktor, Lag-1), rl (Lauflängen, 0 = kein Alarm in T Lkw), Mittel und Anteil der Alarme bis 100 Lkw."""
    st = N.Stream(seed)
    g = GateSim(st, R)
    for _ in range(burn):
        g.step(C.LAM0, C.MU)
    naiv = TwoSidedFixed(R, mean_w, sd_w)
    W = np.empty((R, T))
    for t in range(1, T + 1):
        n, w = g.step(C.LAM0, C.MU)
        W[:, t - 1] = w
        naiv.update(t, n, w)
    rl = np.where(naiv.rl == 0, T, naiv.rl).astype(float)
    return {
        "R": R, "T": T,
        "acf": acf_lags(W, (1, 2, 3, 4, 5)),
        "blocks": block_stats(W, sd_w),
        "arl0": float(rl.mean()), "se": float(rl.std(ddof=1) / np.sqrt(R)),
        "p100": float(np.mean(rl <= 100)), "cens": float(np.mean(naiv.rl == 0)),
        "mean_w": float(W.mean()),
    }
