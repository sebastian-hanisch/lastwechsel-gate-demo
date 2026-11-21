"""Orakel-Tests: die exakten Rechnungen der Demo gegen ANDERE Rechenwege.

(a) Übergangsmatrix M: Resolvente der Demo gegen das Integral ∫ λ e^{−λt} expm(Gt) dt (scipy.integrate.quad_vec).
(b) Randverteilung der gesehenen Zahl: Kette der Demo gegen die Geburts-Sterbe-Formel (PASTA); mittlere Wartezeit gegen die Formel mit den Erlang-Anteilen.
(c) Karte auf (n, C): gegen eine eigene 1-D-Kette bei unabhängigem n, gegen die Summe P(RL > t), gegen Monte-Carlo der Kette und gegen eine ereignisdiskrete Simulation des
    Gates Lkw für Lkw (stdlib random, FIFO mit c Spuren; kein Code der Demo).
(d) Bezugsgrößen für unabhängige Beobachtungen: Nyström-Verfahren gegen eine eigene Zellenkette und gegen Monte-Carlo; Zellenkette für die schiefe Randverteilung gegen Monte-Carlo.
Handrechnung mit Brüchen: tests/test_handbeispiele.py. Literaturwerte: keine (siehe tests/test_literatur_lwg.py)."""

import math
import random

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import gamma as gamma_d
from scipy.stats import norm

import lwg_cards as CA
import lwg_chain as CH
import lwg_constants as C
import lwg_iid as IID

G0, G1 = CH.Gate(1.6), CH.Gate(2.0)


# ----------------------------------------------------------------------------------------------------------------- (a) Übergangsmatrix
@pytest.mark.parametrize("lam", [1.6, 2.0, 3.0])
def test_m_resolvente_gleich_integral_der_matrixexponentialfunktion(lam):
    G = CH.death_gen()
    integral, _ = quad_vec(lambda t: lam * math.exp(-lam * t) * expm(G * t), 0.0, 60.0, epsabs=1e-12, epsrel=1e-12)
    assert np.abs(CH.m_matrix(lam) - integral).max() < 1e-8


def test_zeilensummen_von_m_und_r_sind_eins():
    for lam in (1.0, 1.6, 2.0, 4.0):
        assert np.allclose(CH.m_matrix(lam).sum(axis=1), 1.0, atol=1e-13)
        assert np.allclose(CH.acc_chain(lam).sum(axis=1), 1.0, atol=1e-13)


# ----------------------------------------------------------------------------------------------------------------- (b) Randverteilung
def birth_death_seen(lam, c=2, K=6, mu=1.0):
    """Geburts-Sterbe-Formel: zeitstationär p_j ∝ a^j/j! (j ≤ c), a^j/(c!·c^{j−c}) sonst; eine Ankunft sieht p (PASTA), angenommen bei j < K."""
    a = lam / mu
    p = [a ** j / math.factorial(j) if j <= c else a ** j / (math.factorial(c) * c ** (j - c)) for j in range(K + 1)]
    z = sum(p)
    p = [x / z for x in p]
    acc = np.array(p[:K]) / (1 - p[K])
    return acc, p[K]


@pytest.mark.parametrize("lam", [1.0, 1.6, 2.0, 3.0])
def test_randverteilung_gleich_geburts_sterbe_formel(lam):
    acc, loss = birth_death_seen(lam)
    assert np.abs(CH.Gate(lam).pi - acc).max() < 1e-12
    assert 0 < loss < 1


def test_verlust_und_wartezeit_von_markov_queue():
    """Zahlen der Quelle (markov-queue-demo): Verlust 7,60 %, mittlere Wartezeit 1,98 min bei Last 80 %; Mittel aus der Formel mit den Erlang-Anteilen."""
    acc, loss = birth_death_seen(1.6)
    assert loss == pytest.approx(0.0760, abs=5e-5)
    mean_formel = sum(acc[n] * (n - 1) / 2 for n in range(2, 6))
    assert G0.mean_w == pytest.approx(mean_formel, rel=1e-12)
    assert G0.mean_w * C.MIN_PRO_EINHEIT == pytest.approx(1.98, abs=0.005)


def test_standardabweichung_der_wartezeit_aus_den_momenten_der_erlangverteilung():
    acc, _ = birth_death_seen(1.6)
    m1 = sum(acc[n] * (n - 1) / 2 for n in range(2, 6))
    m2 = sum(acc[n] * (n - 1) * n / 4 for n in range(2, 6))        # E[Erlang(a, r)²] = a(a + 1)/r²
    assert G0.sd_w == pytest.approx(math.sqrt(m2 - m1 ** 2), rel=1e-12)
    assert G0.p0 == pytest.approx(acc[0] + acc[1], rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------- (c) Karte auf (n, C)
S20 = CA.raw_score(8, 20)


def one_d_chain_arl(pi, q, kappa, H):
    """Eigene 1-D-Kette für UNABHÄNGIGES n: Zustand C, Übergang mit Score q·m − κ und Wahrscheinlichkeit pi[m]."""
    Q = np.zeros((H, H))
    for Cs in range(H):
        for m in range(len(pi)):
            C2 = max(0, Cs + q * m - kappa)
            if C2 < H:
                Q[Cs, C2] += pi[m]
    return float(np.linalg.solve(np.eye(H) - Q, np.ones(H))[0])


def test_gitterkette_bei_unabhaengigem_n_gleich_eigener_1d_kette():
    Riid = np.tile(G0.pi, (6, 1))
    a2 = CA.exact_card(Riid, Riid, S20, 60, G0.pi)["arl0"]
    assert a2 == pytest.approx(one_d_chain_arl(G0.pi, 8, 20, 60), rel=1e-9)
    assert a2 == pytest.approx(82.848959, abs=5e-7)


def test_arl0_gleich_summe_der_ueberlebenswahrscheinlichkeiten():
    e = CA.exact_card(G0.R, G0.R, S20, 60, G0.pi, tmax=12000)
    assert e["surv0"].sum() == pytest.approx(e["arl0"], rel=1e-9)
    assert e["arl0"] == pytest.approx(28.8270, abs=5e-5)


def test_kette_gegen_monte_carlo_der_kette():
    rng = np.random.default_rng(1700)
    K, H, n_runs = 6, 60, 4000
    for R1, exakt_key in ((G0.R, "arl0"), (G1.R, "arl1")):
        cum = np.cumsum(R1, axis=1)
        n = rng.choice(K, size=n_runs, p=G0.pi)
        Cs = np.zeros(n_runs, dtype=np.int64)
        rl = np.zeros(n_runs)
        alive = np.ones(n_runs, bool)
        t = 0
        while alive.any():
            t += 1
            u = rng.random(n_runs)
            m = np.minimum((u[:, None] > cum[n]).sum(axis=1), K - 1)
            Cs = np.maximum(0, Cs + S20[n, m])
            n = m
            hit = alive & (Cs >= H)
            rl[hit] = t
            alive &= ~hit
        exakt = CA.exact_card(G0.R, R1, S20, H, G0.pi)[exakt_key]
        assert abs(rl.mean() - exakt) < 4 * rl.std(ddof=1) / math.sqrt(n_runs), (exakt_key, rl.mean(), exakt)


def event_sim_gate(rng, lam_pre, lam_post, burn, runs, H, kappa, c=2, K=6):
    """Ereignisdiskrete Simulation des Gates Lkw für Lkw (FIFO, c Spuren, K Plätze, stdlib random). Der burn-te angenommene Lkw ist ν: danach gilt lam_post, die Karte startet bei 0."""
    out = []
    for _ in range(runs):
        t, lam, acc, Cs, steps = 0.0, lam_pre, 0, 0.0, 0
        free = [0.0] * c
        ends = []
        while True:
            t += rng.expovariate(lam)
            ends = [e for e in ends if e > t]
            n = len(ends)
            if n >= K:
                continue
            j = min(range(c), key=lambda i: free[i])
            start = max(t, free[j])
            end = start + rng.expovariate(1.0)
            free[j] = end
            ends.append(end)
            acc += 1
            if acc == burn:
                lam = lam_post
            elif acc > burn:
                steps += 1
                Cs = max(0, Cs + 8 * n - kappa)
                if Cs >= H:
                    out.append(steps)
                    break
    return out


@pytest.mark.parametrize("lam_b, key", [(1.6, "arl0"), (2.0, "arl1")])
def test_kette_gegen_ereignissimulation_des_gates(lam_b, key):
    rng = random.Random(20261119)
    v = event_sim_gate(rng, 1.6, lam_b, 60, 5000, 60, 20)
    mean = sum(v) / len(v)
    se = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / math.sqrt(len(v))
    exakt = CA.exact_card(G0.R, G1.R if key == "arl1" else G0.R, S20, 60, G0.pi)[key]
    assert abs(mean - exakt) < 4 * se, (key, mean, se, exakt)


def test_zusammenbruch_h97_gegen_ereignissimulation():
    """Die Kernzahl des Stücks, ein anderer Weg: Schwelle H = 97 (unabhängig 408,2) hat auf dem Gate ARL₀ 51,2 (Ereignisdiskrete Simulation, 4 Standardfehler)."""
    rng = random.Random(97)
    v = event_sim_gate(rng, 1.6, 1.6, 60, 4000, 97, 20)
    mean = sum(v) / len(v)
    se = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / math.sqrt(len(v))
    assert abs(mean - 51.2431) < 4 * se, (mean, se)


# ----------------------------------------------------------------------------------------------------------------- (d) Bezugsgrößen für unabhängige Beobachtungen
def brook_evans_normal(k, h, delta, m):
    """Eigene Zellenkette (Brook/Evans) für den einseitigen CUSUM bei x ~ N(delta, 1): Zelle 0 = [0, w/2), Zelle j = [(j − 1/2)w, (j + 1/2)w), w = h/(m − 1/2)."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    Cdf = norm.cdf(upper[None, :] - v[:, None] + k - delta)
    Rm = np.diff(Cdf, axis=1, prepend=0.0)
    return float(np.linalg.solve(np.eye(m) - Rm, np.ones(m))[0])


@pytest.mark.parametrize("k, h, delta", [(0.5, 4.0965, 0.0), (0.1393, 9.2627, 0.0), (0.1393, 9.2627, 0.2786), (0.5, 3.0, 1.0)])
def test_nystroem_gleich_eigener_zellenkette(k, h, delta):
    assert IID.normal_arl(k, h, delta) == pytest.approx(brook_evans_normal(k, h, delta, 500), rel=5e-4)


def test_einseitiger_cusum_h40965_gibt_370_4():
    """Einseitig k = 0,5, h = 4,0965 hat ARL₀ ≈ 370,4 (gleich 1/(2·Φ(−3)) = 370,398 der 3σ-Shewhart-Karte; h auf vier Stellen gerundet, Band 2·10⁻⁵)."""
    assert IID.normal_arl(0.5, 4.0965) == pytest.approx(1 / (2 * norm.sf(3)), rel=2e-5)


def test_normalkarte_gegen_monte_carlo():
    rng = np.random.default_rng(11)
    k, h = 0.1393, 9.2627
    for delta, exakt in ((0.0, IID.normal_arl(k, h)), (0.2786, IID.normal_arl(k, h, 0.2786))):
        n_runs = 3000
        Cs = np.zeros(n_runs)
        rl = np.zeros(n_runs)
        alive = np.ones(n_runs, bool)
        t = 0
        while alive.any():
            t += 1
            Cs = np.maximum(0.0, Cs + rng.normal(delta, 1.0, n_runs) - k)
            hit = alive & (Cs >= h)
            rl[hit] = t
            alive &= ~hit
        assert abs(rl.mean() - exakt) < 4 * rl.std(ddof=1) / math.sqrt(n_runs)


def test_schiefe_randverteilung_der_wartezeit_zellenkette_gegen_monte_carlo():
    """iid mit der exakten Randverteilung der Wartezeit (Atom 0,4076 bei 0 plus Erlang-Anteile), Schwelle h* = 10,25: ARL₀ = 370 (Zellenkette) gegen Monte-Carlo aus der Mischung."""
    import lwg_reference as REF   # gefrorene Werte (die Neurechnung prüft tests/test_tables.py)
    ref = REF.iid_wartezeit()
    h, a1 = ref["exakt"]["400"]["h"], ref["exakt"]["400"]["arl1"]
    k = ref["k"]
    rng = np.random.default_rng(5)

    def draw(g, size):
        n = rng.choice(6, size=size, p=g.pi)
        w = np.zeros(size)
        big = n >= 2
        w[big] = rng.gamma(shape=n[big] - 1, scale=0.5)
        return (w - G0.mean_w) / G0.sd_w
    for g, exakt in ((G0, 370.0), (G1, a1)):
        n_runs = 1500
        Cs = np.zeros(n_runs)
        rl = np.zeros(n_runs)
        idx = np.arange(n_runs)          # nur die Läufe ohne Alarm werden weiter gezogen
        t = 0
        while idx.size:
            t += 1
            Cs[idx] = np.maximum(0.0, Cs[idx] + draw(g, idx.size) - k)
            hit = Cs[idx] >= h
            rl[idx[hit]] = t
            idx = idx[~hit]
        assert abs(rl.mean() - exakt) < 4 * rl.std(ddof=1) / math.sqrt(n_runs), (g.lam, rl.mean(), exakt)


def test_wartezeit_cdf_gleich_scipy_gamma():
    cdf = IID.wait_cdf_factory(G0.pi, G0.mean_w, G0.sd_w)
    z = np.array([-2.0, -0.7, 0.0, 0.5, 2.0, 6.0])
    w = z * G0.sd_w + G0.mean_w
    ref = np.where(w >= 0, G0.pi[:2].sum(), 0.0) + sum(np.where(w >= 0, G0.pi[n] * gamma_d.cdf(np.maximum(w, 0), n - 1, scale=0.5), 0.0) for n in range(2, 6))
    assert np.allclose(cdf(z), ref, atol=1e-13)
