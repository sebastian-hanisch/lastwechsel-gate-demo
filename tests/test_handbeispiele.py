"""Handbeispiele des Mini-Gates (eine Spur, drei Plätze insgesamt, λ₀ = 1 → λ₁ = 2, Karte C' = max(0, C + n − 1), Alarm bei C ≥ 2) als exakte Tests.

Zwei Seiten: (1) die Handrechnung mit Brüchen (`tests/mini_gate.py`, kein Code der Demo) gegen die von Hand hergeleiteten Werte; (2) der Code der Demo (numpy, beliebige Spurenzahl und Plätze)
gegen dieselben Brüche. Die Werte stehen als Brüche im Test; sie sind von Hand nachgerechnet (Gleichungen in den Kommentaren)."""

import math
import random
from fractions import Fraction as Fr

import numpy as np
import pytest

import lwg_cards as CA
import lwg_chain as CH
import mini_gate as MG

SCORE = [-1, 0, 1]   # n − 1
H = 2


@pytest.fixture(scope="module")
def mini():
    M0, R0 = MG.R_matrix(1)
    M1, R1 = MG.R_matrix(2)
    return M0, R0, M1, R1


# ----------------------------------------------------------------------------------------------------------------- H1: Übergänge
@pytest.mark.parametrize("lam", [1, 2])
def test_h1_zeilen_von_m_und_r_als_formeln_in_a_und_b(mini, lam):
    """Bei c = 1 hat jeder Abgang die Rate 1: a = λ/(λ + 1), b = 1 − a. M[1] = (b, a); M[2] = (b², ab, a); M[3] = (b³, ab², ab, a); R[0] = (b, a, 0), R[1] = R[2] = (b², ab, a)."""
    M0, R0, M1, R1 = mini
    M, R = (M0, R0) if lam == 1 else (M1, R1)
    a = Fr(lam, lam + 1)
    b = 1 - a
    assert M[1][1] == a and M[1][0] == b
    assert (M[2][2], M[2][1], M[2][0]) == (a, a * b, b * b)
    assert (M[3][3], M[3][2], M[3][1], M[3][0]) == (a, a * b, a * b * b, b ** 3)
    assert R[0] == [b, a, 0]
    assert R[1] == [b * b, a * b, a] and R[2] == R[1]
    assert all(sum(row) == 1 for row in M) and all(sum(row) == 1 for row in R)


def test_h1_r_matrizen_als_brueche(mini):
    _, R0, _, R1 = mini
    assert R0 == [[Fr(1, 2), Fr(1, 2), 0], [Fr(1, 4), Fr(1, 4), Fr(1, 2)], [Fr(1, 4), Fr(1, 4), Fr(1, 2)]]
    assert R1 == [[Fr(1, 3), Fr(2, 3), 0], [Fr(1, 9), Fr(2, 9), Fr(2, 3)], [Fr(1, 9), Fr(2, 9), Fr(2, 3)]]


@pytest.mark.parametrize("lam", [1, 2])
def test_h1_code_der_demo_gleich_den_brueche(mini, lam):
    _, R0, _, R1 = mini
    R = R0 if lam == 1 else R1
    mine = CH.acc_chain(lam, c=1, K=3)
    assert np.allclose(mine, np.array([[float(x) for x in row] for row in R]), rtol=0, atol=1e-14)


# ----------------------------------------------------------------------------------------------------------------- H2: Randverteilung
def test_h2_randverteilungen(mini):
    _, R0, _, R1 = mini
    assert MG.stationary(R0) == [Fr(1, 3)] * 3
    assert MG.stationary(R1) == [Fr(1, 7), Fr(2, 7), Fr(4, 7)]
    for lam, R in ((1, R0), (2, R1)):
        assert np.allclose(CH.stat_dist(CH.acc_chain(lam, c=1, K=3)), [float(x) for x in MG.stationary(R)], atol=1e-14)


@pytest.mark.parametrize("lam, verlust", [(1, Fr(1, 4)), (2, Fr(8, 15))])
def test_h2_geburts_sterbe_formel_als_gegenprobe(mini, lam, verlust):
    """Anderer Weg: P(n) ∝ ρⁿ (n = 0 … 3), bedingt auf Annahme (n < 3) ist die Verteilung der gesehenen Zahl; Verlustanteil P(3)."""
    _, R0, _, R1 = mini
    R = R0 if lam == 1 else R1
    w = [Fr(lam) ** n for n in range(4)]
    acc = [x / sum(w[:3]) for x in w[:3]]
    assert acc == MG.stationary(R)
    assert Fr(w[3], sum(w)) == verlust


def test_h2_mittel_und_varianz_und_wartezeit_ueber_little(mini):
    _, R0, _, R1 = mini
    pi0, pi1 = MG.stationary(R0), MG.stationary(R1)
    mean0 = sum(n * pi0[n] for n in range(3))
    mean1 = sum(n * pi1[n] for n in range(3))
    assert mean0 == 1 and mean1 == Fr(10, 7)
    assert sum(n * n * pi0[n] for n in range(3)) - mean0 ** 2 == Fr(2, 3)
    # Wartezeit = Mittel n (c = 1); anderer Weg: Little, L_q / λ_angenommen mit der zeitstationären Verteilung p_j ∝ ρʲ
    for lam, mn in ((1, mean0), (2, mean1)):
        pj = [Fr(lam) ** j / sum(Fr(lam) ** i for i in range(4)) for j in range(4)]
        lq = sum((j - 1) * pj[j] for j in range(2, 4))
        assert lq / (lam * (1 - pj[3])) == mn
    # die beiden Brüche der Handrechnung: λ = 1: (3/4)/(3/4) = 1; λ = 2: (4/3)/(14/15) = 10/7
    assert Fr(3, 4) / Fr(3, 4) == 1 and Fr(4, 3) / Fr(14, 15) == Fr(10, 7)


# ----------------------------------------------------------------------------------------------------------------- H3: ARL von Hand
def test_h3_arl_von_hand(mini):
    """Zeilen 1 und 2 von R sind gleich: n ∈ {1, 2} zu B gelumpt, n = 0 ist A, C ∈ {0, 1}. λ = 1: a₀ = 1 + a₀/2 + b₀/2, b₀ = 1 + a₀/4 + b₀/4 + b₁/2, b₁ = 1 + a₀/4 + b₁/4."""
    _, R0, _, R1 = mini
    pi0 = MG.stationary(R0)
    arl0, L0 = MG.chain_arl(R0, SCORE, H, pi0)
    idx = lambda n, C: n * H + C  # noqa: E731
    a0, b0, b1 = Fr(19, 2), Fr(15, 2), Fr(9, 2)
    assert a0 == 1 + a0 / 2 + b0 / 2 and b0 == 1 + a0 / 4 + b0 / 4 + b1 / 2 and b1 == 1 + a0 / 4 + b1 / 4      # die Handlösung erfüllt die drei Gleichungen
    assert L0[idx(0, 0)] == a0 and L0[idx(1, 0)] == b0 and L0[idx(2, 0)] == b0 and L0[idx(1, 1)] == b1
    assert arl0 == Fr(1, 3) * a0 + Fr(2, 3) * b0 == Fr(49, 6)
    arl1, L1 = MG.chain_arl(R1, SCORE, H, pi0)
    a0n, b0n = Fr(127, 24), Fr(91, 24)
    assert L1[idx(0, 0)] == a0n and L1[idx(1, 0)] == b0n
    assert arl1 == (a0n + 2 * b0n) / 3 == Fr(103, 24)


def test_h3_unabhaengig_und_die_zwei_wirkungen(mini):
    """Dieselbe Karte mit unabhängigen n aus der Randverteilung: ARL₀ = 9, ARL₁ = 63/16. Abhängigkeit: ARL₀ sinkt UND ARL₁ steigt."""
    _, R0, _, R1 = mini
    pi0, pi1 = MG.stationary(R0), MG.stationary(R1)
    a0_iid, _ = MG.chain_arl([list(pi0)] * 3, SCORE, H, pi0)
    a1_iid, _ = MG.chain_arl([list(pi1)] * 3, SCORE, H, pi0)
    assert a0_iid == 9 and a1_iid == Fr(63, 16)
    assert Fr(49, 6) < a0_iid and Fr(103, 24) > a1_iid


def test_h3_code_der_demo_gleich_den_brueche(mini):
    """Der numpy-Code der Demo (lwg_cards) für c = 1, K = 3, Score q = 1, κ = 1 gibt 49/6, 103/24, 9 und 63/16."""
    R0, R1 = CH.acc_chain(1, c=1, K=3), CH.acc_chain(2, c=1, K=3)
    pi0 = CH.stat_dist(R0)
    pi1 = CH.stat_dist(R1)
    S = CA.raw_score(1, 1, K=3)
    e = CA.exact_card(R0, R1, S, H, pi0)
    assert e["arl0"] == pytest.approx(49 / 6, rel=1e-12) and e["arl1"] == pytest.approx(103 / 24, rel=1e-12)
    ei = CA.exact_card(np.tile(pi0, (3, 1)), np.tile(pi1, (3, 1)), S, H, pi0)
    assert ei["arl0"] == pytest.approx(9.0, rel=1e-12) and ei["arl1"] == pytest.approx(63 / 16, rel=1e-12)


# ----------------------------------------------------------------------------------------------------------------- H4: Lauflängenverteilung
def test_h4_lauflaengenverteilung(mini):
    _, R0, _, R1 = mini
    pi0 = MG.stationary(R0)
    d0 = MG.rl_distribution(R0, SCORE, H, pi0, 400)
    d0iid = MG.rl_distribution([list(pi0)] * 3, SCORE, H, pi0, 400)
    assert d0[0] == 0.0 and d0iid[0] == 0.0       # RL = 1 ist unmöglich: ein Schritt erreicht höchstens C = 1
    # P(RL = 2) = Σₙ π(n)·R[n, 2]·R[2, 2] = (1/3)(0 + 1/2 + 1/2)(1/2) = 1/6; bei unabhängigen n (1/3)(1/3)... = 1/9
    assert d0[1] == pytest.approx(1 / 6, abs=1e-15) and d0iid[1] == pytest.approx(1 / 9, abs=1e-15)
    assert sum(d0) == pytest.approx(1, abs=1e-12)
    assert sum(t * p for t, p in zip(range(1, 401), d0)) == pytest.approx(49 / 6, abs=1e-9)
    d1 = MG.rl_distribution(R1, SCORE, H, pi0, 400)
    assert sum(t * p for t, p in zip(range(1, 401), d1)) == pytest.approx(103 / 24, abs=1e-9)
    assert sum(d0[:5]) == pytest.approx(0.4661, abs=5e-5) and sum(d0iid[:5]) == pytest.approx(0.4074, abs=5e-5) and sum(d1[:5]) == pytest.approx(0.7665, abs=5e-5)


def test_h4_code_der_demo_ueberlebenskurve(mini):
    R0 = CH.acc_chain(1, c=1, K=3)
    pi0 = CH.stat_dist(R0)
    Q = CA.build_Q(R0, CA.raw_score(1, 1, K=3), H)
    s = CA.surv_curve(Q, CA.start_vec(pi0, H), 20)
    assert s[0] == pytest.approx(1) and s[1] == pytest.approx(1)                 # nach einem Lkw kann die Karte noch nicht angeschlagen haben
    assert 1 - s[2] == pytest.approx(1 / 6, abs=1e-15)                           # P(RL ≤ 2) = P(RL = 2) = 1/6


# ----------------------------------------------------------------------------------------------------------------- H5: Abhängigkeit
def test_h5_autokorrelation_und_blockvarianz(mini):
    _, R0, _, _ = mini
    pi0 = MG.stationary(R0)
    mu = sum(n * pi0[n] for n in range(3))
    var0 = sum(n * n * pi0[n] for n in range(3)) - mu ** 2

    def cov(lag):
        P = [row[:] for row in R0]
        for _ in range(lag - 1):
            P = [[sum(P[i][k] * R0[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        return sum(pi0[n] * P[n][m] * n * m for n in range(3) for m in range(3)) - mu * mu
    c1, c2 = cov(1), cov(2)
    assert c1 == Fr(1, 4) and c1 / var0 == Fr(3, 8)
    assert c2 == Fr(1, 16) and c2 / var0 == Fr(3, 32)                  # je Lag der Faktor 1/4 (zweiter Eigenwert der Kette)
    blk = (2 * var0 + 2 * c1) / 4
    assert blk == Fr(11, 24) and blk / (var0 / 2) == Fr(11, 8)


# ----------------------------------------------------------------------------------------------------------------- H6: Information
def test_h6_information_je_lkw(mini):
    _, R0, _, R1 = mini
    pi0, pi1 = MG.stationary(R0), MG.stationary(R1)
    kl_rate = sum(float(pi1[n] * R1[n][m]) * math.log(float(R1[n][m] / R0[n][m])) for n in range(3) for m in range(3) if R1[n][m] > 0)
    kl_marg = sum(float(pi1[n]) * math.log(float(pi1[n] / pi0[n])) for n in range(3))
    geschlossen = (float(pi1[0]) * (math.log(2 / 3) / 3 + 2 * math.log(4 / 3) / 3)
                   + float(pi1[1] + pi1[2]) * (math.log(4 / 9) / 9 + 2 * math.log(8 / 9) / 9 + 2 * math.log(4 / 3) / 3))
    assert kl_rate == pytest.approx(geschlossen, abs=1e-12)
    assert kl_rate == pytest.approx(0.07281, abs=5e-6) and kl_marg == pytest.approx(0.14291, abs=5e-6) and kl_rate / kl_marg == pytest.approx(0.51, abs=5e-3)


def test_h6_code_der_demo_information():
    import lwg_evaluation as EV  # noqa: F401  (information() ist auf das große Gate festgelegt; hier die Formel mit den Matrizen der Demo)
    R0, R1 = CH.acc_chain(1, c=1, K=3), CH.acc_chain(2, c=1, K=3)
    pi1 = CH.stat_dist(R1)
    llr = CA.lr_score(R1, R0, 1)[1]
    assert float((pi1[:, None] * R1 * llr).sum()) == pytest.approx(0.07281, abs=5e-6)


# ----------------------------------------------------------------------------------------------------------------- H7: unabhängiges Orakel
def test_h7_ereignissimulation_des_gates(mini):
    """Ereignisdiskrete Simulation Lkw für Lkw (stdlib random, kein Code der Demo): ARL₀ und ARL₁ im Vier-Standardfehler-Band um die Brüche 49/6 und 103/24."""
    rng = random.Random(8080)
    for exakt, lam_b in ((49 / 6, 1), (103 / 24, 2)):
        v = MG.simulate(rng, 1, lam_b, 40, 20000)
        mean = sum(v) / len(v)
        se = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / math.sqrt(len(v))
        assert abs(mean - exakt) < 4 * se, (mean, se, exakt)
