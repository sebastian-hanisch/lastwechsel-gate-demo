"""Gate-Simulator und Karten auf der Wartezeit: Simulator gegen die exakte Kette (Bänder), Karten an handgesetzten Folgen, Live-Probe."""

import math

import numpy as np
import pytest
from scipy.special import gammainc

import lwg_cards as CA
import lwg_chain as CH
import lwg_normal as N
import lwg_sim as S

G0, G1 = CH.Gate(1.6), CH.Gate(2.0)


# ----------------------------------------------------------------------------------------------------------------- Simulator gegen Kette
@pytest.fixture(scope="module")
def lauf():
    """1 500 Gates, 400 Lkw Einschwingen, danach 600 Lkw: gesehene Zahl und Wartezeit je Schritt."""
    g = S.GateSim(N.Stream(4242), 1500)
    for _ in range(400):
        g.step(1.6, 1.0)
    ns, ws = [], []
    for _ in range(600):
        n, w = g.step(1.6, 1.0)
        ns.append(n)
        ws.append(w)
    return np.array(ns), np.array(ws)


def test_simulator_randverteilung_der_gesehenen_zahl_gleich_der_kette(lauf):
    ns, _ = lauf
    freq = np.bincount(ns.ravel(), minlength=6) / ns.size
    assert np.abs(freq - G0.pi).max() < 0.005


def test_simulator_uebergangshaeufigkeiten_gleich_r(lauf):
    ns, _ = lauf
    trans = np.zeros((6, 6))
    np.add.at(trans, (ns[:-1].ravel(), ns[1:].ravel()), 1)
    rows = trans.sum(axis=1)
    assert (rows > 5000).all()
    assert np.abs(trans / rows[:, None] - G0.R).max() < 0.01


def test_simulator_wartezeit_mittel_sd_und_atom_bei_null(lauf):
    _, ws = lauf
    assert abs(ws.mean() - G0.mean_w) < 4 * ws.std() / math.sqrt(ws.size / 10.0)     # Folge abhängig: Faktor 10 als grobe effektive Stichprobe
    assert abs(ws.std() - G0.sd_w) < 0.02
    assert abs((ws == 0).mean() - G0.p0) < 0.006


def test_simulator_autokorrelation_der_wartezeit(lauf):
    _, ws = lauf
    acf = S.acf_lags(ws.T, (1, 2, 3, 4, 5, 10))
    for lag, soll in zip((1, 2, 3, 4, 5), (0.781, 0.586, 0.412, 0.255, 0.178)):
        assert abs(acf[lag] - soll) < 0.02, (lag, acf[lag])
    assert abs(acf[10]) < 0.06


def test_wartezeit_gegeben_n_ist_wirklich_erlang_und_pit_ist_gleichverteilt(lauf):
    """Gegeben n ≥ 2 ist W Erlang(n − 1, 2): die Wahrscheinlichkeits-Integraltransformation U = F(W | n) ist gleichverteilt (Mittel 1/2, Varianz 1/12), für JEDE Ankunftsrate."""
    for lam in (1.6, 2.0, 3.0):
        g = S.GateSim(N.Stream(77), 1000)
        for _ in range(300):
            g.step(lam, 1.0)
        us = []
        for _ in range(300):
            n, w = g.step(lam, 1.0)
            big = n >= 2
            us.append(gammainc((n[big] - 1).astype(float), 2.0 * w[big]))
        u = np.concatenate(us)
        assert abs(u.mean() - 0.5) < 4 * math.sqrt(1 / 12 / u.size * 8), lam
        assert abs(u.var() - 1 / 12) < 0.004, lam


# ----------------------------------------------------------------------------------------------------------------- Karten
class Spion:
    """Zählt die Aufrufe (nur für run_pass)."""
    needs_pre = False

    def __init__(self):
        self.t = []

    def update(self, t, n, w):
        self.t.append(t)


def test_run_pass_gibt_den_karten_genau_t_schritte_von_1_bis_t():
    sp = Spion()
    S.run_pass(1, 5, 10, 25, 1.6, 2.0, 1.0, 1.0, [sp])
    assert sp.t == list(range(1, 26))


def test_grid_haelt_die_erste_durchgangszeit_je_schwelle():
    g = S.Grid(2, [1.0, 2.0, 3.0])
    g.update(1, np.array([0.5, 2.5]))     # Lauf 1 erreicht keine Schwelle, Lauf 2 die ersten beiden
    g.update(2, np.array([1.5, 0.0]))
    g.update(3, np.array([3.5, 9.0]))
    assert g.rl.tolist() == [[2, 3, 3], [1, 1, 3]]


def test_rohkarte_rechnet_standardisiert_und_nach_unten_gekappt():
    card = S.RawCard(1, [0.5, 1.0], 0.25, 1.0, 2.0, "x")
    for t, w in enumerate([3.0, 3.0, 0.0, 0.0], start=1):     # z = 1, 1, −0,5, −0,5
        card.update(t, np.array([0]), np.array([w]))
    # C: 0,75 → 1,5 → 0,75 → 0 (nach unten bei 0 gekappt)
    assert card.C[0] == 0.0
    assert card.grid.rl.tolist() == [[1, 2]]        # 0,75 ≥ 0,5 bei t = 1; 1,5 ≥ 1,0 bei t = 2


def test_ausduennen_nimmt_nur_jeden_b_ten_lkw():
    card = S.ThinCard(1, [0.1], 0.0, 0.0, 1.0, 3, "x")
    for t in range(1, 8):
        card.update(t, np.array([0]), np.array([1.0 if t == 3 else 0.0]))
    assert card.grid.rl.tolist() == [[3]]
    card2 = S.ThinCard(1, [0.1], 0.0, 0.0, 1.0, 3, "x")
    for t in range(1, 8):
        card2.update(t, np.array([0]), np.array([1.0 if t == 4 else 0.0]))     # t = 4 wird ausgelassen
    assert card2.grid.rl.tolist() == [[0]]


def test_blockkarte_schliesst_bloecke_bei_t_plus_phase_durch_b():
    # b = 4, Phase 0: Blöcke schließen bei t = 4, 8; die Karte sieht das Mittel von 4 Werten
    card = S.BlockCard(1, [0.9, 2.5], 0.0, 0.0, 1.0, 4, "x", phase=0)
    card.start([])
    for t in range(1, 9):
        card.update(t, np.array([0]), np.array([1.0 if t <= 4 else 3.0]))      # Blockmittel 1, dann 3
    assert card.grid.rl.tolist() == [[4, 8]]
    # Phase 2: die letzten zwei Lkw vor ν (Wert 5 und 5) gehören in den ersten Block, der nach t = 2 schließt
    card2 = S.BlockCard(1, [2.0, 3.5], 0.0, 0.0, 1.0, 4, "x", phase=2)
    card2.start([np.array([9.0]), np.array([5.0]), np.array([5.0])])
    for t in range(1, 7):
        card2.update(t, np.array([0]), np.array([1.0]))
    # erster Block (5 + 5 + 1 + 1)/4 = 3 bei t = 2 (≥ 2,0, aber < 3,5): C = 3 − 0 = 3; zweiter (4 · 1)/4 = 1 bei t = 6: C = 4 ≥ 3,5
    assert card2.grid.rl.tolist() == [[2, 6]]


def test_naiver_zweiseitiger_cusum_schlaegt_erst_ueber_h_an_und_zaehlt_beide_richtungen():
    card = S.TwoSidedFixed(2, 0.0, 1.0, k=0.5, h=2.0)
    for t in range(1, 6):
        card.update(t, np.array([0, 0]), np.array([1.5, -1.5]))   # z = ±1,5: C wächst um 1 je Lkw, Alarm bei C > 2 (t = 3)
    assert card.rl.tolist() == [3, 3]
    equal = S.TwoSidedFixed(1, 0.0, 1.0, k=0.0, h=2.0)
    for t in range(1, 3):
        equal.update(t, np.array([0]), np.array([1.0]))
    assert equal.rl.tolist() == [0]                               # C = 2 genau: kein Alarm (strikt größer)


def test_karte_auf_n_simuliert_gleich_der_exakten_kette():
    """Simulation der Karte auf n (H = 60) gegen die exakte Kette: ARL₀ 28,83 und ARL₁ 17,33 im Vier-Standardfehler-Band."""
    for lam, key in ((1.6, "arl0"), (2.0, "arl1")):
        card = S.NCard(3000, [60], 8, 20, "n")
        S.run_pass(99, 3000, 300, 400, 1.6, lam, 1.0, 1.0, [card])
        a, se, cens = S.arl_from_grid(card.grid, 400)
        exakt = CA.exact_card(G0.R, G1.R if key == "arl1" else G0.R, CA.raw_score(8, 20), 60, G0.pi)[key]
        assert cens[0] < 0.001
        assert abs(a[0] - exakt) < 4 * se[0] + 0.3, (key, a[0], se[0], exakt)    # +0,3: Start nach 300 Lkw statt stationär


def test_residuenkarte_bleibt_unter_aenderung_der_rate_nahe_null_drift():
    """U ist für jede Rate gleichverteilt: mittleres z ≈ 0 vor und nach dem Wechsel (der Score der Karte ist z − 0,25)."""
    for lam in (1.6, 2.4):
        card = S.ResidCard(600, [50.0], 0.25, N.Stream(5), "r")
        S.run_pass(21, 600, 300, 300, 1.6, lam, 1.0, 1.0, [card])
        # C wächst im Mittel nur durch die Veränderung der Abhängigkeit der z-Folge: nach 300 Lkw ist die Karte weit unter der Schwelle 50
        assert card.C.mean() < 25


def test_card_at_target_interpoliert_und_nennt_unschaerfe_der_schwelle():
    class FakeCard:
        def __init__(self, rl):
            self.grid = S.Grid(len(rl), [1.0, 2.0])
            self.grid.rl = np.array(rl, dtype=np.int32)
    # ARL₀ an den zwei Schwellen: 100 und 1000 → Ziel 370 liegt dazwischen
    pre = FakeCard([[50, 500], [150, 1500]])            # Mittel 100 und 1000
    post = FakeCard([[10, 100], [30, 300]])             # Mittel 20 und 200
    r = S.card_at_target(pre, post, 5000, 5000, target=316.22776601683796)
    assert r["h"] == pytest.approx(1.5) and r["arl1"] == pytest.approx(110.0)
    assert S.card_at_target(pre, post, 5000, 5000, target=5000.0) is None


# ----------------------------------------------------------------------------------------------------------------- Live-Probe
def test_live_probe_ist_reproduzierbar_und_zeigt_den_zusammenbruch():
    a = S.live_probe(100, 1000, 8080, G0.mean_w, G0.sd_w)
    b = S.live_probe(100, 1000, 8080, G0.mean_w, G0.sd_w)
    c = S.live_probe(100, 1000, 8081, G0.mean_w, G0.sd_w)
    assert a == b and a["arl0"] != c["arl0"]
    assert a["R"] == 100 and a["T"] == 1000
    assert a["arl0"] < 80 and a["p100"] > 0.7                  # weit unter 368,6; die Zahl 42 steht in README und Tests der Messreihe
    assert set(a["acf"]) == {1, 2, 3, 4, 5} and set(a["blocks"]) == {5, 10, 20, 40}
    assert a["acf"][1] > a["acf"][2] > a["acf"][5] > 0


def test_blockstatistik_an_handfolgen():
    W = np.array([[1.0, 3.0, 1.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
    bs = S.block_stats(W, 1.0, blocks=(2,))
    # Blockmittel: (2, 2) und (2, 2): alle gleich: sd 0; Faktor 0
    assert bs[2]["sd"] == 0.0 and bs[2]["faktor"] == 0.0 and bs[2]["lag1"] == 0.0
    W2 = np.array([[0.0, 2.0, 4.0, 6.0]])
    bs2 = S.block_stats(W2, 2.0, blocks=(2,))                    # Blockmittel 1 und 5: sd 2, Faktor 2/(2/√2) = √2, Lag-1 = −1
    assert bs2[2]["sd"] == pytest.approx(2.0) and bs2[2]["faktor"] == pytest.approx(math.sqrt(2)) and bs2[2]["lag1"] == pytest.approx(-1.0)
