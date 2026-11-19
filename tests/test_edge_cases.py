"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 2, Grenzen inklusive, keine Änderung, Änderung nach unten, Karte ohne Alarm, kleinste Schwelle,
Lastextreme, Löser an der Reglergrenze."""

import numpy as np
import pytest

import lwg_cards as CA
import lwg_chain as CH
import lwg_constants as C
import lwg_evaluation as EV
import lwg_reference as REF
import lwg_sim as SIM

G0, G1 = CH.Gate(1.6), CH.Gate(2.0)
S20 = CA.raw_score(8, 20)


def test_gate_mit_genau_zwei_plaetzen_und_einer_spur():
    """K = 2, c = 1: der kleinste Fall mit einer Abweisung; n ∈ {0, 1}, R 2 × 2."""
    R = CH.acc_chain(1.0, c=1, K=2)
    assert R.shape == (2, 2) and np.allclose(R.sum(axis=1), 1) and (R > 0).all()
    pi = CH.stat_dist(R)
    assert pi == pytest.approx([0.5, 0.5])          # ρ = 1: gleich wahrscheinlich, bedingt auf Annahme


def test_gate_mit_einem_platz_sieht_immer_null():
    """K = 1 (nur ein Platz): jeder angenommene Lkw sieht ein leeres Gate; R = [[1]]."""
    R = CH.acc_chain(2.0, c=1, K=1)
    assert R.shape == (1, 1) and R[0, 0] == pytest.approx(1.0) and CH.stat_dist(R) == pytest.approx([1.0])


@pytest.mark.parametrize("lam", [0.01, 0.1, 10.0, 50.0])
def test_extreme_lasten(lam):
    g = CH.Gate(lam)
    assert np.isfinite(g.R).all() and np.allclose(g.R.sum(axis=1), 1, atol=1e-12) and g.pi.sum() == pytest.approx(1, abs=1e-12)
    assert 0 <= g.mean_n <= 5
    if lam >= 10:
        assert g.pi[5] > (0.75 if lam == 10.0 else 0.9)   # meist voll: der angenommene Lkw sieht meist 5 (λ = 10: 0,80)
    if lam <= 0.1:
        assert g.pi[0] > 0.9                         # fast immer leer


def test_keine_aenderung_gibt_arl1_gleich_arl0_und_aenderung_nach_unten_gibt_arl1_groesser():
    e = CA.exact_card(G0.R, G0.R, S20, 150, G0.pi)
    assert e["arl1"] == e["arl0"]
    low = CH.Gate(1.2)
    d = CA.exact_card(G0.R, low.R, S20, 150, G0.pi)
    assert d["arl1"] > d["arl0"] > 1                # Lastabfall: die einseitige Karte (nach oben) schlägt noch seltener an


def test_kleinste_schwelle_alarm_beim_ersten_lkw():
    """Alle Scores ≥ H = 1 (κ = −100): Alarm schon beim ersten Lkw, ARL = 1 exakt, Überlebenskurve fällt von 1 auf 0."""
    S = CA.raw_score(8, -100)
    e = CA.exact_card(G0.R, G1.R, S, 1, G0.pi, tmax=5)
    assert e["arl0"] == pytest.approx(1.0) and e["arl1"] == pytest.approx(1.0)
    assert e["surv0"][0] == pytest.approx(1) and e["surv0"][1] == pytest.approx(0, abs=1e-15)


def test_karte_ohne_moeglichen_alarm_meldet_statt_zu_rechnen():
    """Kein Score ist positiv (κ = 100) oder alle Scores sind 0 (q_s sehr klein): die Karte schlägt nie an; mit Schutz kommt die Meldung, nie eine Zahl."""
    for S in (CA.raw_score(8, 100), np.zeros((6, 6), dtype=np.int64), CA.lr_score(G1.R, G0.R, 1)[0]):
        with pytest.raises(CA.ArlTooLarge):
            CA.exact_card(G0.R, G1.R, S, 50, G0.pi, guard=True)
    S, _ = CA.lr_score(G1.R, G0.R, 1)
    assert not (S > 0).any() or S.max() < 50


def test_schutz_laesst_die_voreinstellungen_der_app_durch():
    for h, kappa in ((97, 20), (309, 20), (700, 20), (10, 28), (50, 18)):
        r = EV.exp1(h, kappa)
        assert r["arl0_kette"] is not None and r["arl0_kette"] >= 1


def test_ziele_am_rand_der_kennlinien():
    """Ziel 50 und 2000 für jedes κ und jedes λ₁: entweder Zahlen oder None (außerhalb), nie ein Absturz; Zahlen sind positiv und die Kette ist langsamer als die unabhängige Rechnung."""
    for kappa in C.KAPPA_OPTIONS:
        for target in (50, 2000):
            for lam1 in (1.7, 4.0):
                r = EV.exp2(kappa, target, lam1)
                if r["iid"] is None or r["kette"] is None:
                    assert EV.verdict_exp2(r)[0] == "warning"
                    continue
                assert 0 < r["iid"][1] and r["kette"][1] > r["iid"][1] and r["kette"][0] > r["iid"][0]


def test_exp3_alle_kombinationen_haben_eine_karte_ausser_ausduennen_40():
    for verfahren in C.VERFAHREN:
        for b in C.BLOCK_OPTIONS:
            for gr in C.GROESSE_OPTIONS:
                r = EV.exp3(verfahren, b, gr)
                if verfahren == C.VERFAHREN[1] and b == 40:
                    assert r["name"] is None and r["sel"] is None
                else:
                    assert r["sel"] is not None and r["roh"] is not None and r["sel"][0] > 0
                kind, text = EV.verdict_exp3(r)
                assert kind in ("info", "warning") and text


def test_exp5_alle_aenderungen():
    for ae in C.AENDERUNGEN:
        r = EV.exp5(ae)
        assert r["roh"] and r["n"] and r["resid"] and r["resid"][0] < 370 and r["roh"][0] < 370      # keine Karte ist blind (alle ARL₁ unter ARL₀ = 370)


def test_gemeinsame_endpunkte_des_exp4():
    """Alle wahren und Entwurf-λ₁ der Regler: kein Absturz, alle ARL₁ endlich und unter 370."""
    for qs in (20, 80):
        for entw in (1.8, 4.0):
            r = EV.exp4(2.0, entw, 1.4, qs)
            assert all(np.isfinite(z["roh"]) and np.isfinite(z["modell"]) and z["roh"] < 370 and z["modell"] < 370 for z in r["rows"])


def test_gefrorene_steady_state_ausserhalb_des_fensters_ist_none():
    assert REF.steady_at_target(20, 20.0) == {"gleich": None, "eq": None}          # ARL₀ = 20 liegt unter dem Fenster der Steady-state-Spalten (40 bis 4 000)
    assert REF.at_target_zero(20, "kette", 2.0, 1e12) is None


def test_live_probe_mit_wenigen_laeufen_und_kurzer_reihe():
    p = SIM.live_probe(5, 60, 3, G0.mean_w, G0.sd_w, burn=20)
    assert p["R"] == 5 and p["T"] == 60 and 0 < p["arl0"] <= 60 and set(p["blocks"]) == {5, 10, 20, 40}
    assert p["blocks"][40]["sd"] >= 0 and np.isnan(p["blocks"][40]["lag1"]) and np.isfinite(p["blocks"][20]["lag1"])   # T = 60: ein Block der Länge 40, drei der Länge 20


def test_zufalls_seed_null_und_maximum_laufen():
    for seed in (0, C.LIVE_SEED_MAX):
        p = SIM.live_probe(10, 100, seed, G0.mean_w, G0.sd_w, burn=10)
        assert np.isfinite(p["arl0"])
