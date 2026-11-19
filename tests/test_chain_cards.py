"""Eigenschaften der Kette der gesehenen Zahl und der Karten (feste Zufallsinstanzen, exakte Rechnung)."""

import numpy as np
import pytest
import scipy.sparse.linalg as spl

import lwg_cards as CA
import lwg_chain as CH
import lwg_constants as C

G0, G1 = CH.Gate(1.6), CH.Gate(2.0)


# ----------------------------------------------------------------------------------------------------------------- Kette
@pytest.mark.parametrize("c, K, lam", [(2, 6, 1.6), (1, 3, 1.0), (3, 8, 2.5), (2, 2, 1.0), (1, 1, 1.0 / 3)])
def test_r_ist_stochastisch_und_pi_stationaer(c, K, lam):
    R = CH.acc_chain(lam, c, K)
    assert (R >= -1e-15).all() and np.allclose(R.sum(axis=1), 1, atol=1e-13)
    pi = CH.stat_dist(R)
    assert pi.min() > 0 and pi.sum() == pytest.approx(1, abs=1e-13) and np.allclose(pi @ R, pi, atol=1e-13)


def test_von_n_aus_sind_nur_zahlen_bis_n_plus_1_erreichbar_und_alle_darunter_haben_positive_wahrscheinlichkeit():
    R = G0.R
    for n in range(6):
        assert (R[n, n + 2:] == 0).all() and (R[n, :n + 2] > 0).all()


def test_mittlere_gesehene_zahl_waechst_mit_der_rate():
    means = [CH.Gate(lam).mean_n for lam in (1.0, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0)]
    assert means == sorted(means) and len(set(np.round(means, 9))) == len(means)


def test_momente_der_gesehenen_zahl_und_der_wartezeit():
    assert G0.mean_n == pytest.approx(2.1612, abs=5e-5) and G0.sd_n == pytest.approx(1.5655, abs=5e-5)
    assert G1.mean_n == pytest.approx(30 / 11, rel=1e-12)                    # Rate 2,0: π = (1/11, 2/11, ...), Mittel 2/11·(1+2+3+4+5) = 30/11
    assert G1.pi == pytest.approx([1 / 11] + [2 / 11] * 5, abs=1e-13)
    assert G1.mean_w == pytest.approx(10 / 11, rel=1e-12)                    # 2/11·(1+2+3+4)/2
    assert G0.mean_w == pytest.approx(0.6590, abs=5e-5) and G0.sd_w == pytest.approx(0.8977, abs=5e-5) and G0.p0 == pytest.approx(0.4076, abs=5e-5)


def test_wartezeit_gegeben_n_hat_den_mittelwert_der_formel():
    assert CH.wait_mean_given_n().tolist() == [0, 0, 0.5, 1.0, 1.5, 2.0]
    assert CH.wait_mean_given_n(1, 3, 1.0).tolist() == [0, 1.0, 2.0]       # eine Spur: n Lkw im Gate, n Abfertigungen vor dem eigenen
    assert G0.mean_w == pytest.approx(float(G0.pi @ CH.wait_mean_given_n()), rel=1e-13)


# ----------------------------------------------------------------------------------------------------------------- Karten
def test_q_ist_substochastisch_und_die_fehlende_masse_ist_die_alarmwahrscheinlichkeit():
    S = CA.raw_score(8, 20)
    H = 40
    Q = CA.build_Q(G0.R, S, H)
    miss = 1 - np.asarray(Q.sum(axis=1)).ravel()
    assert (miss >= -1e-14).all() and miss.max() < 1 and miss.max() > 0
    # Alarmwahrscheinlichkeit aus (n, C): Summe von R[n, m] über die Übergänge mit C + s(n, m) ≥ H
    for n in (0, 3, 5):
        for Cs in (0, 17, 39):
            erwartet = sum(G0.R[n, m] for m in range(6) if max(0, Cs + int(S[n, m])) >= H)
            assert miss[n * H + Cs] == pytest.approx(erwartet, abs=1e-14)


@pytest.mark.parametrize("kappa", C.KAPPA_OPTIONS)
def test_arl0_waechst_mit_der_schwelle(kappa):
    S = CA.raw_score(8, kappa)
    vals = [CA.exact_card(G0.R, G0.R, S, H, G0.pi)["arl0"] for H in range(2, 61, 2)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])) and vals[0] >= 1


def test_arl1_kleiner_als_arl0_und_faellt_mit_wachsendem_lam1():
    S = CA.raw_score(8, 20)
    arl1 = [CA.exact_card(G0.R, CH.Gate(lam).R, S, 100, G0.pi)["arl1"] for lam in (1.7, 1.8, 2.0, 2.4, 3.0, 4.0)]
    arl0 = CA.exact_card(G0.R, G0.R, S, 100, G0.pi)["arl0"]
    assert all(a < arl0 for a in arl1) and arl1 == sorted(arl1, reverse=True)


def test_arl0_faellt_mit_wachsender_wahrer_rate_ohne_aenderung():
    S = CA.raw_score(8, 20)
    vals = []
    for lam in (1.4, 1.5, 1.6, 1.7, 1.8):
        g = CH.Gate(lam)
        vals.append(CA.exact_card(g.R, g.R, S, 308, g.pi)["arl0"])
    assert vals == sorted(vals, reverse=True)


@pytest.mark.parametrize("kappa, H", [(20, 150), (20, 308), (24, 120), (28, 80)])
def test_zero_state_langsamer_als_steady_state_an_derselben_schwelle(kappa, H):
    S = CA.raw_score(8, kappa)
    e = CA.exact_card(G0.R, G1.R, S, H, G0.pi, steady=True)
    assert e["arl1"] > e["arl1_ss"] and e["arl0"] > e["arl0_ss"]           # Zero-state startet bei C = 0: länger bis zum Alarm, auch ohne Änderung


def test_skalierung_scores_und_schwelle_durch_zwei_aendert_nichts():
    """Alle Scores (q = 8, κ = 20) sind Vielfache von 4: Scores und H durch 2 geteilt gibt dieselbe Kette."""
    a = CA.exact_card(G0.R, G1.R, CA.raw_score(8, 20), 100, G0.pi)
    b = CA.exact_card(G0.R, G1.R, CA.raw_score(4, 10), 50, G0.pi)
    assert a["arl0"] == pytest.approx(b["arl0"], rel=1e-12) and a["arl1"] == pytest.approx(b["arl1"], rel=1e-12)


def test_ohne_aenderung_ist_arl1_gleich_arl0():
    e = CA.exact_card(G0.R, G0.R, CA.raw_score(8, 20), 80, G0.pi)
    assert e["arl1"] == e["arl0"]


def test_ueberlebenskurve_faellt_und_summe_ist_arl0():
    e = CA.exact_card(G0.R, G1.R, CA.raw_score(8, 20), 60, G0.pi, tmax=3000)
    for s in (e["surv0"], e["surv1"]):
        assert s[0] == pytest.approx(1) and (np.diff(s) <= 1e-15).all()
    assert e["surv0"].sum() == pytest.approx(e["arl0"], rel=1e-6)
    assert e["surv1"].sum() == pytest.approx(e["arl1"], rel=1e-6)


def test_quasistationaere_verteilung_gleich_dem_eigenvektor():
    """Zweiter Weg: linker Eigenvektor von Q zum größten Eigenwert (scipy eigs) gegen die Potenzmethode der Demo."""
    S = CA.raw_score(8, 20)
    Q = CA.build_Q(G0.R, S, 60)
    qs = CA.quasi_stat(Q)
    w, v = spl.eigs(Q.T.tocsc(), k=1, which="LM")
    vec = np.real(v[:, 0])
    vec = vec / vec.sum()
    assert qs.sum() == pytest.approx(1) and np.abs(qs - vec).max() < 1e-8
    assert np.real(w[0]) < 1


def test_schutz_vor_unbrauchbaren_loesungen():
    """κ = 28: ab H ≈ 400 liegt ARL₀ der Kette über 2·10⁵; der Löser liefert dort Unsinn (bei der unabhängigen Rechnung negative Zahlen). Der Schutz meldet es, und was er durchlässt, ist plausibel."""
    S = CA.raw_score(8, 28)
    Riid = np.tile(G0.pi, (6, 1))
    for R in (G0.R, Riid):
        for H in (100, 200, 300, 400, 500, 700):
            try:
                a0 = CA.exact_card(R, R, S, H, G0.pi, guard=True)["arl0"]
            except CA.ArlTooLarge:
                continue
            assert np.isfinite(a0) and 1 <= a0 <= C.ARL_CAP * 1.5, (H, a0)
    assert CA.exact_card(Riid, Riid, S, 300, G0.pi)["arl0"] < 0            # ohne Schutz: der Löser gibt dort eine negative ARL₀ (so begründet sich der Schutz)
    with pytest.raises(CA.ArlTooLarge):
        CA.exact_card(G0.R, G0.R, S, 700, G0.pi, guard=True)
    with pytest.raises(CA.ArlTooLarge):
        CA.exact_card(Riid, Riid, S, 300, G0.pi, guard=True)


def test_schutz_laesst_alle_werte_unter_der_grenze_durch_und_ist_monoton():
    """Für jeden angebotenen κ und H bis 700: entweder Meldung oder eine endliche Zahl zwischen 1 und 1,5·Grenze; nach der ersten Meldung kommt nie wieder eine Zahl (ARL₀ wächst mit H)."""
    for kappa in C.KAPPA_OPTIONS:
        S = CA.raw_score(8, kappa)
        gemeldet = False
        for H in range(10, 701, 30):
            try:
                a0 = CA.exact_card(G0.R, G0.R, S, H, G0.pi, guard=True)["arl0"]
            except CA.ArlTooLarge:
                gemeldet = True
                continue
            assert not gemeldet, (kappa, H)
            assert np.isfinite(a0) and 1 <= a0 <= C.ARL_CAP * 1.5


def test_kleinste_schwelle_fuer_das_ziel():
    S = CA.raw_score(8, 20)
    H = CA.smallest_h(G0.R, S, G0.pi, 370.0)
    assert H == 309
    assert CA.exact_card(G0.R, G0.R, S, H, G0.pi)["arl0"] >= 370 > CA.exact_card(G0.R, G0.R, S, H - 1, G0.pi)["arl0"]
    assert CA.smallest_h(G0.R, S, G0.pi, 1e12, hi=300) is None


def test_at_target_interpoliert_log_linear_und_meldet_ausserhalb_none():
    Hs, a0, a1 = [10, 20, 30], [10.0, 100.0, 1000.0], [5.0, 50.0, 500.0]
    h, v, t = CA.at_target(Hs, a0, a1, 316.22776601683796)         # 10^2,5: die Mitte zwischen 100 und 1000 in log
    assert t == pytest.approx(0.5) and h == pytest.approx(25.0) and v == pytest.approx(275.0)
    assert CA.at_target(Hs, a0, a1, 5.0) is None and CA.at_target(Hs, a0, a1, 2000.0) is None


def test_at_target_exact_gleich_der_interpolation_ueber_ein_raster():
    S = CA.raw_score(8, 20)
    r = CA.at_target_exact(G0.R, {2.0: G1.R}, S, G0.pi)
    assert r["h"] == pytest.approx(308.2722, abs=5e-5) and r["arl1"][2.0] == pytest.approx(101.1606, abs=5e-5)
    assert r["arl0_lo"] < 370 <= r["arl0_hi"] and r["h_int"] == 309


def test_likelihood_quotient_score_ist_null_wenn_nichts_sich_aendert():
    S, llr = CA.lr_score(G0.R, G0.R, 40)
    assert (S == 0).all() and (llr == 0).all()
    S2, llr2 = CA.lr_score(G1.R, G0.R, 40)
    assert (llr2[G0.R == 0] == 0).all() and (S2 == np.rint(40 * llr2)).all()
