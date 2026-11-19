"""Literatur- und Quellenwerte: GERÜST ohne Orakel-Anspruch.

Nur die Zahlen der Quellen-Demo und der Vormessung stehen hier, mit Quelle und Stand; für alles Übrige ist „offen“ vermerkt (nicht gelesen oder ohne Zahl, die man nachrechnen könnte).
Diese Datei ist KEIN Orakel (`tests/test_oracle_lastwechsel_gate.py` ist es); sie zählt für `oracle_coverage.py` nicht."""

import pytest

import lwg_chain as CH
import lwg_sim as SIM  # noqa: F401  (Simulator: die Autokorrelation unten kommt aus der gefrorenen Messreihe)
import lwg_reference as REF

# Quelle: markov-queue-demo (README, Stand 2026-11-18, nur gelesen): zwei Spuren, 6 Plätze insgesamt, Last 80 % (Angebot 1,6)
MARKOV_QUEUE = {"verlust": 0.0760, "wartezeit_min": 1.98}
# Quelle: Vormessung V2 (Abschnitt 6, out_ae_v2.txt, Simulation): Autokorrelation der Wartezeit, Lag 1 bis 5
V2_ACF = (0.781, 0.586, 0.412, 0.255, 0.178)

OFFEN = {
    "Vanhatalo/Kulahci 2015": "Aussage der Quelle (angepasste Grenze wirkt bei hoher Autokorrelation schlechter; Residuenkarte nur bei bekanntem Modell) ohne Zahl, die man nachrechnen könnte",
    "Larsen u. a., arXiv 1907.05453": "Aussage der Einleitung (Residuen erkennen die Verschiebung zuerst, dann immer schlechter) ohne Zahl",
    "Zwetsloot/Jones-Farmer/Woodall 2024": "Aussage (Aggregation kostet Auflösung) ohne Zahl",
    "Baron/Malov, arXiv 2502.09185": "Aussage (CUSUM-Rekursion = Lindley-Rekursion) nur als Hinweis",
    "Chen/Zhou 2014, Jana/Dey 2000": "nicht gelesen",
    "Lorden 1971, Moustakides 1986": "nicht gelesen; die Näherung ln γ/I wird nie als Schranke geführt",
}


def test_markov_queue_verlust_und_wartezeit():
    g = CH.Gate(1.6)
    import test_oracle_lastwechsel_gate as O
    _, loss = O.birth_death_seen(1.6)
    assert loss == pytest.approx(MARKOV_QUEUE["verlust"], abs=5e-5)
    assert g.mean_w * 3.0 == pytest.approx(MARKOV_QUEUE["wartezeit_min"], abs=0.005)


def test_v2_autokorrelation_der_wartezeit_als_band_in_beiden_saetzen():
    for st in REF.sim()["saetze"]:
        for lag, soll in zip("12345", V2_ACF):
            assert abs(st["ref"]["acf"][lag] - soll) < 0.02, (lag, st["ref"]["acf"][lag])


def test_alles_uebrige_ist_offen_und_wird_nicht_als_orakel_benutzt():
    assert len(OFFEN) == 6 and all(OFFEN.values())
    assert "Literaturwerte: keine" in open(__file__.replace("test_literatur_lwg.py", "test_oracle_lastwechsel_gate.py"), encoding="utf-8").read()
