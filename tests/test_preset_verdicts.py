"""Je Schnellstart die erwartete Verdikt-Zeile der App (DEMO-PLAYBOOK §11a Punkt 7): ein Szenario ohne Effekt meldet keinen Effekt, jedes andere nennt seine Zahlen.

Die Texte kommen aus echten AppTest-Läufen (alle fünf Verdikte stehen gleichzeitig auf der Seite)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import lwg_constants as C

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _verdikte(name):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return [(art, w.value) for art in ("success", "info", "warning") for w in getattr(at, art)]


ERWARTET = {
    "Lehrbuch-Schwelle (unabhängig, ARL₀ 408)": [("warning", "ARL₀ = 408,2 Lkw"), ("warning", "sind es 51,2 Lkw (87,2 %), Faktor 8,0"), ("info", "Rohkarte (jeder Lkw, k = Sprung/2)"),
                                           ("warning", "ARL₀ = 370 (zwischen den Rasterschwellen interpoliert), aber nach dem Wechsel 1,6 → 2,0 im Mittel 101,2 Lkw statt 37,2"), ("warning", "Ankunftsrate 1,6 → 2,0: Residuenkarte")],
    "Schwelle angepasst (370 auf der Kette)": [("info", "Schwelle H = 309: Mit der Randverteilung der gesehenen Zahl und unabhängigen Lkw läge ARL₀ über 2·10⁵ Lkw (keine Zahl)"),
                                               ("info", "Auf der tatsächlichen Kette sind es 378,1 Lkw")],
    "Blockmittel b = 20": [("info", "Blockmittel b = 20: 135,4 Lkw (± 1,4), Rohkarte 134,7 (± 1,4) bei ARL₀ = 370 Lkw, Wechsel 1,6 → 2,0: der Unterschied von +0,7 Lkw ist nicht aufgelöst"),
                           ("info", "Liegt die Änderung nicht auf einer Blockgrenze")],
    "Modellkarte für 2,0": [("info", "entworfen für 2,0 (Gitter 1/40), 102,3; für das wahre 2,0 entworfen 102,3"), ("info", "Das Nullmodell stimmt (λ₀ = 1,6)")],
    "Falsches Nullmodell (1,7 statt 1,6)": [("warning", "λ₀ = 1,7 statt 1,6), liegt ARL₀ der Rohkarte bei 244,9 (Faktor 0,65)")],
    "Größerer Sprung (2,4)": [("warning", "nach dem Wechsel 1,6 → 2,4 im Mittel 55,3 Lkw statt 17,2"), ("info", "Wechsel 1,6 → 2,4: 78,9 Lkw (± 0,7; Wartezeit, Simulation)"), ("info", "1,6 → 2,4 55,3 Lkw, die Likelihood-Quotient-Karte"),
                              ("warning", "Ankunftsrate 1,6 → 2,4: Residuenkarte")],
    "Abfertigung langsamer (0,85)": [("success", "Abfertigung 1,0 → 0,85: Residuenkarte (Wartezeit gegeben die gesehene Zahl) 107,8 Lkw (± 1,2), Rohkarte 102,2 (± 1,0)"), ("success", "etwa 5 % langsamer als die Rohkarte (Faktor 1,05)")],
}


def test_jedes_preset_hat_eine_erwartung():
    assert set(ERWARTET) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdikte(name):
    v = _verdikte(name)
    for art, teil in ERWARTET[name]:
        assert any(a == art and teil in t for a, t in v), (name, art, teil, [(a, t[:90]) for a, t in v])


def test_szenarien_ohne_effekt_melden_keinen_effekt():
    """Ohne Änderung des Nullmodells (λ₀′ = 1,6) kein „ARL₀ sinkt“; die Blockkarte bei b = 1 (= Rohkarte) meldet keinen Unterschied; bei H = 30 (Faktor 1,15) kein Zusammenbruch."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.slider(key="h_slider").set_value(30).run(timeout=TIMEOUT)
    texte = [(art, w.value) for art in ("success", "info", "warning") for w in getattr(at, art)]
    assert any(a == "info" and "Faktor 1,2" in t and "nahe beieinander" in t for a, t in texte)
    assert not any("Hier kehrt sich der Faktor um" in t or ("Faktor 1,2" in t and a == "warning") for a, t in texte)
    assert any("Das Nullmodell stimmt (λ₀ = 1,6)" in t for _, t in texte)
    assert not any("Standardfehler der Differenz" in t and "langsamer" in t for _, t in texte)      # Experiment 3 bei b = 1: kein Vergleich Roh gegen Roh
