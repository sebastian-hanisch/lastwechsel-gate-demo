"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Erweiterungen dieser Demo (der Anfang der Datei bleibt die Vorlage aus tools/templates/test_app_smoke.py) -----------------------------------
import pytest  # noqa: E402

import lwg_constants as C  # noqa: E402
import lwg_presets as P  # noqa: E402


def _texte(at) -> str:
    return " ".join([w.value for w in at.success] + [w.value for w in at.info] + [w.value for w in at.warning])


def test_seite_hat_fuenf_experimente_und_acht_diagramme():
    at = _fresh_app()
    assert at.title and len([s for s in at.subheader if s.value.startswith("🔬")]) == 5
    assert len(at.get("plotly_chart")) == 8          # je Experiment 1 bis 2 Diagramme, jedes mit eigenem key


def test_standard_zeigt_die_verdikte_der_fuenf_experimente():
    text = _texte(_fresh_app())
    assert "ARL₀ = 408,2 Lkw" in text and "51,2 Lkw" in text              # Experiment 1
    assert "Die angepasste Schwelle stellt das Versprechen wieder her" in text   # Experiment 2
    assert "Bei keinem der" in text and "holt die Kette den Wert" in text
    assert "Rohkarte (jeder Lkw" in text                                     # Experiment 3 (b = 1)
    assert "Likelihood-Quotient-Karte der Übergänge" in text                 # Experiment 4
    assert "Residuenkarte (Wartezeit gegeben die gesehene Zahl)" in text     # Experiment 5


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_laeuft_und_setzt_alle_regler(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        soll = C.PRESETS[name].get(key, P.SETTING_SPECS[state_key].default)
        ist = at.session_state[state_key]
        assert abs(ist - soll) < 1e-9 if isinstance(soll, float) else ist == soll, (name, state_key)


def test_zufalls_button_aendert_den_seed():
    at = _fresh_app()
    alt = at.session_state["live_seed_input"]
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)
    assert not at.exception and at.session_state["live_seed_input"] != alt


def test_every_option_of_every_selection_does_not_raise():
    """Jede Option jeder Auswahl (select_slider, radio) einzeln; die Werte stammen aus SETTING_SPECS."""
    gefunden = 0
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for option in spec.options:
            at = _fresh_app()
            try:
                w = at.sidebar.select_slider(key=state_key)
            except KeyError:
                w = at.sidebar.radio(key=state_key)
            w.set_value(option).run(timeout=TIMEOUT)
            gefunden += 1
            assert not at.exception, (state_key, option, [str(e) for e in at.exception])
            assert at.session_state[state_key] == option
    assert gefunden == 6 + 11 + 6 + 5 + 2 + 3 + 6 + 5 + 7 + 3 + 4      # kappa, Ziel, λ₁, b, Verfahren, Größe, wahres λ₁, Entwurf, λ₀′, q_s, Änderung


def test_schwelle_ueber_der_reglergrenze_zeigt_meldung_statt_zahl():
    """κ = 28, H = 700: ARL₀ liegt weit über 2·10⁵ (der Löser liefert dort keine verlässliche Zahl): Meldung, kein Absturz, keine negative Zahl."""
    at = _fresh_app()
    at.sidebar.select_slider(key="kappa_select").set_value(28).run(timeout=TIMEOUT)
    at.sidebar.slider(key="h_slider").set_value(700).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert "über 2·10⁵" in _texte(at)
    arl0 = {m.label: m.value for m in at.metric if m.label.startswith("ARL₀")}
    assert arl0 == {"ARL₀ bei unabhängigen Lkw": "über 2·10⁵", "ARL₀ auf der Kette": "über 2·10⁵"}


def test_ziel_ausserhalb_der_gefrorenen_kennlinie_nennt_es():
    """Ziel 2000 bei κ = 18 liegt innerhalb (Schwelle 960); die Kennlinie endet bei H = 1200: kein Absturz bei keinem Ziel."""
    for kappa in C.KAPPA_OPTIONS:
        at = _fresh_app()
        at.sidebar.select_slider(key="kappa_select").set_value(kappa).run(timeout=TIMEOUT)
        at.sidebar.select_slider(key="target_select").set_value(2000).run(timeout=TIMEOUT)
        assert not at.exception, (kappa, [str(e) for e in at.exception])


def test_ausduennen_jeder_40_ist_nicht_gemessen_und_sagt_es():
    at = _fresh_app()
    at.sidebar.radio(key="verfahren_radio").set_value(C.VERFAHREN[1]).run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="block_select").set_value(40).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Ausdünnen jeder 40. Beobachtung ist nicht gemessen" in _texte(at)
