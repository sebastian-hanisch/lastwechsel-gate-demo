"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import lwg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # Auswahl (select_slider, radio): nur diese Werte sind gültig
    step: float | None = None         # Raster eines Zahlenreglers (Permalink rastet ein)


SETTING_SPECS = {
    "h_slider": SettingSpec("h", int, C.DEFAULT_H, C.H_MIN, C.H_MAX, step=1),
    "kappa_select": SettingSpec("kappa", int, C.DEFAULT_KAPPA, C.KAPPA_OPTIONS[0], C.KAPPA_OPTIONS[-1], options=C.KAPPA_OPTIONS),
    "target_select": SettingSpec("ziel", int, C.DEFAULT_TARGET, C.TARGET_OPTIONS[0], C.TARGET_OPTIONS[-1], options=C.TARGET_OPTIONS),
    "lam1_select": SettingSpec("lam1", float, C.DEFAULT_LAM1, C.LAM1_OPTIONS[0], C.LAM1_OPTIONS[-1], options=C.LAM1_OPTIONS),
    "block_select": SettingSpec("b", int, C.DEFAULT_BLOCK, C.BLOCK_OPTIONS[0], C.BLOCK_OPTIONS[-1], options=C.BLOCK_OPTIONS),
    "verfahren_radio": SettingSpec("verf", str, C.DEFAULT_VERFAHREN, options=C.VERFAHREN),
    "groesse_select": SettingSpec("gr", float, C.DEFAULT_GROESSE, C.GROESSE_OPTIONS[0], C.GROESSE_OPTIONS[-1], options=C.GROESSE_OPTIONS),
    "live_r_slider": SettingSpec("lr", int, C.DEFAULT_LIVE_R, C.LIVE_R_MIN, C.LIVE_R_MAX, step=C.LIVE_R_STEP),
    "live_seed_input": SettingSpec("seed", int, C.DEFAULT_LIVE_SEED, 0, C.LIVE_SEED_MAX),
    "lam1_wahr_select": SettingSpec("lw", float, C.DEFAULT_LAM1, C.LAM1_WAHR_OPTIONS[0], C.LAM1_WAHR_OPTIONS[-1], options=C.LAM1_WAHR_OPTIONS),
    "lam1_entw_select": SettingSpec("le", float, C.DEFAULT_LAM1, C.LAM1_ENTW_OPTIONS[0], C.LAM1_ENTW_OPTIONS[-1], options=C.LAM1_ENTW_OPTIONS),
    "lam0_wahr_select": SettingSpec("l0", float, C.DEFAULT_LAM0_WAHR, C.LAM0_WAHR_OPTIONS[0], C.LAM0_WAHR_OPTIONS[-1], options=C.LAM0_WAHR_OPTIONS),
    "qs_select": SettingSpec("qs", int, C.DEFAULT_QS, C.QS_OPTIONS[0], C.QS_OPTIONS[-1], options=C.QS_OPTIONS),
    "aenderung_radio": SettingSpec("aend", str, C.DEFAULT_AENDERUNG, options=C.AENDERUNGEN),
}
PRESET_KEYS = {"h": "h_slider", "kappa": "kappa_select", "target": "target_select", "lam1": "lam1_select", "block": "block_select", "verfahren": "verfahren_radio",
               "groesse": "groesse_select", "live_r": "live_r_slider", "seed": "live_seed_input", "lam1_wahr": "lam1_wahr_select", "lam1_entw": "lam1_entw_select",
               "lam0_wahr": "lam0_wahr_select", "qs": "qs_select", "aenderung": "aenderung_radio"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(value, spec: SettingSpec):
    """Permalink-Wert auf die Reglergrenzen begrenzen und auf das Raster bzw. die Auswahl einrasten; None, wenn nicht möglich."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else None
        return min(spec.options, key=lambda o: abs(o - value))
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step is not None and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(spec.hi, value)
        if spec.caster is float:
            value = round(value, 4)
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            value = _snap(value, spec)
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt ALLE Regler: was es nicht nennt, kehrt zum Standardwert zurück (sonst bliebe ein früheres Preset halb erhalten)."""
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-9
    return a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["live_seed_input"] = random.randint(0, C.LIVE_SEED_MAX)
