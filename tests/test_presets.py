"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, paarweise verschieden; Permalink begrenzt und rastet ein."""

import lwg_constants as C
import lwg_evaluation as ev
import lwg_presets as P


def test_every_preset_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    assert all(C.PRESET_HELP[name] for name in C.PRESETS)
    assert len(C.PRESETS) == 7


def test_preset_keys_gehoeren_zu_den_reglern():
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name


def test_preset_werte_liegen_in_den_optionen_und_grenzen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_standard_preset_ist_die_standardeinstellung():
    assert C.PRESETS[next(iter(C.PRESETS))] == {}
    assert ev.Settings() == ev.Settings(**{})


def test_standardwerte_der_regler_sind_die_der_settings():
    s = ev.Settings()
    soll = {"h_slider": s.h, "kappa_select": s.kappa, "target_select": s.target, "lam1_select": s.lam1, "block_select": s.block, "verfahren_radio": s.verfahren, "groesse_select": s.groesse,
            "live_r_slider": s.live_r, "live_seed_input": s.live_seed, "lam1_wahr_select": s.lam1_wahr, "lam1_entw_select": s.lam1_entw, "lam0_wahr_select": s.lam0_wahr, "qs_select": s.qs,
            "aenderung_radio": s.aenderung}
    assert {k: spec.default for k, spec in P.SETTING_SPECS.items()} == soll


def test_alle_presets_sind_paarweise_verschieden_und_ein_preset_wird_aktiv_erkannt():
    """Ein Preset, das mit einem früheren zusammenfiele, würde nie hervorgehoben (aktives_preset nimmt das erste passende)."""
    voll = []
    for preset in C.PRESETS.values():
        voll.append({sk: preset.get(k, P.SETTING_SPECS[sk].default) for k, sk in P.PRESET_KEYS.items()})
    assert all(a != b for i, a in enumerate(voll) for b in voll[i + 1:])


def test_setting_specs_decken_jeden_preset_schluessel():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_einrasten_von_permalink_werten():
    snap = P._snap
    assert snap(1000, P.SETTING_SPECS["h_slider"]) == 700 and snap(-5, P.SETTING_SPECS["h_slider"]) == 10
    assert snap(23, P.SETTING_SPECS["kappa_select"]) == 22 or snap(23, P.SETTING_SPECS["kappa_select"]) == 24
    assert snap(1.9, P.SETTING_SPECS["lam1_select"]) in (1.8, 2.0) and snap("Blockmittel", P.SETTING_SPECS["verfahren_radio"]) == "Blockmittel"
    assert snap("gibt es nicht", P.SETTING_SPECS["verfahren_radio"]) is None
    assert snap(120, P.SETTING_SPECS["live_r_slider"]) == 100 and snap(180, P.SETTING_SPECS["live_r_slider"]) == 200     # Raster 50
