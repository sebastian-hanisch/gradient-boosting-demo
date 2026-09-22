"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Aufgaben- und Verlustwechsel, Abspielen, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gb_constants as C
from gb_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_gives_one_verdict():
    at = _run()
    assert any("Gradient Boosting in Aktion" in m.value for m in at.markdown) and not at.error
    assert any("Sieht vernünftig aus" in m.value for m in at.markdown)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and not at.exception


def test_a_single_step_shows_no_play_button_and_says_so():
    at = _run(lambda a: _apply(a, C.PRESETS["🪓 Ein Schritt (kein Boosting)"]))
    assert not any(b.label == "▶️ Abspielen" for b in at.button)
    assert any("Nur eine Runde eingestellt" in i.value for i in at.info)


def test_switching_task_swaps_the_loss_options_and_keeps_a_valid_choice():
    at = _run()
    assert at.selectbox(key="loss_select").value in C.LOSSES["class"]
    at.session_state["task_select"] = "reg"
    at.run()
    assert not at.exception
    assert at.selectbox(key="loss_select").value in C.LOSSES["reg"]
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception
    assert at.selectbox(key="loss_select").value in C.LOSSES["class"]


def test_regression_hides_label_noise_and_shows_outliers_instead():
    at = _run()
    at.session_state["task_select"] = "reg"
    at.run()
    labels = {w.label for w in at.sidebar.slider}
    assert "Grobe Ausreißer im Training [%]" in labels and "Falsche Etiketten im Training [%]" not in labels
    at.session_state["task_select"] = "class"
    at.run()
    labels = {w.label for w in at.sidebar.slider}
    assert "Falsche Etiketten im Training [%]" in labels and "Grobe Ausreißer im Training [%]" not in labels


def test_outlier_experiment_only_appears_for_regression():
    at = _run()
    assert not any(b.key == "out_start" for b in at.button)
    at.session_state["task_select"] = "reg"
    at.run()
    assert any(b.key == "out_start" for b in at.button)


def test_a_kept_slider_survives_a_round_trip_through_the_other_task():
    at = _run()
    at.session_state["label_noise_slider"] = 12
    at.run()
    at.session_state["task_select"] = "reg"
    at.run()
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception and at.slider(key="label_noise_slider").value == 12


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_slider"] = C.N_MIN
        at.session_state["depth_slider"] = C.DEPTH_MIN
        at.session_state["leaf_slider"] = C.LEAF_MAX
        at.session_state["n_rounds_slider"] = C.N_ROUNDS_MIN
        at.session_state["n_noise_slider"] = 0
        at.session_state["subsample_slider"] = C.SUBSAMPLE_MIN

    def big(at):
        at.session_state["task_select"] = "reg"
        at.session_state["n_slider"] = C.N_MAX
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["leaf_slider"] = 1
        at.session_state["n_rounds_slider"] = 80
        at.session_state["n_noise_slider"] = C.NOISE_MAX
        at.session_state["outlier_slider"] = C.OUTLIER_MAX
    for setup in (small, big):
        at = _run(setup)
        assert not at.exception


def test_map_features_beyond_the_columns_fall_back_after_fewer_noise_features():
    at = _run()
    at.session_state["map_x_select"] = 10
    at.run()
    at.session_state["n_noise_slider"] = 0
    at.run()
    assert not at.exception and at.selectbox(key="map_x_select").value == C.DEFAULT_MAP[0]


def test_the_test_delivery_slider_survives_a_smaller_data_set():
    at = _run()
    at.session_state["sample_slider"] = 300
    at.run()
    at.session_state["n_slider"] = C.N_MIN
    at.run()
    assert not at.exception and at.slider(key="sample_slider").value == 0


def test_step_slider_returns_to_the_last_round_when_settings_change():
    at = _run()
    at.slider(key="gb_step").set_value(5)
    at.run()
    assert at.slider(key="gb_step").value == 5
    at.session_state["n_rounds_slider"] = 50
    at.run()
    assert not at.exception and at.slider(key="gb_step").value == 50


def test_every_round_of_a_small_ensemble_renders():
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 6))
    for k in range(1, int(at.slider(key="gb_step").max) + 1):
        at.slider(key="gb_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys(monkeypatch):
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb die Runde (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 8))
    _play(at)
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped_and_unknown_choices_fall_back():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["nr"] = "99999"
    at.query_params["depth"] = "-4"
    at.query_params["loss"] = "bogus"
    at.query_params["fx"] = "99"
    at.run()
    assert not at.exception
    assert at.slider(key="n_rounds_slider").value == C.N_ROUNDS_MAX and at.slider(key="depth_slider").value == C.DEPTH_MIN
    assert at.selectbox(key="loss_select").value == C.DEFAULT_LOSS[C.DEFAULT_TASK]                    # "bogus" gehört zu keiner Verlustfunktion


def test_the_address_bar_mirrors_the_settings():
    at = _run(lambda a: _apply(a, C.PRESETS["🚀 Zu große Lernrate"]))
    assert str(at.query_params["nr"]) in ("150", "['150']") and str(at.query_params["lr"]) in ("1.0", "['1.0']")


def test_learning_rate_experiment_runs_on_demand():
    at = _run()
    assert not any("Minimum bei Lernrate" in c.value for c in at.caption)
    at.button(key="lr_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "Minimum bei Lernrate" in text


def test_outlier_experiment_runs_on_demand():
    at = _run(lambda a: a.session_state.__setitem__("task_select", "reg"))
    at.button(key="out_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "grobe Ausreißer" in text or "Ausreißern im Training" in text


def test_subsample_experiment_runs_on_demand():
    at = _run()
    at.button(key="sub_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "Geschwindigkeitshebel" in text


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert sorted(set(keys)) == sorted(["tree_chart", "map_chart", "round_chart", "importance_chart", "lr_chart", "rounds_chart", "outlier_rmse_chart", "outlier_mae_chart",
                                        "subsample_chart", "subsample_timing_chart"]), keys
    looped = [c for c in calls if 'key=f"' in c]
    assert len(looped) == 2 and all('_{current}"' in c for c in looped)                              # die Bilder der Abspiel-Schleife tragen die Runde
    viz = (ROOT / "gb_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("lock_axes(fig") >= 6


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
