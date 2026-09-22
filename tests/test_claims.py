"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-22, Toleranzen fangen Rundung ab). `analyse()` und die Experiment-Funktionen sind deterministisch (kein Bootstrap,
kein Zufall außer im Datenerzeuger und der - festen, mit `seed` reproduzierbaren - Teilstichprobe) - keine Zufallsstreuung zwischen Testläufen."""

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

import gb_algorithm as gb
import gb_constants as C
import gb_evaluation as ev
import gb_scenario as S

ADABOOST_DIR = Path(__file__).resolve().parents[2] / "adaboost-demo"

PRESET = {"standard": "🌳 Standard", "stump": "🪓 Ein Schritt (kein Boosting)", "highlr": "🚀 Zu große Lernrate", "lowlr": "🐌 Kleine Lernrate, viele Runden",
          "exp": "⚔️ Exponentiell (= AdaBoost)", "outlier": "🛡️ Robuster Verlust bei Ausreißern", "sub": "🎲 Teilstichprobe"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["loss"], p["depth"], p["leaf"], p["n_rounds"], p["lr"], p["subsample"], p["n"], p["n_noise"], p["label_noise"], p["outlier"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


@functools.lru_cache(maxsize=None)
def _default():
    return ev.analyse("class", "logloss", 2, 5, C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_SUBSAMPLE, C.DEFAULT_N, C.DEFAULT_NOISE, 0, 0, C.DEFAULT_SEED)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_standard_preset():
    a = _preset("standard")
    assert a.verdict == "ok" and len(a.ensemble.trees) == 60
    assert (a.train["error"], a.test["error"]) == pytest.approx((0.1, 0.1556), abs=0.0005)
    _help("standard", "10.0 %", "15.6 %")


def test_single_step_preset_is_no_better_than_a_shallow_stump():
    a = _preset("stump")
    assert len(a.ensemble.trees) == 1 and a.verdict == "stump"
    assert a.test["error"] == pytest.approx(0.2444, abs=0.0005)
    _help("stump", "24.4 %")


def test_too_large_learning_rate_overfits():
    a = _preset("highlr")
    standard = _preset("standard")
    assert (a.train["error"], a.test["error"]) == pytest.approx((0.0, 0.1778), abs=0.0005)
    assert a.test["error"] > standard.test["error"]                                              # schlechter als der Standard trotz perfektem Trainingsfehler
    _help("highlr", "0 %", "17.8 %", "15.6 %")


def test_small_learning_rate_many_rounds_is_at_least_as_good():
    a = _preset("lowlr")
    standard = _preset("standard")
    assert a.test["error"] == pytest.approx(0.15, abs=0.0005)
    assert a.test["error"] <= standard.test["error"]
    _help("lowlr", "15.0 %", "15.6 %")


def test_exponential_preset_numbers():
    a = _preset("exp")
    assert a.loss == "exponential" and len(a.ensemble.trees) == 30
    assert a.test["error"] == pytest.approx(0.2056, abs=0.0005)
    _help("exp", "20.6 %", "92.7 %")


def test_robust_loss_beats_squared_under_outliers():
    a = _preset("outlier")
    assert a.task == "reg" and a.loss == "huber"
    assert a.test["rmse"] == pytest.approx(10.618, abs=0.01)
    squared = ev.analyse("reg", "squared", a.depth, a.leaf, a.n_rounds, a.lr, a.subsample, C.DEFAULT_N, C.DEFAULT_NOISE, 0, 10, C.DEFAULT_SEED)
    assert squared.test["rmse"] == pytest.approx(12.312, abs=0.01)
    assert a.test["rmse"] < squared.test["rmse"]
    _help("outlier", "10.6", "12.3")


def test_subsample_preset_numbers():
    a = _preset("sub")
    assert a.subsample == 40 and a.n_rounds == 100
    assert a.test["error"] == pytest.approx(0.1333, abs=0.0005)
    _help("sub", "13.3 %", "15.0 %", "0.15 s", "0.26 s")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["loss"] in C.LOSSES[p["task"]]
        assert C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX and C.LEAF_MIN <= p["leaf"] <= C.LEAF_MAX and C.N_ROUNDS_MIN <= p["n_rounds"] <= C.N_ROUNDS_MAX
        assert C.LR_MIN <= p["lr"] <= C.LR_MAX and C.SUBSAMPLE_MIN <= p["subsample"] <= C.SUBSAMPLE_MAX
        d = C.N_BASE + p["n_noise"]
        assert C.N_MIN <= p["n"] <= C.N_MAX and 0 <= p["fx"] < d and 0 <= p["fy"] < d and name in C.PRESET_HELP


# --- Standardansicht --------------------------------------------------------------------------------------------------------------------------------

def test_default_view_numbers():
    a = _default()
    assert (a.train["error"], a.test["error"], a.baseline) == pytest.approx((0.1, 0.1556, 0.4639), abs=0.0005)
    assert a.verdict == "ok"


def test_round_curve_reaches_a_minimum_before_the_last_round():
    a = _default()
    rows = ev.round_rows(a)
    best = ev.best_round(rows)
    assert best["k"] <= a.n_rounds and best["test"] <= rows[-1]["test"] + 1e-9
    assert best["k"] == 35 and best["test"] == pytest.approx(0.15, abs=0.0005)


# --- Lernrate x Rundenzahl ---------------------------------------------------------------------------------------------------------------------------

def test_learning_rate_grid_has_a_minimum_away_from_the_extremes():
    rows = ev.lr_grid_rows("class", "logloss", 2, 5, 150, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    vals = [r["test"] for r in rows]
    assert vals == pytest.approx([0.15, 0.1417, 0.1367, 0.1389, 0.1572, 0.17], abs=0.001)
    assert min(vals) == vals[2]                                                                    # Minimum bei Lernrate 0.1, nicht an den Rändern


def test_rounds_grid_shows_the_learning_rate_tradeoff():
    small = ev.rounds_grid_rows("class", "logloss", 2, 5, 0.1, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    large = ev.rounds_grid_rows("class", "logloss", 2, 5, 1.0, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    small_vals = [r["test"] for r in small]
    large_vals = [r["test"] for r in large]
    assert small_vals == pytest.approx([0.2183, 0.1844, 0.155, 0.1494, 0.1372, 0.1367, 0.1361], abs=0.001)
    assert large_vals == pytest.approx([0.1506, 0.1483, 0.1611, 0.1628, 0.1711, 0.17, 0.1689], abs=0.001)
    assert small_vals[-1] < small_vals[0]                                                          # kleine Lernrate: Testfehler sinkt oder bleibt stabil mit mehr Runden
    assert large_vals[-1] > large_vals[1]                                                          # große Lernrate: Testfehler steigt nach einem frühen Minimum wieder


# --- Verlustwahl bei Ausreißern ----------------------------------------------------------------------------------------------------------------------

def test_outlier_loss_experiment_numbers():
    rows = ev.outlier_loss_rows(2, 5, C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_N, C.DEFAULT_NOISE)
    r0, r1 = rows[0], rows[-1]
    assert r0["outlier_pct"] == 0 and r1["outlier_pct"] == 15
    assert (r0["squared_rmse"], r0["huber_rmse"]) == pytest.approx((9.871, 10.379), abs=0.01)
    assert (r1["squared_rmse"], r1["huber_rmse"], r1["absolute_rmse"]) == pytest.approx((12.2, 11.838, 11.533), abs=0.01)
    assert r0["squared_rmse"] < r0["huber_rmse"]                                                   # ohne Ausreißer: Quadratisch leicht vorn
    assert r1["squared_rmse"] > r1["huber_rmse"] > r1["absolute_rmse"] - 0.5                       # mit 15 % Ausreißern: robuste Verluste klar vorn
    squared_growth = r1["squared_rmse"] - r0["squared_rmse"]
    huber_growth = r1["huber_rmse"] - r0["huber_rmse"]
    assert squared_growth > huber_growth                                                            # Quadratisch verschlechtert sich stärker als Huber


# --- Wirkung der Teilstichprobe ---------------------------------------------------------------------------------------------------------------------

def test_subsample_experiment_numbers():
    rows = ev.subsample_rows("class", "logloss", 2, 5, C.DEFAULT_N_ROUNDS, C.DEFAULT_LR, C.DEFAULT_N, C.DEFAULT_NOISE, 0)
    vals = {r["subsample"]: r["test"] for r in rows}
    assert vals == pytest.approx({100: 0.1433, 80: 0.1433, 60: 0.1444, 40: 0.1422, 20: 0.1522}, abs=0.001)
    assert max(vals.values()) - min(vals.values()) < 0.02                                          # uneinheitlich, aber in einer engen Spanne - kein klarer Regularisierungseffekt hier


def test_subsample_timing_decreases_monotonically():
    rows = ev.subsample_timing("class", "logloss", 2, 5, 100, 0.1, C.DEFAULT_N, C.DEFAULT_NOISE, grid=(100, 60, 20))
    secs = [r["seconds"] for r in rows]
    assert secs[0] > secs[1] > secs[2]                                                              # weniger Zeilen je Runde -> schneller, unabhängig von der Maschine


# --- Exponentieller Verlust = AdaBoost-Sonderfall (gegen adaboost-demo) ------------------------------------------------------------------------------

def test_exponential_loss_agrees_with_adaboost_demo_on_average():
    if not ADABOOST_DIR.exists():
        pytest.skip("adaboost-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(ADABOOST_DIR))
    import ada_algorithm as ada

    agreements = []
    for seed in C.SWEEP_SEEDS:
        ds = S.generate_dataset(seed=seed)
        Xtr, ytr, Xte, yte = S.split(ds, "class")
        ens_ada = ada.fit(Xtr, ytr, criterion="gini", max_depth=1, min_leaf=1, n_rounds=25)
        ens_gb = gb.fit(Xtr, ytr.astype(float), "class", "exponential", depth=1, min_leaf=1, n_rounds=25, learning_rate=1.0, subsample=1.0, seed=0)
        agreements.append(float(np.mean(ada.predict(ens_ada, Xte) == gb.predict(ens_gb, Xte))))
    assert np.mean(agreements) == pytest.approx(0.9272, abs=0.005)
    assert min(agreements) > 0.85


# --- Erzeuger (geteilt mit cart-demo) ------------------------------------------------------------------------------------------------------------------

def test_generator_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 11) and ds.names[:2] == ("Distanz", "Ladegewicht")
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150


def test_add_outliers_only_touches_training_rows_and_the_chosen_share():
    ds = S.generate_dataset(1200, 3, 0, 7)
    Xtr, ytr, Xte, yte = S.split(ds, "reg")
    y2 = S.add_outliers(ytr, 20, 7)
    changed = ~np.isclose(y2, ytr)
    assert 0.1 < changed.mean() < 0.3                                                              # ungefähr 20 % (Zufallsanteil)
    assert np.abs(y2[changed] - ytr[changed]).min() >= 60.0 - 1e-9
