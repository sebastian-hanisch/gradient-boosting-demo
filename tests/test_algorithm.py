"""Gradient Boosting gegen unabhängige Referenzen: numerischer Gradienten-Check für jede Verlustfunktion, exakter Abgleich mit scikit-learns GradientBoostingRegressor(loss="squared_error") und
GradientBoostingClassifier(loss="log_loss") (beide deterministisch bei subsample=1 - derselbe Ablauf auf denselben Daten muss exakt gleich sein), Toleranz-Abgleich für Huber (leicht andere
Delta-Konvention) und den exponentiellen Verlust (sklearn nutzt einen einzelnen Newton-Schritt, diese Demo die geschlossene AdaBoost-Alpha-Formel - siehe README)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import gb_algorithm as gb
import gb_scenario as S
import gb_tree as T

ADABOOST_DIR = Path(__file__).resolve().parents[2] / "adaboost-demo"


def _continuous_reg(n=300, d=5, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, noise, n)
    return X, y


def _continuous_cls(n=300, d=5, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    signal = X[:, 0] + 0.7 * np.sin(2 * X[:, 1])
    y = (signal + rng.normal(0, noise, n) > 0).astype(float)
    return X, y


# --- Numerischer Gradienten-Check je Verlust --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("task,loss,delta", [("reg", "squared", None), ("reg", "absolute", None), ("reg", "huber", 1.0), ("class", "logloss", None), ("class", "exponential", None)])
def test_negative_gradient_matches_finite_difference(task, loss, delta):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float) if task == "class" else rng.normal(0, 5, 200)
    F = rng.normal(0, 2, 200)
    eps = 1e-6
    analytic = gb.negative_gradient(y, F, task, loss, delta)
    numeric = -(gb.loss_value(y, F + eps, task, loss, delta) - gb.loss_value(y, F - eps, task, loss, delta)) / (2 * eps)
    assert np.max(np.abs(analytic - numeric)) < 1e-6


# --- Exakter Abgleich: quadratischer Verlust (Regression) ------------------------------------------------------------------------------------------

def test_squared_regression_matches_scikit_learn_exactly():
    X, y = _continuous_reg(300, 5, 1)
    Xt, _ = _continuous_reg(80, 5, 2)
    ens = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=30, learning_rate=0.3, subsample=1.0, seed=0)
    ref = GradientBoostingRegressor(loss="squared_error", n_estimators=30, max_depth=2, min_samples_leaf=5, learning_rate=0.3, subsample=1.0, random_state=0).fit(X, y)
    assert ens.f0 == pytest.approx(ref.init_.constant_.ravel()[0])
    assert np.max(np.abs(gb.predict_value(ens, Xt) - ref.predict(Xt))) < 1e-10


# --- Exakter Abgleich: Log-Loss (Klassifikation) ----------------------------------------------------------------------------------------------------

def test_logloss_classification_matches_scikit_learn_exactly():
    X, y = _continuous_cls(300, 5, 0)
    Xt, _ = _continuous_cls(80, 5, 3)
    ens = gb.fit(X, y, "class", "logloss", depth=2, min_leaf=5, n_rounds=40, learning_rate=0.2, subsample=1.0, seed=0)
    ref = GradientBoostingClassifier(loss="log_loss", n_estimators=40, max_depth=2, min_samples_leaf=5, learning_rate=0.2, subsample=1.0, random_state=0).fit(X, y.astype(int))
    assert np.max(np.abs(gb.predict_value(ens, Xt) - ref.predict_proba(Xt)[:, 1])) < 1e-10
    assert np.array_equal(gb.predict(ens, Xt), ref.predict(Xt))


# --- Toleranz-Abgleich: Huber (Delta-Konvention weicht leicht ab) -----------------------------------------------------------------------------------

def test_huber_regression_is_close_to_scikit_learn():
    X, y = _continuous_reg(300, 5, 4)
    Xt, _ = _continuous_reg(80, 5, 5)
    ens = gb.fit(X, y, "reg", "huber", depth=2, min_leaf=5, n_rounds=30, learning_rate=0.3, subsample=1.0, seed=0)
    ref = GradientBoostingRegressor(loss="huber", alpha=0.9, n_estimators=30, max_depth=2, min_samples_leaf=5, learning_rate=0.3, subsample=1.0, random_state=0).fit(X, y)
    diff = np.abs(gb.predict_value(ens, Xt) - ref.predict(Xt))
    assert diff.mean() < 2.0                                                     # grobe Größenordnung, keine exakte Übereinstimmung (siehe Docstring)


# --- Exponentieller Verlust: geschlossene Blattwertformel + AdaBoost-Sonderfall --------------------------------------------------------------------

def test_exponential_leaf_value_matches_closed_form_minimizer():
    """Der Blattwert minimiert den exponentiellen Verlust exakt in diesem Blatt (nicht nur ein Newton-Schritt) - Nachrechnen mit Brute-Force über ein feines Gitter."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60).astype(float)
    F = rng.normal(0, 1, 60)
    gamma = gb.leaf_value(y, F, "class", "exponential")
    grid = np.linspace(gamma - 2, gamma + 2, 4001)
    losses = [float(gb.loss_value(y, F + g, "class", "exponential").sum()) for g in grid]
    assert grid[int(np.argmin(losses))] == pytest.approx(gamma, abs=2e-3)


def test_exponential_loss_reduces_to_adaboost_with_stumps_and_no_shrinkage():
    """Tiefe 1, Lernrate 1, exponentieller Verlust: dieselbe Blattwertformel wie AdaBoosts Stimmgewicht alpha (halbiert, halbe Log-Odds-Skala) - die Vorhersagen stimmen mit adaboost-demo überwiegend
    überein (nicht exakt: die Bäume wachsen hier auf Pseudo-Residuen mit dem Varianz-Kriterium, AdaBoost gewichtet Gini direkt)."""
    if not ADABOOST_DIR.exists():
        pytest.skip("adaboost-demo nicht neben diesem Repo gefunden")
    sys.path.insert(0, str(ADABOOST_DIR))
    import ada_algorithm as ada

    agreements = []
    for seed in range(5):
        ds = S.generate_dataset(seed=100000 + seed)
        Xtr, ytr, Xte, yte = S.split(ds, "class")
        ens_ada = ada.fit(Xtr, ytr, criterion="gini", max_depth=1, min_leaf=1, n_rounds=25)
        ens_gb = gb.fit(Xtr, ytr.astype(float), "class", "exponential", depth=1, min_leaf=1, n_rounds=25, learning_rate=1.0, subsample=1.0, seed=0)
        agreements.append(float(np.mean(ada.predict(ens_ada, Xte) == gb.predict(ens_gb, Xte))))
    assert np.mean(agreements) > 0.85


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_a_single_round_equals_f0_plus_learning_rate_times_the_first_tree():
    X, y = _continuous_reg(150, 4, 0)
    ens = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=1, learning_rate=0.5, subsample=1.0, seed=0)
    expected = ens.f0 + 0.5 * T.predict_value(ens.trees[0], X)
    assert np.allclose(gb.predict_value(ens, X), expected)


def test_min_leaf_is_respected_in_every_round_tree():
    X, y = _continuous_reg(200, 4, 1)
    ens = gb.fit(X, y, "reg", "squared", depth=4, min_leaf=15, n_rounds=10, learning_rate=0.2, subsample=1.0, seed=0)
    for tree in ens.trees:
        leaves = tree.feature < 0
        counts = np.bincount(T.apply(tree, X), minlength=tree.n_nodes)
        assert (counts[leaves] >= 15).all()


def test_subsample_is_reproducible_with_the_same_seed_and_varies_with_a_different_one():
    X, y = _continuous_reg(300, 5, 2)
    ens1 = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens2 = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=7)
    ens3 = gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=20, learning_rate=0.2, subsample=0.5, seed=8)
    assert np.array_equal(gb.predict_value(ens1, X), gb.predict_value(ens2, X))
    assert not np.array_equal(gb.predict_value(ens1, X), gb.predict_value(ens3, X))


def test_more_rounds_improve_or_maintain_training_fit_for_squared_loss():
    X, y = _continuous_reg(300, 5, 3)
    err5 = np.mean((y - gb.predict_value(gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=5, learning_rate=0.2, subsample=1.0, seed=0), X)) ** 2)
    err50 = np.mean((y - gb.predict_value(gb.fit(X, y, "reg", "squared", depth=2, min_leaf=5, n_rounds=50, learning_rate=0.2, subsample=1.0, seed=0), X)) ** 2)
    assert err50 < err5
