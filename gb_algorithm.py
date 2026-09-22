"""Gradient Boosting (Friedman 2001): AdaBoost verallgemeinert auf **jede differenzierbare Verlustfunktion**. Jede Runde: den negativen Gradienten der Verlustfunktion an den aktuellen Vorhersagen ausrechnen
(die "Pseudo-Residuen"), einen Regressionsbaum darauf wachsen (derselbe Kern wie in cart-demo, Varianz-Kriterium - Pseudo-Residuen sind immer Zahlen, auch bei Klassifikation), dann in jedem Blatt den
verlustoptimalen Wert nachtragen (bei quadratischem Verlust ist das der Mittelwert, den der Baum ohnehin schon liefert; sonst ein anderer, verlustspezifischer Wert). Die Vorhersage wächst additiv:
F_m(x) = F_{m-1}(x) + Lernrate · Baum_m(x)."""

from dataclasses import dataclass

import numpy as np

import gb_tree as T

EPS = 1e-12


@dataclass(frozen=True)
class Ensemble:
    trees: tuple
    f0: float                     # Startwert (bester konstanter Wert vor der ersten Runde)
    learning_rate: float
    task: str                     # "class" | "reg"
    loss: str
    n_train: int
    deltas: tuple                  # Huber: der je Runde verwendete Schwellwert delta (sonst leer)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# --- Verlustfunktionen: Wert, negativer Gradient (Pseudo-Residuum), optimaler Blattwert -----------------------------------------------------------------

def loss_value(y, F, task, loss, delta=None):
    """Der Verlust je Zeile (nicht gemittelt) - für den numerischen Gradienten-Check."""
    if task == "reg":
        r = y - F
        if loss == "squared":
            return 0.5 * r ** 2
        if loss == "absolute":
            return np.abs(r)
        if loss == "huber":
            a = np.abs(r)
            return np.where(a <= delta, 0.5 * r ** 2, delta * (a - 0.5 * delta))
    p = _sigmoid(F)
    if loss == "logloss":
        p = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if loss == "exponential":
        y_pm1 = 2.0 * y - 1.0
        return np.exp(-y_pm1 * F)
    raise ValueError((task, loss))


def negative_gradient(y, F, task, loss, delta=None):
    """-dL/dF an der Stelle F, zeilenweise (die "Pseudo-Residuen", auf die der nächste Baum wächst)."""
    if task == "reg":
        r = y - F
        if loss == "squared":
            return r
        if loss == "absolute":
            return np.sign(r)
        if loss == "huber":
            return np.clip(r, -delta, delta)
    if loss == "logloss":
        return y - _sigmoid(F)
    if loss == "exponential":
        y_pm1 = 2.0 * y - 1.0
        return y_pm1 * np.exp(-y_pm1 * F)
    raise ValueError((task, loss))


def leaf_value(y, F, task, loss, delta=None):
    """Der verlustoptimale konstante Wert für ein Blatt mit den Zeilen `y`, `F` (aktuelle Vorhersage vor diesem Baum)."""
    if task == "reg":
        r = y - F
        if loss == "squared":
            return float(r.mean())
        if loss == "absolute":
            return float(np.median(r))
        if loss == "huber":
            med = float(np.median(r))
            centered = r - med
            correction = float(np.mean(np.sign(centered) * np.minimum(delta, np.abs(centered))))
            return med + correction
    if loss == "logloss":
        p = _sigmoid(F)
        num = float((y - p).sum())                     # Newton-Schritt: Zähler ist der Gradient (y-p), NICHT y-F (F ist die Log-Odds-Skala, keine Wahrscheinlichkeit)
        den = float((p * (1.0 - p)).sum())
        return num / den if den > EPS else 0.0
    if loss == "exponential":
        y_pm1 = 2.0 * y - 1.0
        w = np.exp(-y_pm1 * F)
        num = float(w[y_pm1 > 0].sum())
        den = float(w[y_pm1 < 0].sum())
        return 0.5 * np.log(max(num, EPS) / max(den, EPS))
    raise ValueError((task, loss))


def init_value(y, task, loss):
    """Der beste konstante Start (F_0): Mittelwert bzw. Median für Regression, (halbe) Log-Odds für Klassifikation."""
    if task == "reg":
        return float(np.median(y)) if loss in ("absolute", "huber") else float(np.mean(y))
    p = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
    return 0.5 * np.log(p / (1.0 - p)) if loss == "exponential" else np.log(p / (1.0 - p))


# --- Fit / Vorhersage ------------------------------------------------------------------------------------------------------------------------------------

def fit(X, y, task, loss, depth=2, min_leaf=5, n_rounds=60, learning_rate=0.1, subsample=1.0, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    f0 = init_value(y, task, loss)
    F = np.full(n, f0)
    rng = np.random.default_rng(seed)
    trees, deltas = [], []
    for m in range(n_rounds):
        if subsample < 1.0:
            k = max(2, int(round(subsample * n)))
            idx = np.sort(rng.choice(n, k, replace=False))
        else:
            idx = np.arange(n)
        delta = None
        if loss == "huber":
            delta = float(np.quantile(np.abs(y[idx] - F[idx]), 0.9))
            delta = max(delta, 1e-6)
        grad = negative_gradient(y[idx], F[idx], task, loss, delta)
        tree = T.grow(X[idx], grad, "reg", "variance", depth, min_leaf)
        y_idx, F_idx = y[idx], F[idx]
        tree = T.set_leaf_values(tree, X[idx], lambda li: leaf_value(y_idx[li], F_idx[li], task, loss, delta))
        F = F + learning_rate * T.predict_value(tree, X)
        trees.append(tree)
        deltas.append(delta if delta is not None else float("nan"))
    return Ensemble(tuple(trees), f0, learning_rate, task, loss, n, tuple(deltas))


def predict_raw(ensemble, X, upto=None):
    """F(x): die additive Summe aus Start und den (mit der Lernrate skalierten) Bäumen - vor jeder Umrechnung in Wahrscheinlichkeit oder Klasse."""
    trees = ensemble.trees[:upto] if upto else ensemble.trees
    F = np.full(len(X), ensemble.f0)
    for t in trees:
        F = F + ensemble.learning_rate * T.predict_value(t, X)
    return F


def predict_value(ensemble, X, upto=None):
    F = predict_raw(ensemble, X, upto)
    if ensemble.task == "reg":
        return F
    scale = 2.0 if ensemble.loss == "exponential" else 1.0                    # exponentieller Verlust rechnet auf der "halben" Log-Odds-Skala (wie SAMME)
    return _sigmoid(scale * F)


def predict(ensemble, X, upto=None):
    v = predict_value(ensemble, X, upto)
    return (v > 0.5).astype(int) if ensemble.task == "class" else v
