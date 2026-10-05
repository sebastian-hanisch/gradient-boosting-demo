"""Unabhängige Orakel für Gradient Boosting (Regressionstest der Orakelprüfung).

1. Replay-Orakel: die Runden werden mit eigenem Gradienten, eigenem Blattwert (Mittel, Median, Huber-Korrektur, Newton-Schritt, geschlossene Form) und einer Brute-Force-Split-Suche in Schleifen
   nachgerechnet. Der Baum wird Knoten für Knoten geprüft (maximaler Gain, Mindestblattgröße, Tiefe, Reinheit) - bei Gleichständen ist jeder maximale Split zulässig, deshalb kein Vergleich
   mit festen Bäumen (Pseudo-Residuen der ersten Log-Loss-Runde sind binär: lauter Gain-Gleichstände, die Bibliotheken unterschiedlich auflösen).
2. Trainingskurve und Kennzahl müssen dieselben Zielwerte messen (mit groben Ausreißern im Training lag die Kurve vorher gegen die sauberen Werte)."""

import math

import numpy as np
import pytest

import gb_algorithm as gb
import gb_evaluation as ev
import gb_tree as T


def _gains(X, y, ml):
    m, d = X.shape
    mu = math.fsum(y) / m
    parent = math.fsum((v - mu) ** 2 for v in y) / m

    def sse(ix):
        mm = math.fsum(y[i] for i in ix) / len(ix)
        return math.fsum((y[i] - mm) ** 2 for i in ix)

    out = []
    for f in range(d):
        for t in sorted(set(X[:, f]))[:-1]:
            left = [i for i in range(m) if X[i, f] <= t]
            right = [i for i in range(m) if X[i, f] > t]
            if len(left) >= ml and len(right) >= ml:
                out.append(parent - (sse(left) + sse(right)) / m)
    return out


def _check_tree(tree, X, g, depth, ml):
    members = {0: np.arange(len(g))}
    stack = [0]
    while stack:
        t = stack.pop()
        idx = members[t]
        yy = g[idx]
        gains = _gains(X[idx], yy, ml) if len(idx) >= 2 else []
        can_split = tree.depth[t] < depth and float(np.mean((yy - yy.mean()) ** 2)) > 1e-12 and bool(gains)
        if tree.feature[t] < 0:
            assert not can_split, f"Knoten {t} ist Blatt, obwohl ein Split möglich wäre"
            continue
        assert can_split, f"Knoten {t} wurde geteilt, obwohl nicht erlaubt"
        left = X[idx, tree.feature[t]] <= tree.threshold[t]
        assert left.sum() >= ml and (~left).sum() >= ml
        mine = yy.var() - (left.sum() * yy[left].var() + (~left).sum() * yy[~left].var()) / len(idx)
        assert max(gains) - mine <= 1e-9 * max(1.0, abs(max(gains))), f"Knoten {t}: Split nicht optimal"
        lt, rt = int(tree.left[t]), int(tree.right[t])
        members[lt], members[rt] = idx[left], idx[~left]
        stack += [lt, rt]


def _median(r):
    s = sorted(r)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _own_grad(y, F, task, loss, delta):
    if task == "reg":
        r = y - F
        if loss == "squared":
            return r
        if loss == "absolute":
            return np.sign(r)
        return np.array([ri if abs(ri) <= delta else math.copysign(delta, ri) for ri in r])
    if loss == "logloss":
        return y - 1.0 / (1.0 + np.exp(-F))
    s = 2 * y - 1
    return s * np.exp(-s * F)


def _own_leaf(y, F, task, loss, delta):
    if task == "reg":
        r = y - F
        if loss == "squared":
            return math.fsum(r) / len(r)
        if loss == "absolute":
            return _median(r)
        med = _median(r)
        return med + math.fsum(max(-delta, min(delta, ri - med)) for ri in r) / len(r)
    if loss == "logloss":
        p = 1.0 / (1.0 + np.exp(-F))
        return float((y - p).sum() / (p * (1 - p)).sum())
    s = 2 * y - 1
    return 0.5 * math.log(max(np.exp(-F[s > 0]).sum(), 1e-12) / max(np.exp(F[s < 0]).sum(), 1e-12))


CASES = [("reg", "squared"), ("reg", "absolute"), ("reg", "huber"), ("class", "logloss"), ("class", "exponential")]


@pytest.mark.parametrize("task,loss", CASES)
def test_ensemble_replay_matches_independent_gradient_leaf_and_brute_force_splits(task, loss):
    rng = np.random.default_rng(11)
    for it in range(8):
        n, d = int(rng.integers(8, 30)), int(rng.integers(1, 4))
        depth, ml, rounds, lr = int(rng.integers(1, 4)), int(rng.integers(1, 5)), int(rng.integers(1, 5)), float(rng.choice([0.1, 0.5, 1.0]))
        X = rng.integers(0, 6, (n, d)).astype(float) if it % 2 else rng.normal(size=(n, d))        # ganzzahlig = viele Gleichstände
        if task == "reg":
            y = 2 * X[:, 0] + rng.normal(size=n)
            if it % 3 == 0:
                y[:2] += 30.0
        else:
            y = (X[:, 0] + rng.normal(0, 0.7, n) > X[:, 0].mean()).astype(float)
            if y.min() == y.max():
                y[0] = 1.0 - y[0]
        e = gb.fit(X, y, task, loss, depth=depth, min_leaf=ml, n_rounds=rounds, learning_rate=lr)
        if task == "reg":
            f0 = float(np.median(y)) if loss in ("absolute", "huber") else float(np.mean(y))
        else:
            p = y.mean()
            f0 = math.log(p / (1 - p)) * (0.5 if loss == "exponential" else 1.0)
        assert e.f0 == pytest.approx(f0, abs=1e-12)
        F = np.full(n, f0)
        for k, tree in enumerate(e.trees):
            delta = float(np.quantile(np.abs(y - F), 0.9)) if loss == "huber" else None
            _check_tree(tree, X, _own_grad(y, F, task, loss, delta), depth, ml)
            leaf = T.apply(tree, X)
            newF = F.copy()
            for t in np.unique(leaf):
                ix = leaf == t
                v = _own_leaf(y[ix], F[ix], task, loss, delta)
                assert tree.value[t] == pytest.approx(v, rel=1e-8, abs=1e-8), (loss, k, t)
                newF[ix] = F[ix] + lr * v
            F = newF
        assert np.allclose(gb.predict_raw(e, X), F, atol=1e-8)


def test_squared_loss_matches_scikit_learn_on_many_random_instances():
    """Quadratischer Verlust hat kontinuierliche Pseudo-Residuen (keine Gleichstände) und stimmt mit scikit-learn exakt überein."""
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(3)
    for it in range(12):
        n, d = int(rng.integers(60, 120)), int(rng.integers(2, 5))
        depth, ml, rounds, lr = int(rng.integers(1, 4)), int(rng.integers(3, 9)), int(rng.integers(1, 15)), float(rng.choice([0.1, 0.3, 1.0]))
        X = rng.normal(size=(n, d))
        y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.5, n)
        Xt = rng.normal(size=(30, d))
        e = gb.fit(X, y, "reg", "squared", depth=depth, min_leaf=ml, n_rounds=rounds, learning_rate=lr)
        ref = ensemble.GradientBoostingRegressor(loss="squared_error", n_estimators=rounds, max_depth=depth, min_samples_leaf=ml, learning_rate=lr, random_state=0).fit(X, y)
        assert np.max(np.abs(gb.predict_value(e, Xt) - ref.predict(Xt))) < 1e-10


@pytest.mark.parametrize("task,loss,outlier", [("reg", "squared", 0), ("reg", "huber", 15), ("reg", "squared", 15), ("class", "logloss", 0)])
def test_training_curve_ends_on_the_same_training_error_as_the_metric(task, loss, outlier):
    a = ev.analyse(task, loss, 2, 5, 12, 0.2, 100, 300, 2, 10 if task == "class" else 0, outlier, 3)
    last = ev.round_rows(a, ks=[a.n_rounds])[0]
    assert last["train"] == pytest.approx(ev.primary(a.train, task), abs=1e-12)
    assert last["test"] == pytest.approx(ev.primary(a.test, task), abs=1e-12)
