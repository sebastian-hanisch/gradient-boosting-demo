"""Messungen an Gradient Boosting: Testfehler-Kurve gegen die Rundenzahl (Frühstopp), Lernrate x Rundenzahl (Kompromiss), Verlustwahl bei Ausreißern (Regression), Wirkung der Teilstichprobe."""

from dataclasses import dataclass

import numpy as np

import gb_algorithm as gb
import gb_constants as C
import gb_scenario as S
import gb_tree as T

SIX = (C.DEFAULT_SEED,) + C.SWEEP_SEEDS


def baseline_error(ds, task):
    """Fehler ohne Modell: Klassifikation = immer die häufigere Klasse des Trainings raten, Regression = immer der Trainings-Mittelwert."""
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def _metrics(ensemble, X, y, task):
    if task == "class":
        pred = gb.predict(ensemble, X)
        return {"error": float(np.mean(pred != y))}
    v = gb.predict_value(ensemble, X)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def primary(metrics, task):
    return metrics["error"] if task == "class" else metrics["rmse"]


_primary = primary


def ensemble_importances(ensemble):
    """Wichtigkeit je Merkmal, gemittelt über alle Bäume des Ensembles (jeder Baum liefert Anteile, die sich zu 1 summieren)."""
    imps = [T.importances(t) for t in ensemble.trees]
    return np.mean(imps, axis=0) if imps else np.zeros(0)


@dataclass
class Analysis:
    ds: object
    task: str
    loss: str
    depth: int
    leaf: int
    n_rounds: int
    lr: float
    subsample: float
    ensemble: object
    train: dict
    test: dict
    baseline: float
    verdict: str
    imp: np.ndarray


def analyse(task, loss, depth, leaf, n_rounds, lr, subsample, n, n_noise, label_noise, outlier, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    if task == "reg":
        ytr = S.add_outliers(ytr, outlier, seed)
    ensemble = gb.fit(Xtr, ytr.astype(float), task, loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=subsample / 100.0, seed=0)
    train = _metrics(ensemble, Xtr, ytr, task)
    test = _metrics(ensemble, Xte, yte, task)
    baseline = baseline_error(ds, task)
    a = Analysis(ds, task, loss, depth, leaf, n_rounds, lr, subsample, ensemble, train, test, baseline, "", ensemble_importances(ensemble))
    a.verdict = verdict(a)
    return a


def verdict(a):
    """'stump' (nur ein Schritt), 'overfit' (Test deutlich schlechter als Training), 'underfit' (kaum besser als Raten), sonst 'ok'."""
    tr, te = _primary(a.train, a.task), _primary(a.test, a.task)
    if a.n_rounds <= 1:
        return "stump"
    over = (te - tr > C.OVERFIT_GAP_CLASS) if a.task == "class" else (te > C.OVERFIT_RATIO_REG * max(tr, 1e-9))
    if over:
        return "overfit"
    return "underfit" if te > C.UNDERFIT_SHARE * a.baseline else "ok"


# --- Testfehler-Kurve gegen die Rundenzahl (Frühstopp) ---------------------------------------------------------------------------------------------

def round_rows(a, ks=None):
    """Trainings- und Testfehler je Rundenzahl `k` (bis `a.n_rounds`, geometrisch verteilt) - zeigt, ob und wann der Testfehler nach einem Minimum wieder steigt."""
    ks = ks or sorted(set(np.unique(np.round(np.geomspace(1, a.n_rounds, min(24, a.n_rounds))).astype(int))))
    Xtr, ytr, Xte, yte = S.split(a.ds, a.task)
    if a.task == "reg":
        ytr = S.add_outliers(ytr, 0, a.ds.seed)
    rows = []
    for k in ks:
        tr = _primary(_metrics_upto(a.ensemble, Xtr, ytr, a.task, k), a.task)
        te = _primary(_metrics_upto(a.ensemble, Xte, yte, a.task, k), a.task)
        rows.append({"k": int(k), "train": tr, "test": te})
    return rows


def _metrics_upto(ensemble, X, y, task, upto):
    if task == "class":
        pred = gb.predict(ensemble, X, upto=upto)
        return {"error": float(np.mean(pred != y))}
    v = gb.predict_value(ensemble, X, upto=upto)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def best_round(rows):
    """Die Runde mit dem kleinsten Testfehler (bei Gleichstand die frühere)."""
    return min(rows, key=lambda r: (r["test"], r["k"]))


# --- Lernrate x Rundenzahl -------------------------------------------------------------------------------------------------------------------------

LR_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
ROUNDS_GRID = (5, 10, 20, 40, 80, 150, 300)


def lr_grid_rows(task, loss, depth, leaf, n_rounds, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS, grid=LR_GRID):
    """Testfehler je Lernrate bei fester Rundenzahl, gemittelt über mehrere Datensätze."""
    rows = []
    for lr in grid:
        errs = []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, sd)
            Xtr, ytr, Xte, yte = S.split(ds, task)
            ens = gb.fit(Xtr, ytr.astype(float), task, loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0)
            errs.append(_primary(_metrics(ens, Xte, yte, task), task))
        rows.append({"lr": lr, "test": float(np.mean(errs)), "sd": float(np.std(errs))})
    return rows


def rounds_grid_rows(task, loss, depth, leaf, lr, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS, grid=ROUNDS_GRID):
    """Testfehler je Rundenzahl bei fester Lernrate, gemittelt über mehrere Datensätze (zeigt Sättigung bzw. Überanpassung, je nach Lernrate)."""
    rows = []
    for n_rounds in grid:
        errs = []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, sd)
            Xtr, ytr, Xte, yte = S.split(ds, task)
            ens = gb.fit(Xtr, ytr.astype(float), task, loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0)
            errs.append(_primary(_metrics(ens, Xte, yte, task), task))
        rows.append({"n_rounds": n_rounds, "test": float(np.mean(errs)), "sd": float(np.std(errs))})
    return rows


# --- Verlustwahl bei Ausreißern (Regression) --------------------------------------------------------------------------------------------------------

OUTLIER_GRID = (0, 5, 10, 15)
OUTLIER_LOSSES = ("squared", "absolute", "huber")


def outlier_loss_rows(depth, leaf, n_rounds, lr, n, n_noise, seeds=C.SWEEP_SEEDS, grid=OUTLIER_GRID, losses=OUTLIER_LOSSES):
    """RMSE und MAE je Verlust und Ausreißer-Anteil, gemittelt über mehrere Datensätze (Ausreißer nur im Training)."""
    rows = []
    for pct in grid:
        row = {"outlier_pct": pct}
        for loss in losses:
            rmses, maes = [], []
            for sd in seeds:
                ds = S.generate_dataset(n, n_noise, 0, sd)
                Xtr, ytr, Xte, yte = S.split(ds, "reg")
                ytr = S.add_outliers(ytr, pct, sd)
                ens = gb.fit(Xtr, ytr, "reg", loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=1.0, seed=0)
                m = _metrics(ens, Xte, yte, "reg")
                rmses.append(m["rmse"]); maes.append(m["mae"])
            row[f"{loss}_rmse"] = float(np.mean(rmses))
            row[f"{loss}_mae"] = float(np.mean(maes))
        rows.append(row)
    return rows


# --- Wirkung der Teilstichprobe ---------------------------------------------------------------------------------------------------------------------

SUBSAMPLE_GRID = (100, 80, 60, 40, 20)


def subsample_rows(task, loss, depth, leaf, n_rounds, lr, n, n_noise, label_noise, seeds=C.SWEEP_SEEDS, grid=SUBSAMPLE_GRID):
    """Testfehler je Teilstichprobenanteil, gemittelt über mehrere Datensätze."""
    rows = []
    for sub in grid:
        errs = []
        for sd in seeds:
            ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, sd)
            Xtr, ytr, Xte, yte = S.split(ds, task)
            ens = gb.fit(Xtr, ytr.astype(float), task, loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=sub / 100.0, seed=0)
            errs.append(_primary(_metrics(ens, Xte, yte, task), task))
        rows.append({"subsample": sub, "test": float(np.mean(errs)), "sd": float(np.std(errs))})
    return rows


def subsample_timing(task, loss, depth, leaf, n_rounds, lr, n, n_noise, seed=C.DEFAULT_SEED, grid=SUBSAMPLE_GRID):
    """Rechenzeit je Teilstichprobenanteil (ein Lauf, `time.perf_counter`) - der verlässliche Teil des Effekts, unabhängig vom Testfehler."""
    import time
    ds = S.generate_dataset(n, n_noise, 0, seed)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    rows = []
    for sub in grid:
        t0 = time.perf_counter()
        gb.fit(Xtr, ytr.astype(float), task, loss, depth=depth, min_leaf=leaf, n_rounds=n_rounds, learning_rate=lr, subsample=sub / 100.0, seed=0)
        rows.append({"subsample": sub, "seconds": time.perf_counter() - t0})
    return rows
