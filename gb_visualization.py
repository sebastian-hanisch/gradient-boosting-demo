"""Plotly-Darstellungen: Entscheidungsgrenze/Regressionsfläche des Ensembles, ein einzelner Runden-Baum (Pseudo-Residuen), Fehlerkurve gegen Runden, Lernrate x Rundenzahl, Verlustwahl bei Ausreißern,
Teilstichprobe, Wichtigkeit. Alle Achsen sind gesperrt (Touch-Scrollen)."""

import numpy as np
import plotly.graph_objects as go

import gb_algorithm as gb
import gb_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]        # pünktlich (grün) -> zu spät (rot)
REG_SCALE = "Viridis"
LOSS_COLORS = {"squared": C.COLORS["squared"], "absolute": C.COLORS["absolute"], "huber": C.COLORS["huber"]}


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "")
    return f"{names[f]} [{unit}]" if unit else names[f]


# --- Karte: Entscheidungsgrenze (Klassifikation) bzw. Regressionsfläche -----------------------------------------------------------------------------

def build_map(ensemble, ds, task, fx, fy, upto=None, sample=None, height=430):
    """Vorhersage des Ensembles über zwei Merkmale (die übrigen auf ihrem Median im Training) mit den Trainingspunkten darüber; `upto` zeigt den Stand nach `upto` Runden."""
    Xtr = ds.X[ds.train]
    ytr = ds.y(task)[ds.train]
    med = np.median(Xtr, axis=0)
    gx = np.round(np.linspace(Xtr[:, fx].min(), Xtr[:, fx].max(), 60), 4)
    gy = np.round(np.linspace(Xtr[:, fy].min(), Xtr[:, fy].max(), 60), 4)
    XX, YY = np.meshgrid(gx, gy)
    grid = np.tile(med, (XX.size, 1))
    grid[:, fx], grid[:, fy] = XX.ravel(), YY.ravel()
    z = np.round(gb.predict_value(ensemble, grid, upto), 3).reshape(XX.shape)
    vmin, vmax = (0.0, 1.0) if task == "class" else (float(np.min(ytr)), float(np.max(ytr)))
    scale = CLASS_SCALE if task == "class" else REG_SCALE
    fig = go.Figure(go.Heatmap(x=gx, y=gy, z=z, colorscale=scale, zmin=vmin, zmax=vmax, opacity=0.55, showscale=False, hovertemplate="%{z:.2f}<extra></extra>"))
    if task == "class":
        fig.add_trace(go.Contour(x=gx, y=gy, z=z, contours=dict(start=0.5, end=0.5, size=1, coloring="none"), line=dict(color="#111111", width=2), showscale=False, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=np.round(Xtr[:, fx], 4), y=np.round(Xtr[:, fy], 4), mode="markers", marker=dict(size=5, color=ytr, colorscale=scale, cmin=vmin, cmax=vmax, line=dict(color="#333333", width=0.5)),
                             hovertemplate="%{x:.3g} / %{y:.3g}<extra></extra>", showlegend=False))
    if sample is not None:
        fig.add_trace(go.Scatter(x=[sample[fx]], y=[sample[fy]], mode="markers", marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#111111", width=2)), hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=feature_label(ds.names, fx))
    fig.update_yaxes(title=feature_label(ds.names, fy))
    return lock_axes(fig, height)


# --- Ein einzelner Runden-Baum (wächst auf den Pseudo-Residuen) -------------------------------------------------------------------------------------

def tree_layout(tree):
    x = np.zeros(tree.n_nodes)
    counter = 0
    stack = [(0, False)]
    while stack:
        t, done = stack.pop()
        if tree.feature[t] < 0:
            x[t] = counter
            counter += 1
        elif done:
            x[t] = (x[tree.left[t]] + x[tree.right[t]]) / 2.0
        else:
            stack += [(t, True), (int(tree.right[t]), False), (int(tree.left[t]), False)]
    return x, -tree.depth.astype(float)


def build_round_tree(tree, names, height=280):
    """Der Baum einer Runde: Blattwerte sind die verlustoptimalen Korrekturen (nicht Wahrscheinlichkeiten oder Minuten direkt) - mit Lernrate multipliziert zur Vorhersage addiert."""
    x, y = tree_layout(tree)
    inner = tree.feature >= 0
    fig = go.Figure()
    ex, ey = [], []
    for t in np.nonzero(inner)[0]:
        for c in (tree.left[t], tree.right[t]):
            ex += [x[t], x[c], None]
            ey += [y[t], y[c], None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    vmax = float(np.max(np.abs(tree.value[~inner]))) if (~inner).any() else 1.0
    size = 12 + 8 * tree.depth.max() - 5 * tree.depth
    color = np.where(inner, np.nan, tree.value)
    fig.add_trace(go.Scatter(x=x[~inner], y=y[~inner], mode="markers+text", text=[f"{tree.value[t]:+.3g}" for t in np.nonzero(~inner)[0]], textposition="bottom center", textfont=dict(size=9),
                             marker=dict(size=size[~inner], color=color[~inner], colorscale="RdBu_r", cmin=-vmax, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[f"Blatt: Korrektur {tree.value[t]:+.4g}, n={tree.n[t]}" for t in np.nonzero(~inner)[0]], hoverinfo="text", showlegend=False))
    labels = [f"{names[tree.feature[t]]} ≤ {tree.threshold[t]:.3g}" for t in np.nonzero(inner)[0]]
    fig.add_trace(go.Scatter(x=x[inner], y=y[inner], mode="markers+text", text=labels if tree.n_nodes <= 15 else "", textposition="top center", textfont=dict(size=9),
                             marker=dict(size=size[inner], color="#ffffff", line=dict(color="#555555", width=1)),
                             hovertext=[f"{names[tree.feature[t]]} ≤ {tree.threshold[t]:.4g}?" for t in np.nonzero(inner)[0]], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Fehlerkurve gegen Runden (Frühstopp) -----------------------------------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_round_curve(rows, task, baseline, current_k, best_k=None, height=340):
    k = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[r["train"] for r in rows], mode="lines+markers", name="Training", line=dict(color=C.COLORS["train"])))
    fig.add_trace(go.Scatter(x=k, y=[r["test"] for r in rows], mode="lines+markers", name="Test", line=dict(color=C.COLORS["test"])))
    fig.add_hline(y=baseline, line=dict(color="#888888", dash="dash"), annotation_text="ohne Modell (Raten)", annotation_position="top right")
    fig.add_vline(x=current_k, line=dict(color="#111111", dash="dot"))
    if best_k is not None and best_k != current_k:
        fig.add_vline(x=best_k, line=dict(color="#2ca02c", dash="dot"), annotation_text="bester Testfehler", annotation_position="bottom left")
    fig.update_xaxes(title="Runden", type="log" if k[-1] > 30 else "linear")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Lernrate x Rundenzahl ---------------------------------------------------------------------------------------------------------------------------

def build_lr_curve(rows, task, height=320):
    lr = [r["lr"] for r in rows]
    fig = go.Figure(go.Scatter(x=lr, y=[r["test"] for r in rows], mode="lines+markers", line=dict(color="#1f77b4"),
                               error_y=dict(type="data", array=[r["sd"] for r in rows], visible=True, color="#9aa0a6"),
                               hovertemplate="Lernrate %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Lernrate", type="log")
    _error_axis(fig, task)
    return lock_axes(fig, height, showlegend=False)


def build_rounds_curve(rows_by_lr, task, height=340):
    """rows_by_lr: [(label, rows, color)] - Testfehler gegen Rundenzahl für mehrere Lernraten."""
    fig = go.Figure()
    for label, rows, color in rows_by_lr:
        n = [r["n_rounds"] for r in rows]
        fig.add_trace(go.Scatter(x=n, y=[r["test"] for r in rows], mode="lines+markers", name=label, line=dict(color=color),
                                 error_y=dict(type="data", array=[r["sd"] for r in rows], visible=True, color=color, thickness=1)))
    fig.update_xaxes(title="Runden", type="log")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Verlustwahl bei Ausreißern (Regression) --------------------------------------------------------------------------------------------------------

def build_outlier_loss_chart(rows, metric="rmse", height=340):
    op = [r["outlier_pct"] for r in rows]
    fig = go.Figure()
    for loss, label in (("squared", "Quadratisch"), ("absolute", "Absolut"), ("huber", "Huber")):
        fig.add_trace(go.Scatter(x=op, y=[r[f"{loss}_{metric}"] for r in rows], mode="lines+markers", name=label, line=dict(color=LOSS_COLORS[loss])))
    fig.update_xaxes(title="Grobe Ausreißer im Training [%]")
    fig.update_yaxes(title="Test-RMSE [min]" if metric == "rmse" else "Test-MAE [min]", rangemode="tozero")
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Teilstichprobe ----------------------------------------------------------------------------------------------------------------------------------

def build_subsample_chart(rows, task, height=320):
    sub = [r["subsample"] for r in rows]
    fig = go.Figure(go.Scatter(x=sub, y=[r["test"] for r in rows], mode="lines+markers", line=dict(color="#9467bd"),
                               error_y=dict(type="data", array=[r["sd"] for r in rows], visible=True, color="#9aa0a6"),
                               hovertemplate="%{x} %%: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title="Trainingszeilen je Runde [%]")
    _error_axis(fig, task)
    return lock_axes(fig, height, showlegend=False)


def build_subsample_timing(rows, height=260):
    sub = [r["subsample"] for r in rows]
    fig = go.Figure(go.Bar(x=[f"{s} %" for s in sub], y=[r["seconds"] for r in rows], marker_color="#9467bd", text=[f"{r['seconds']:.2f} s" for r in rows], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Rechenzeit [s]", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False)


# --- Wichtigkeit ---------------------------------------------------------------------------------------------------------------------------------------

def build_importance(names, imp, height=330):
    order = np.argsort(-imp, kind="stable")
    colors = [C.COLORS.get("noise", "#ff7f0e") if f >= C.N_BASE else "#1f77b4" for f in order]
    fig = go.Figure(go.Bar(x=imp[order], y=[names[f] for f in order], orientation="h", marker_color=colors, text=[f"{imp[f]:.1%}" for f in order], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (Anteil an der Abnahme der Varianz)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=60, t=30, b=10))
