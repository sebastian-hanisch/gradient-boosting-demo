"""Gradient Boosting - AdaBoost verallgemeinert auf jede Verlustfunktion - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Gradient Boosting - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Baumbasierten Linie der "Konzepte"-Reihe, zweites Stück des Boosting-Asts (nach AdaBoost): AdaBoost gewichtet Zeilen nach Fehlklassifikation um, Gradient Boosting wächst
stattdessen auf dem negativen Gradienten einer WÄHLBAREN Verlustfunktion - AdaBoost wird darin zu einem Sonderfall (exponentieller Verlust).
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import gb_algorithm as gb
import gb_constants as C
import gb_evaluation as ev
from gb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from gb_visualization import (
    build_importance,
    build_lr_curve,
    build_map,
    build_outlier_loss_chart,
    build_round_curve,
    build_round_tree,
    build_rounds_curve,
    build_subsample_chart,
    build_subsample_timing,
    feature_label,
)

st.set_page_config(page_title="Gradient Boosting – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ **Nur ein Schritt eingestellt** - das ist die Vorhersage eines einzelnen (mit Lernrate skalierten) Baums, noch kein Boosting.",
    "overfit": "⚠️ **Überanpassung:** der Testfehler liegt deutlich über dem Trainingsfehler.",
    "underfit": "⚠️ **Unteranpassung:** kaum besser als Raten - mehr Runden, größere Lernrate oder tiefere Bäume könnten helfen.",
    "ok": "✅ **Sieht vernünftig aus:** Training und Test liegen nicht weit auseinander.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=24)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _round_rows(*params):
    a = _analysis(*params)
    return ev.round_rows(a)


@st.cache_data(show_spinner=False, max_entries=6)
def _lr_grid(task, loss, depth, leaf, n_rounds, n, n_noise, label_noise):
    return ev.lr_grid_rows(task, loss, depth, leaf, n_rounds, n, n_noise, label_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _rounds_grid(task, loss, depth, leaf, lr, n, n_noise, label_noise):
    return ev.rounds_grid_rows(task, loss, depth, leaf, lr, n, n_noise, label_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _outlier_loss(depth, leaf, n_rounds, lr, n, n_noise):
    return ev.outlier_loss_rows(depth, leaf, n_rounds, lr, n, n_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _subsample(task, loss, depth, leaf, n_rounds, lr, n, n_noise, label_noise):
    return ev.subsample_rows(task, loss, depth, leaf, n_rounds, lr, n, n_noise, label_noise)


@st.cache_data(show_spinner=False, max_entries=6)
def _subsample_timing(task, loss, depth, leaf, n_rounds, lr, n, n_noise):
    return ev.subsample_timing(task, loss, depth, leaf, n_rounds, lr, n, n_noise)


st.title("📈🌳 Gradient Boosting – AdaBoost verallgemeinert auf jede Verlustfunktion")
st.markdown(
    """
**AdaBoost** (voriges Stück) gewichtet Trainingszeilen nach Fehlklassifikation um - das funktioniert nur für Klassenlabels und nur mit dem exponentiellen Verlust, den SAMME implizit benutzt.
**Gradient Boosting** (Friedman 2001) verallgemeinert die Idee: statt Gewichte anzupassen, wächst jede Runde einen Regressionsbaum auf den **Pseudo-Residuen** - dem negativen Gradienten einer
frei wählbaren Verlustfunktion an der aktuellen Vorhersage. Für quadratischen Verlust (Regression) sind die Pseudo-Residuen die gewöhnlichen Residuen; für Log-Loss (Klassifikation) sind es
Wahrscheinlichkeitsfehler; für den exponentiellen Verlust ergibt sich - bis auf die Schrittlänge - wieder AdaBoost. Jeder neue Baum wird mit einer **Lernrate** geschrumpft addiert: $F_m = F_{m-1} + \\eta \\cdot \\text{Baum}_m$.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Baumbasierten Linie der \"Konzepte\"-Reihe und zweites Stück des "
    "**Boosting-Asts** (nach AdaBoost) - **ein** Verfahren an einem wachsenden Beispiel. Das Verfahren geht auf Friedman (2001, \"Greedy Function Approximation\") zurück; alle Lieferungen, Merkmale und Zahlen "
    "dieser Demo sind erzeugt und gemessen - keine echten Daten. Der Baumkern (`gb_tree.py`) ist wortgleich aus cart-demo übernommen (Regressionsbäume, Varianz-Kriterium - dieselbe Mathematik, ob die äußere "
    "Aufgabe Klassifikation oder Regression ist); scikit-learn kommt nur in den Tests als Gegenprobe vor."
)
st.caption(
    "**Beide Aufgaben:** anders als AdaBoost (nur Klassifikation) deckt Gradient Boosting hier auch Regression ab - derselbe Baumkern, nur die Verlustfunktion (und damit Pseudo-Residuum und Blattwert) wechselt."
)
st.caption(
    "**Bezug zu OR:** die vorhergesagte Lieferdauer (Regression) ist Eingabe der Tourenplanung (Zeitfenster, VRP); eine robuste Verlustfunktion (siehe Experimente unten) verhindert, dass einzelne grobe "
    "Messfehler im Training die Vorhersage für alle anderen Lieferungen verzerren."
)

with st.expander("So funktioniert Gradient Boosting", expanded=True):
    st.markdown(
        r"""
1. **Start:** die beste konstante Vorhersage $F_0$ (Mittelwert bzw. Median für Regression, (halbe) Log-Odds der Basisrate für Klassifikation).
2. **Jede Runde $m$:** die Pseudo-Residuen $r_i = -\partial L(y_i, F_{m-1}(x_i))/\partial F$ ausrechnen (der negative Gradient der gewählten Verlustfunktion) und einen Regressionsbaum darauf wachsen
   (derselbe Kern wie in cart-demo, Varianz-Kriterium - Pseudo-Residuen sind immer Zahlen).
3. **Blattwerte nachtragen:** in jedem Blatt wird der beim Wachsen berechnete Mittelwert der Pseudo-Residuen durch den für den Verlust **optimalen** konstanten Wert ersetzt (bei quadratischem Verlust ist das
   dasselbe; bei Log-Loss ein Newton-Schritt; bei absolutem/Huber-Verlust der Median bzw. eine robuste Lageschätzung).
4. **Update:** $F_m(x) = F_{m-1}(x) + \eta \cdot \text{Baum}_m(x)$ - die Lernrate $\eta$ bremst jeden einzelnen Schritt.
5. **Teilstichprobe (optional):** jede Runde nur einen zufälligen Teil der Trainingszeilen benutzen (*stochastic gradient boosting*) - schneller je Runde, mit unterschiedlicher Wirkung auf den Testfehler (gemessen unten).
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k],
                        help="Klassifikation: kommt die Lieferung zu spät? Regression: wie lange dauert sie? Dieselben Lieferungen und Merkmale, nur das Ziel (und damit die möglichen Verlustfunktionen) wechselt.")
    loss_options = C.LOSSES[task]
    if st.session_state["loss_select"] not in loss_options:
        st.session_state["loss_select"] = C.DEFAULT_LOSS[task]
    loss = st.selectbox("Verlustfunktion", loss_options, key="loss_select", format_func=lambda k: C.LOSS_LABELS[k],
                        help="Bestimmt Pseudo-Residuum und optimalen Blattwert jeder Runde. Exponentiell (nur Klassifikation) ist - bei Tiefe 1 und Lernrate 1 - bis auf die Schrittlänge AdaBoost.")
    depth = st.slider("Tiefe der Bäume", *bounds("depth_slider"), key="depth_slider", help="Tiefe jedes einzelnen Runden-Baums. Höhere Tiefe macht jeden Baum stärker (mehr Wechselwirkungen), aber auch überanpassungsfreudiger.")
    leaf = st.slider("Mindestgröße eines Blatts", *bounds("leaf_slider"), key="leaf_slider")
    n_rounds = st.slider("Zahl der Runden", *bounds("n_rounds_slider"), key="n_rounds_slider", help="Wie viele Bäume nacheinander gewachsen werden.")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=0.01, format="%.2f",
                   help="Skaliert jeden neuen Baum vor dem Addieren. Kleiner = kleinere Schritte, mehr Runden nötig, aber meist besserer Testfehler (Kompromiss, siehe Experimente).")
    subsample = st.slider("Teilstichprobe je Runde [%]", *bounds("subsample_slider"), key="subsample_slider", format="%.0f%%",
                          help="Anteil der Trainingszeilen, die jede Runde sieht (zufällig neu gezogen). 100 % = jede Runde sieht alle Zeilen.")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100)
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider")
    if task == "class":
        seed_widget("label_noise_slider")
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider",
                                help="Anteil vertauschter Trainingsetiketten; der Test bleibt sauber.")
        st.session_state["_label_noise_kept"] = label_noise
        outlier = int(st.session_state.get("_outlier_kept", C.DEFAULT_OUTLIER))
    else:
        seed_widget("outlier_slider")
        outlier = st.slider("Grobe Ausreißer im Training [%]", *bounds("outlier_slider"), key="outlier_slider",
                            help="Anteil der Trainingszeilen mit einem großen, zufälligen Schock auf die Dauer (60 bis 120 Minuten) - grobe Messfehler oder Sonderfahrten. Der Test bleibt sauber.")
        st.session_state["_outlier_kept"] = outlier
        label_noise = int(st.session_state.get("_label_noise_kept", C.DEFAULT_LABEL_NOISE))
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

base_params = (task, loss, int(depth), int(leaf), int(n_rounds), float(lr), int(subsample))
data_params = (int(n), int(n_noise), int(label_noise), int(outlier), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
names = ds.names
n_feat = len(names)
n_test = len(ds.test)
n_trees = len(a.ensemble.trees)

with st.sidebar:
    st.markdown("**Ansicht**")
    for key, default in (("map_x_select", C.DEFAULT_MAP[0]), ("map_y_select", C.DEFAULT_MAP[1])):
        if st.session_state[key] >= n_feat:
            st.session_state[key] = default
    fx = st.selectbox("Karte: waagerecht", range(n_feat), key="map_x_select", format_func=lambda f: feature_label(names, f))
    fy = st.selectbox("Karte: senkrecht", range(n_feat), key="map_y_select", format_func=lambda f: feature_label(names, f))
    if st.session_state.get("sample_slider", 0) > n_test - 1:
        st.session_state["sample_slider"] = 0
    sample_idx = st.slider("Testlieferung", 0, n_test - 1, 0, key="sample_slider")
sync_query_params({"task_select": task, "loss_select": loss, "depth_slider": int(depth), "leaf_slider": int(leaf), "n_rounds_slider": int(n_rounds), "lr_slider": float(lr),
                   "subsample_slider": int(subsample), "n_slider": int(n), "n_noise_slider": int(n_noise), "label_noise_slider": int(label_noise), "outlier_slider": int(outlier),
                   "seed_input": int(seed), "map_x_select": int(fx), "map_y_select": int(fy)})

view_key = (base_params, data_params)
if st.session_state.get("gb_owner") != view_key:
    st.session_state["gb_owner"] = view_key
    st.session_state["gb_step"] = n_trees

# --- Gradient Boosting in Aktion --------------------------------------------------------------------------------------------------------------------

st.markdown("## 📈 Gradient Boosting in Aktion")
st.caption("Runde für Runde: links der Baum dieser Runde (Blattwerte = Korrekturen, nicht die Endvorhersage), rechts die Vorhersage des Ensembles bis dahin.")
if n_trees > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Runden", 1, n_trees, key="gb_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur eine Runde eingestellt - mehr Runden in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()
Xte_full, yte_full = ds.X[ds.test], ds.y_true[ds.test] if task == "class" else ds.y_reg[ds.test]
sample_x = Xte_full[sample_idx]
sample_y = yte_full[sample_idx]


def _render(current):
    tree = a.ensemble.trees[current - 1]
    with view_slot.container():
        c1, c2 = st.columns([2, 3])
        with c1:
            st.plotly_chart(build_round_tree(tree, names), width="stretch", key=f"tree_chart_{current}")
            st.caption(f"Runde {current}: Blattwerte sind Korrekturen (mit der Lernrate {lr:.2f} skaliert addiert), nicht die Endvorhersage.")
        with c2:
            st.plotly_chart(build_map(a.ensemble, ds, task, fx, fy, upto=current, sample=sample_x), width="stretch", key=f"map_chart_{current}")
        pred_here = gb.predict_value(a.ensemble, sample_x.reshape(1, -1), upto=current)[0]
        pred_text = f"{pred_here:.0%} zu spät" if task == "class" else f"{pred_here:.1f} min"
        truth_text = ("zu spät" if sample_y == 1 else "pünktlich") if task == "class" else f"{sample_y:.1f} min"
        st.markdown(f"**Testlieferung {sample_idx}:** Vorhersage nach {current} Runden = **{pred_text}**; tatsächlich: **{truth_text}**.")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, n_trees, min(12, n_trees))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = n_trees
else:
    _render(step)

st.markdown("---")

# --- Was das Ensemble gelernt hat -------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was das Ensemble gelernt hat – und wie gut es auf neuen Lieferungen ist")
rrows = _round_rows(*base_params, *data_params)
best = ev.best_round(rrows)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runden", n_trees)
m2.metric("Trainingsfehler" if task == "class" else "Trainings-RMSE", _err(task, ev.primary(a.train, task)))
m3.metric("Testfehler" if task == "class" else "Test-RMSE", _err(task, ev.primary(a.test, task)), delta=f"ohne Modell: {_err(task, a.baseline)}", delta_color="off")
m4.metric("Bester Testfehler" if task == "class" else "Bester Test-RMSE", _err(task, best["test"]), delta=f"bei Runde {best['k']}", delta_color="off")
st.markdown(VERDICT_TEXT[a.verdict])
if task == "reg" and outlier > 0:
    st.caption("⚠️ Der Trainingsfehler ist hier gegen die **verrauschten** Trainingswerte gemessen (inklusive der Ausreißer-Schocks) - ein robuster Verlust (Huber, Absolut) ignoriert diese Ausreißer bewusst, "
               "daher wirkt der Trainingsfehler höher, obwohl der Testfehler (gegen die sauberen Werte) gut ist.")

st.markdown("**Testfehler gegen die Rundenzahl (Frühstopp)**")
st.plotly_chart(build_round_curve(rrows, task, a.baseline, n_trees, best["k"]), width="stretch", key="round_chart")
st.caption(f"Der Trainingsfehler sinkt fast durchgehend; der Testfehler erreicht sein Minimum bei Runde {best['k']} ({_err(task, best['test'])}) und kann danach wieder steigen (Überanpassung) - "
           "das ist die Grundlage für Frühstopp: die Rundenzahl anhand eines zurückgehaltenen Testfehlers wählen, statt eine feste Zahl zu erzwingen.")

st.markdown("**Wichtigkeit je Merkmal**")
st.plotly_chart(build_importance(names, a.imp), width="stretch", key="importance_chart")
st.caption("Gemittelt über alle Bäume des Ensembles: Summe der gewichteten Varianzabnahmen je Merkmal, auf 1 normiert.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Lernrate × Rundenzahl")
if st.button("Testfehler gegen Lernrate und Rundenzahl messen (dauert einen Moment)", key="lr_start"):
    st.session_state["lr_on"] = True
if st.session_state.get("lr_on"):
    with st.spinner("Trainiere über mehrere Lernraten und Rundenzahlen auf fünf Datensätzen ..."):
        lgrid = _lr_grid(task, loss, int(depth), int(leaf), 150, int(n), int(n_noise), int(label_noise))
        rgrid_small = _rounds_grid(task, loss, int(depth), int(leaf), 0.1, int(n), int(n_noise), int(label_noise))
        rgrid_large = _rounds_grid(task, loss, int(depth), int(leaf), 1.0, int(n), int(n_noise), int(label_noise))
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_lr_curve(lgrid, task), width="stretch", key="lr_chart")
    c2.plotly_chart(build_rounds_curve([("Lernrate 0.1", rgrid_small, "#1f77b4"), ("Lernrate 1.0", rgrid_large, "#d62728")], task), width="stretch", key="rounds_chart")
    lo = min(lgrid, key=lambda r: r["test"])
    st.caption(f"Links: Testfehler bei 150 Runden gegen die Lernrate (Mittel über fünf Datensätze) - Minimum bei Lernrate {lo['lr']}. Rechts: Testfehler gegen die Rundenzahl für zwei Lernraten - bei Lernrate 0.1 "
               f"sinkt der Testfehler mit mehr Runden weiter (oder bleibt zumindest stabil); bei Lernrate 1.0 steigt er nach wenigen Runden wieder (Überanpassung) - der klassische Kompromiss aus Friedmans Originalarbeit: "
               "kleine Schritte brauchen mehr Runden, sind aber nicht so schnell überangepasst.")

st.markdown("---")

if task == "reg":
    st.subheader("🔬 Verlustwahl bei Ausreißern")
    if st.button("RMSE und MAE je Verlustfunktion gegen den Ausreißer-Anteil messen (dauert einen Moment)", key="out_start"):
        st.session_state["out_on"] = True
    if st.session_state.get("out_on"):
        with st.spinner("Trainiere drei Verlustfunktionen über vier Ausreißer-Stufen auf fünf Datensätzen ..."):
            orows = _outlier_loss(int(depth), int(leaf), int(n_rounds), float(lr), int(n), int(n_noise))
        c1, c2 = st.columns(2)
        c1.plotly_chart(build_outlier_loss_chart(orows, "rmse"), width="stretch", key="outlier_rmse_chart")
        c2.plotly_chart(build_outlier_loss_chart(orows, "mae"), width="stretch", key="outlier_mae_chart")
        r0, r1 = orows[0], orows[-1]
        st.caption(f"Ohne Ausreißer (0 %) liegt Quadratisch knapp vorn (RMSE {r0['squared_rmse']:.1f} gegen {r0['huber_rmse']:.1f} bei Huber) - die übliche Annahme (Gauß-Rauschen) stimmt dann am besten. "
                   f"Mit {r1['outlier_pct']} % groben Ausreißern im Training dreht es sich: Quadratisch verschlechtert sich am stärksten (RMSE {r0['squared_rmse']:.1f} → {r1['squared_rmse']:.1f}), Huber "
                   f"und Absolut bleiben deutlich stabiler ({r0['huber_rmse']:.1f} → {r1['huber_rmse']:.1f} bzw. {r0['absolute_rmse']:.1f} → {r1['absolute_rmse']:.1f}) - beide gewichten große Residuen "
                   "nur linear statt quadratisch und lassen sich von einzelnen groben Fehlern weniger beeindrucken.")
    st.markdown("---")

st.subheader("🔬 Wirkung der Teilstichprobe")
if st.button("Testfehler und Rechenzeit gegen die Teilstichprobe messen (dauert einen Moment)", key="sub_start"):
    st.session_state["sub_on"] = True
if st.session_state.get("sub_on"):
    with st.spinner("Trainiere über fünf Teilstichprobenanteile auf fünf Datensätzen, dazu eine Zeitmessung ..."):
        subrows = _subsample(task, loss, int(depth), int(leaf), int(n_rounds), float(lr), int(n), int(n_noise), int(label_noise))
        timerows = _subsample_timing(task, loss, int(depth), int(leaf), 100, float(lr), int(n), int(n_noise))
    c1, c2 = st.columns(2)
    c1.plotly_chart(build_subsample_chart(subrows, task), width="stretch", key="subsample_chart")
    c2.plotly_chart(build_subsample_timing(timerows), width="stretch", key="subsample_timing_chart")
    fast, full = timerows[-1], timerows[0]
    st.caption(f"Rechenzeit sinkt ungefähr proportional zum Teilstichprobenanteil ({full['seconds']:.2f} s bei {full['subsample']} % gegen {fast['seconds']:.2f} s bei {fast['subsample']} %, 100 Runden) - "
               "jede Runde wächst nur auf einem Bruchteil der Zeilen. Die Wirkung auf den Testfehler ist dagegen in diesem Datensatz **uneinheitlich**: mal etwas besser, mal etwas schlechter als 100 % "
               "(siehe Streuung im Diagramm) - anders als die klare Lernrate-Rundenzahl-Kompromisslinie oben ist die Regularisierung durch Teilstichproben hier kein verlässlicher Hebel, nur ein Geschwindigkeitshebel.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Verlustfunktion muss zum Rauschen passen** | Quadratischer Verlust nimmt implizit Gauß-Rauschen an; grobe Ausreißer im Training verzerren ihn stark stärker als robuste Verluste (gemessen oben). | Huber- oder absoluter Verlust |
| **Zu große Lernrate ohne genug Bremsung** | Trainingsfehler fällt schnell auf 0, der Testfehler steigt nach wenigen Runden wieder (gemessen oben). | kleinere Lernrate, Frühstopp anhand des Testfehlers |
| **Feste Rundenzahl statt Frühstopp** | Ohne einen zurückgehaltenen Testfehler weiß man nicht, wann man aufhören sollte - zu wenige Runden unteranpassen, zu viele überanpassen (siehe Rundenkurve oben). | Testfehler laufend mitmessen, bei der besten Runde aufhören |
| **Teilstichproben als verlässliche Regularisierung** | In diesem Datensatz ändert die Teilstichprobe den Testfehler nur uneinheitlich - verlässlich ist nur der Geschwindigkeitsgewinn (gemessen oben). | größere Datensätze, wo der Regularisierungseffekt klarer greift |
| **Bäume sind einfache, achsenparallele Splits** | Wie jeder Baum in dieser Linie liefert auch Gradient Boosting eine Stufenfunktion je Baum; erst die Summe vieler Bäume nähert sich einer glatten Fläche an. | mehr Runden, größere Tiefe |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gradient Boosting** (Friedman 2001). Trainingsdaten $(x_i,y_i)_{i=1}^n$, eine differenzierbare Verlustfunktion $L(y,F)$, Startwert $F_0(x) = \arg\min_c \sum_i L(y_i, c)$.

**Runde $m$:** Pseudo-Residuen $r_{im} = -\left[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$, einen Regressionsbaum $h_m$ auf $(x_i, r_{im})$ wachsen, dann je Blatt $R_{jm}$ den optimalen Wert
$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$ nachtragen (Zeile für Zeile via Newton-Schritt bzw. geschlossener Lösung, je nach Verlust).
Update: $F_m(x) = F_{m-1}(x) + \eta \sum_j \gamma_{jm}\,\mathbb 1[x \in R_{jm}]$.

**Verlustfunktionen dieser Demo:**

| Verlust | $-\partial L/\partial F$ | optimaler Blattwert |
|---|---|---|
| Quadratisch (Regression) | $y - F$ | Mittelwert des Residuums |
| Absolut (Regression) | $\text{sign}(y-F)$ | Median des Residuums |
| Huber (Regression) | $\text{clip}(y-F,\,-\delta,\,\delta)$ | robuste Lageschätzung um den Median ($\delta$ = 0.9-Quantil von $\lvert y-F \rvert$, je Runde neu) |
| Log-Loss (Klassifikation) | $y - \sigma(F)$ | $\sum(y-p)\,/\,\sum p(1-p)$ (Newton-Schritt) |
| Exponentiell (Klassifikation) | $\tilde y\,e^{-\tilde y F}$, $\tilde y=2y-1$ | $\tfrac12\ln\!\big(\sum_{\tilde y=1} e^{-F} \,/\, \sum_{\tilde y=-1} e^{F}\big)$ |

Bei Tiefe 1, Lernrate 1 und exponentiellem Verlust ist der Blattwert exakt die halbe AdaBoost-Stimmgewichtsformel $\alpha_m$ - die Vorhersagen stimmen mit adaboost-demo bis auf einen Unterschied
im Baumkern (Varianz- statt Gini-Kriterium für die Split-Suche) weitgehend überein (gemessen: 92.7 % Übereinstimmung, Mittel über fünf Datensätze).

Implementiert in `gb_tree.py` (Baumkern, aus cart-demo übernommen, plus `set_leaf_values`), `gb_algorithm.py` (Verlustfunktionen, Fit, Vorhersage), `gb_evaluation.py` (Analyse, Rundenkurve,
Lernrate-Rundenzahl-, Ausreißer- und Teilstichproben-Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
