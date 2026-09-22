"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

# Merkmale der Lieferungen: (Name, Einheit) - wortgleich aus cart-demo
FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre")]
N_BASE = len(FEATURES)

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"

LOSSES = {"class": ("logloss", "exponential"), "reg": ("squared", "absolute", "huber")}
LOSS_LABELS = {"squared": "Quadratisch (kleinste Fehlerquadrate)", "absolute": "Absolut (mittlerer Betrag)", "huber": "Huber (quadratisch nahe 0, sonst absolut)",
               "logloss": "Log-Loss (logistisch)", "exponential": "Exponentiell (= AdaBoost)"}
DEFAULT_LOSS = {"class": "logloss", "reg": "squared"}
HUBER_DELTA = 1.0                    # in Einheiten der standardisierten Residuen (siehe gb_algorithm._huber_delta)

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0
OUTLIER_MIN, OUTLIER_MAX, DEFAULT_OUTLIER = 0, 15, 0          # Prozent grobe Ausreißer in der Trainings-Dauer (nur Regression)
TEST_SHARE = 0.3
DEFAULT_SEED = 7

OVERFIT_GAP_CLASS = 0.08      # Testfehler minus Trainingsfehler (Klassifikation) ab hier: Überanpassung
OVERFIT_RATIO_REG = 1.6        # Testfehler / Trainingsfehler (Regression) ab hier: Überanpassung
UNDERFIT_SHARE = 0.75          # Testfehler über diesem Anteil des Rate-Fehlers: das Modell ist zu einfach

SWEEP_SEEDS = tuple(range(100000, 100005))

# Gradient-Boosting-eigene Regler
N_ROUNDS_MIN, N_ROUNDS_MAX, DEFAULT_N_ROUNDS = 1, 300, 60
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 4, 2
LEAF_MIN, LEAF_MAX, DEFAULT_LEAF = 1, 20, 5
LR_MIN, LR_MAX, DEFAULT_LR = 0.02, 1.0, 0.1
SUBSAMPLE_MIN, SUBSAMPLE_MAX, DEFAULT_SUBSAMPLE = 20, 100, 100     # Prozent der Trainingszeilen je Runde

DEFAULT_MAP = (0, 3)          # Kartenausschnitt: Distanz x Verkehr

COLORS = {"train": "#1f77b4", "test": "#d62728", "squared": "#1f77b4", "absolute": "#2ca02c", "huber": "#ff7f0e", "exponential": "#9467bd", "logloss": "#1f77b4"}

PRESETS = {
    "🌳 Standard": dict(task="class", loss="logloss", depth=2, leaf=5, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🪓 Ein Schritt (kein Boosting)": dict(task="class", loss="logloss", depth=1, leaf=5, n_rounds=1, lr=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🚀 Zu große Lernrate": dict(task="class", loss="logloss", depth=2, leaf=5, n_rounds=150, lr=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🐌 Kleine Lernrate, viele Runden": dict(task="class", loss="logloss", depth=2, leaf=5, n_rounds=250, lr=0.05, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "⚔️ Exponentiell (= AdaBoost)": dict(task="class", loss="exponential", depth=1, leaf=1, n_rounds=30, lr=1.0, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
    "🛡️ Robuster Verlust bei Ausreißern": dict(task="reg", loss="huber", depth=2, leaf=5, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, subsample=100, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=10, seed=DEFAULT_SEED, fx=0, fy=3),
    "🎲 Teilstichprobe": dict(task="class", loss="logloss", depth=2, leaf=5, n_rounds=100, lr=0.1, subsample=40, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, outlier=0, seed=DEFAULT_SEED, fx=0, fy=3),
}
PRESET_HELP = {
    "🌳 Standard": "Klassifikation, Log-Loss, 60 Runden Tiefe-2-Bäume, Lernrate 0.1: Trainingsfehler 10.0 %, Testfehler 15.6 % (Raten: 46.4 %). Jede Runde wächst auf den Pseudo-Residuen der vorigen Vorhersage.",
    "🪓 Ein Schritt (kein Boosting)": "Derselbe Tiefe-1-Baum wie CART, aber als ein einzelner Gradient-Boosting-Schritt (Lernrate 1, keine weiteren Runden): Testfehler 24.4 % - kaum besser als Raten. Erst das Wiederholen über viele Runden macht aus Gradient Boosting etwas Brauchbares.",
    "🚀 Zu große Lernrate": "Lernrate 1.0 statt 0.1, 150 Runden: Trainingsfehler fällt auf 0 %, aber der Testfehler steigt auf 17.8 % (Standard: 15.6 %) - das Modell passt sich zu schnell den einzelnen Zeilen des Trainings an.",
    "🐌 Kleine Lernrate, viele Runden": "Lernrate 0.05, dafür 250 statt 60 Runden: Testfehler 15.0 %, minimal besser als der Standard (15.6 %), bei höherem Rechenaufwand - der klassische Kompromiss aus Friedmans Originalarbeit (kleine Schritte, viele davon).",
    "⚔️ Exponentiell (= AdaBoost)": "Tiefe-1-Bäume, Lernrate 1, exponentieller Verlust: Testfehler 20.6 %. Mit denselben Einstellungen (Stümpfe, keine Lernraten-Bremse) stimmen die Vorhersagen zu 92.7 % (Mittel über 5 Datensätze) mit adaboost-demo überein - der Rest kommt daher, dass die Bäume hier auf den Pseudo-Residuen wachsen (Varianz-Kriterium) statt gewichtet nach Gini, auch wenn die Blattgewichts-Formel identisch ist.",
    "🛡️ Robuster Verlust bei Ausreißern": "Regression mit 10 % groben Ausreißern im Training: Huber-Verlust erreicht RMSE 10.6 (Quadratisch: 12.3) - er wiegt große Residuen nur linear statt quadratisch. Ohne Ausreißer ist es umgekehrt (siehe Experimente-Abschnitt): dort gewinnt Quadratisch knapp.",
    "🎲 Teilstichprobe": "Nur 40 % der Trainingszeilen je Runde: hier sogar etwas genauer (Testfehler 13.3 % gegen 15.0 % mit allen Zeilen) und schneller gerechnet (0.15 s gegen 0.26 s für 100 Runden) - der Effekt auf die Genauigkeit ist aber uneinheitlich (siehe Experimente-Abschnitt), der auf die Rechenzeit nicht.",
}
