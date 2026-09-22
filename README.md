# Gradient Boosting – AdaBoost verallgemeinert auf jede Verlustfunktion – Streamlit-Demo

Sechstes Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning" und das **zweite Stück des Boosting-Asts**
(nach AdaBoost): anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Gradient Boosting** (Friedman 2001,
"Greedy Function Approximation") – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/.../adaboost-demo, aber – anders als AdaBoost – **beide Aufgaben** (Klassifikation und Regression), da Gradient Boosting nicht auf Klassenlabels beschränkt ist.
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, scikit-learn nur in den Tests als Gegenprobe.

**Bezug zu OR:** die vorhergesagte Lieferdauer (Regression) ist Eingabe der Tourenplanung (Zeitfenster, VRP); eine robuste Verlustfunktion verhindert, dass einzelne grobe Messfehler im Training
die Vorhersage für alle anderen Lieferungen verzerren.

**Einordnung in die Reihe:** AdaBoost (voriges Stück) gewichtet Trainingszeilen nach Fehlklassifikation um – das funktioniert nur für Klassenlabels und implizit nur mit dem exponentiellen Verlust.
Gradient Boosting verallgemeinert die Idee: jede Runde wächst einen Regressionsbaum auf den **Pseudo-Residuen** (dem negativen Gradienten einer frei wählbaren Verlustfunktion) und trägt danach
den verlustoptimalen Blattwert nach. AdaBoost wird darin zu einem **Sonderfall** (exponentieller Verlust, Tiefe 1, Lernrate 1).

```
CART → Bagging → Random Forest → Extra Trees   (Bagging-Ast, fertig)
CART → AdaBoost → Gradient Boosting (dieses Stück) → { XGBoost, LightGBM, CatBoost }   (Boosting-Ast)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 70 % Training / 30 % Test, Seed 7; Tiefe 2, 60 Runden, Lernrate 0.1, sofern nicht anders angegeben) |
|---|---|
| **Kreuzprobe mit scikit-learn (quadratischer Verlust)** | ✅ Mit denselben Daten stimmt die Vorhersage **exakt** (Abweichung < 1e-10) mit `GradientBoostingRegressor(loss="squared_error")` überein – deterministischer Ablauf bei `subsample = 1`. |
| **Kreuzprobe mit scikit-learn (Log-Loss)** | ✅ Ebenfalls **exakt** mit `GradientBoostingClassifier(loss="log_loss")` überein (Wahrscheinlichkeiten und Klassen) – gefundener und behobener Fehler unterwegs: der Newton-Schritt des Blattwerts muss mit der Wahrscheinlichkeit `σ(F)` rechnen, nicht mit dem rohen Score `F`. |
| Kreuzprobe Huber-Verlust | ⚠️ Nur mit Toleranz nah an `GradientBoostingRegressor(loss="huber")` (mittlere Abweichung < 2 min) – leicht andere Delta-Konvention. |
| Kreuzprobe exponentieller Verlust | ➖ **Bewusst nicht exakt** gegen `GradientBoostingClassifier(loss="exponential")`: sklearn nutzt dort einen einzelnen Newton-Schritt, diese Demo die geschlossene Lösung (= AdaBoosts Stimmgewichtsformel, halbiert). Stattdessen gegen **adaboost-demo** geprüft: bei Tiefe 1, Lernrate 1 stimmen die Vorhersagen zu **92,7 %** überein (Mittel über fünf Datensätze) – der Rest kommt vom Baumkern (Varianz- statt Gini-Kriterium). |
| Ein Schritt gegen 60 Runden | ✅ Ein einzelner (mit Lernrate skalierter) Baum: Testfehler **24,4 %** (Raten: 46,4 %). 60 Runden: **15,6 %**. |
| **Lernrate × Rundenzahl** (Mittel über fünf Datensätze, 150 Runden) | ✅ Testminimum bei Lernrate **0,1** (13,7 %), nicht an den Rändern (0,02: 15,0 %; 1,0: 17,0 %). Bei Lernrate 0,1 sinkt der Testfehler mit mehr Runden weiter oder bleibt stabil (5 → 300 Runden: 21,8 % → 13,6 %); bei Lernrate 1,0 steigt er nach einem frühen Minimum wieder (10 → 300 Runden: 14,8 % → 16,9 %) – klassische Überanpassung. |
| **Frühstopp** (Standardeinstellung) | ✅ Der Testfehler erreicht sein Minimum (15,0 %) bei Runde 35 von 60 und bleibt danach nicht weiter besser – Grundlage für das Aufhören anhand eines zurückgehaltenen Testfehlers statt einer festen Rundenzahl. |
| **Verlustwahl bei Ausreißern** (Regression, Mittel über fünf Datensätze) | ✅ Ohne Ausreißer liegt Quadratisch knapp vorn (RMSE 9,9 gegen 10,4 bei Huber). Mit 15 % groben Ausreißern im Training dreht es sich: Quadratisch verschlechtert sich am stärksten (9,9 → 12,2), Huber und Absolut bleiben deutlich stabiler (10,4 → 11,8 bzw. 11,4 → 11,5). |
| **Wirkung der Teilstichprobe** (Mittel über fünf Datensätze) | ⚠️ Rechenzeit sinkt verlässlich, ungefähr proportional zum Anteil. Die Wirkung auf den Testfehler ist dagegen **uneinheitlich** in diesem Datensatz (100 % bis 20 %: 14,3 % bis 15,2 %, keine klare Richtung) – anders als die Lernrate-Rundenzahl-Kompromisslinie kein verlässlicher Regularisierungshebel hier. |

## Was die Demo zeigt

- **Gradient Boosting in Aktion:** Runde für Runde mit Schritt-Regler und Abspielen: links der Baum dieser Runde (Blattwerte = Korrekturen, nicht die Endvorhersage), rechts die Vorhersage
  des Ensembles bis dahin (Entscheidungsgrenze bei Klassifikation, Regressionsfläche bei Regression).
- **Was das Ensemble gelernt hat:** Trainings- und Testfehler (bzw. RMSE) gegen die Rundenzahl mit dem besten Testpunkt markiert, Wichtigkeit je Merkmal.
- **Regler:** Aufgabe (Klassifikation | Regression), Verlustfunktion (je Aufgabe passend), Tiefe der Bäume, Mindestblattgröße, Rundenzahl, Lernrate, Teilstichprobe je Runde, Rauschmerkmale,
  falsche Etiketten (Klassifikation) bzw. grobe Ausreißer (Regression), Lieferungen, Seed.
- **Drei Experimente auf Knopfdruck:** Lernrate × Rundenzahl, Verlustwahl bei Ausreißern (nur Regression), Wirkung der Teilstichprobe (Testfehler und Rechenzeit getrennt).

## Modell und Verfahren

- **Baumkern** (`gb_tree.py`, wortgleich aus cart-demo übernommen): Regressionsbäume mit Varianz-Kriterium – dieselbe Mathematik, ob die äußere Aufgabe Klassifikation oder Regression ist,
  da jede Runde auf den (immer numerischen) Pseudo-Residuen wächst. Eine neue Funktion `set_leaf_values` ersetzt nach dem Wachsen die Blattwerte durch den verlustoptimalen Wert.
- **Verlustfunktionen** (`gb_algorithm.py`): Regression – quadratisch (Mittelwert), absolut (Median), Huber (robuste Lageschätzung, Delta = 0,9-Quantil der Residuen, je Runde neu);
  Klassifikation – Log-Loss (Newton-Schritt), exponentiell (geschlossene Lösung = AdaBoosts Stimmgewichtsformel).
- **Fit:** Startwert (Mittel/Median bzw. (halbe) Log-Odds der Basisrate), je Runde Pseudo-Residuen ausrechnen, Baum wachsen, Blattwerte nachtragen, mit Lernrate skaliert addieren; optional
  eine zufällige Teilstichprobe der Zeilen je Runde (*stochastic gradient boosting*).

## Was nicht funktioniert hat / gefundener Fehler

- **Gefundener und behobener Fehler:** der erste Entwurf des Log-Loss-Blattwerts rechnete `Σ(y - F) / Σ p(1-p)` statt `Σ(y - σ(F)) / Σ p(1-p)` – `F` ist der rohe Score (Log-Odds-Skala),
  nicht die Wahrscheinlichkeit. Das Ergebnis war ein Modell, das mit sklearn nur zu 67 % übereinstimmte (statt exakt) und systematisch zu selbstsicher war. Nach der Korrektur exakter Abgleich
  (Abweichung < 1e-10).
- **Exponentieller Verlust bewusst nicht exakt gegen sklearn:** sklearns `GradientBoostingClassifier(loss="exponential")` nutzt für **alle** Verluste denselben generischen Newton-Schritt
  (`_update_terminal_regions`), auch dort, wo eine geschlossene Lösung existiert. Diese Demo nutzt bewusst die geschlossene Lösung (identisch mit AdaBoosts Alpha-Formel), weil genau das den
  Bezug zu AdaBoost zeigen soll – der Vergleich läuft deshalb gegen adaboost-demo, nicht gegen sklearn.
- **Teilstichprobe zeigt hier keinen klaren Regularisierungseffekt:** mehrere Regime probiert (verschiedene Tiefen, Lernraten, Etiketten-Rauschen) – der Testfehler bewegt sich uneinheitlich
  mit dem Teilstichprobenanteil. Die App zeigt das ehrlich (kein erzwungener "Teilstichprobe hilft"-Claim), nur die Rechenzeit ist ein verlässlicher Hebel.
- **Trainingsfehler bei Ausreißern kann höher als der Testfehler sein:** ein robuster Verlust (Huber, Absolut) ignoriert Ausreißer im Training bewusst – der Trainingsfehler wird gegen die
  **verrauschten** Zielwerte gemessen und bleibt deshalb hoch, obwohl der Testfehler (gegen saubere Werte) gut ist. Die App weist bei aktiven Ausreißern explizit darauf hin.

## Verifikation

`tests/test_algorithm.py` (14 Tests): numerischer Gradienten-Check für alle fünf Verlustfunktionen; **exakter** Abgleich mit `GradientBoostingRegressor`/`GradientBoostingClassifier` für
quadratischen Verlust und Log-Loss; Toleranz-Abgleich für Huber; die geschlossene Blattwertformel des exponentiellen Verlusts gegen ein Brute-Force-Gitter bestätigt; Übereinstimmung mit
adaboost-demo (Tiefe 1, Lernrate 1) über fünf Datensätze; Grenzfälle (eine Runde, Mindestblattgröße, Reproduzierbarkeit der Teilstichprobe, mehr Runden verbessern den Trainingsfehler).
`tests/test_claims.py` (18 Tests) hält **jede Zahl** aus App und README fest. `tests/test_app.py` (26 Tests) prüft die Oberfläche per AppTest (jedes Preset, Aufgaben- und Verlustwechsel,
Abspielen mit rundenspezifischen Diagramm-Schlüsseln, Permalink, alle drei Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `gb_tree.py` | Baumkern (aus cart-demo übernommen) + `set_leaf_values` |
| `gb_algorithm.py` | Verlustfunktionen, Fit, Vorhersage |
| `gb_scenario.py` | Lieferdaten (wie cart-demo, plus `add_outliers` für die Regression) |
| `gb_evaluation.py` | Analyse, Rundenkurve, Lernrate-Rundenzahl-, Ausreißer- und Teilstichproben-Experimente |
| `gb_visualization.py` | Baum-, Karten-, Kurven- und Wichtigkeitsdiagramme |
| `gb_presets.py`, `gb_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
