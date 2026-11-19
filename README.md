# Lastwechsel am Gate: was nach dem Wechsel noch funktioniert – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lastwechsel-gate-demo.streamlit.app/)**

Stück 8 der Reihe **Änderungserkennung** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Karten der vorigen Stücke
rechnen ihre Schwelle unter der Annahme **unabhängiger** Beobachtungen; hier läuft eine einseitige CUSUM-Karte auf einem **Lkw-Tor** (zwei Spuren, 6 Plätze insgesamt, Poisson-Ankünfte, exponentielle
Abfertigung; Zeiteinheit = mittlere Abfertigungsdauer = 3 Minuten, wie in `markov-queue-demo`: Verlust 7,60 %, Wartezeit 1,98 min bei Last 80 %), dessen Beobachtungen abhängig sind. Die Ankunftsrate
springt von 1,6 auf 2,0 (Last 80 % auf 100 %). Gezählt wird in **angenommenen Lkw**; „6 Plätze“ heißt insgesamt, „Gate“ ist das Lkw-Tor (nicht das Filter-Tor aus `robust-filter-demo`).

Die Kante zu den Stücken 1 bis 5 ist `fix`: Die Annahme „unabhängig“ wird am Gate aufgehoben, und das Stück fragt, was **danach** noch geht. Zu Stück 7 (Brüche im Rückblick finden) ist die Kante
ein Kontrast: dort wird im Rückblick segmentiert, hier läuft die Überwachung online, und gefragt ist die Verzögerung.

**Das Gate ist ein Modell** (zwei Spuren, sechs Plätze insgesamt, exponentielle Zeiten); in den gelesenen Quellen gibt es keine reale Wartezeitreihe eines Gates mit einem Lastwechsel. Gemessen wird, wie lange eine Karte nach
einem Anstieg der Ankunftsrate von 1,6 auf 2,0 bis zum Alarm braucht und wie oft sie vorher fälschlich anspricht. Was das für einen echten Betrieb heißt, wird hier nicht behauptet.

## Warum dieses Problem

Eine Schwelle „ARL₀ = 370“ ist eine Aussage über eine bestimmte Welt: unabhängige Beobachtungen mit einer bestimmten Verteilung. Am Gate hängt die Zahl der Lkw, die ein ankommender Lkw sieht, vom Vorgänger ab
(Lag-1-Autokorrelation der Wartezeit 0,78). **Kernfrage:** Was bedeutet die Schwelle dort wirklich, was ist nach dem Zusammenbruch zu tun, und was bringt nichts?

Die fünf Kernfragen der Seite: (1) Was bedeutet „370“ am Gate? (2) Was bringt es, die Schwelle anzupassen? (3) Helfen Blockmittel oder Ausdünnen? (4) Was bringt ein Modell der Schlange? (5) Welche Änderung sieht eine Residuenkarte?

## Modell

Die **gesehene Zahl** *n* (Zahl der Lkw im Gate, die der *i*-te angenommene Lkw bei der Ankunft sieht, 0 bis 5) ist eine **endliche Markov-Kette**. Zwischen zwei Ankünften sterben nur Lkw (Todesprozess), die Lücke ist
Exp(λ): *M = λ(λI − G)⁻¹*; Abgewiesene sehen *K* = 6 und werden übersprungen: *R[n, m] = M[n+1, m] + M[n+1, K]·M[K, m]/(1 − M[K, K])*. Auf *n* laufen Karten mit ganzzahligem Score *s = 8·n − κ* (oder dem
gerundeten Logarithmus des Likelihood-Quotienten der Übergänge); dann ist *(n, C)* eine endliche Kette, und ARL₀, ARL₁ und die Lauflängenverteilung sind **exakte** lineare Gleichungssysteme. Die Wartezeit *W* gegeben *n* ist
Erlang(n − 1, 2) für *n* ≥ 2 und 0 sonst; Karten auf *W* sind **Simulationen** (gelb). Das vollständige Modell steht im Expander „📐 Mathematische Formulierung“ der App.

**Maße.** ARL₀ = mittlere Zahl Lkw bis zum Alarm ohne Änderung (Zero-state: Karte 0, Gate stationär). ARL₁ = mittlere Zahl Lkw von der Änderung bis zum Alarm; ν = letzter Lkw vor der Änderung, der Alarm-Lkw zählt mit
(Verzögerung ≥ 1), **Zero-state**: die Karte startet bei ν mit 0. **Steady-state** heißt quasistationär (lim für lange Läufe ohne Alarm); die Steady-state-ARL₀ an derselben Schwelle ist **nicht** 370 (342,6 bis 360,3 für κ = 20 bis 28; 322,7 für κ = 18), die Spalten
nennen, woran sie geeicht sind. Das ist weder das Lorden- noch das Pollak-Maß. **Das Ziel 370** (gegen 370,398 der Shewhart-Karte mit Grenze 3 σ) ist das Ziel der Linie; einseitig gehört dazu die Schwelle 4,0954 (für 370,398: 4,0965).

## Methodik

1. **Exakt (Experimente 1, 2, 4):** Karte auf der gesehenen Zahl als Kette; ARL₀ und ARL₁ per `scipy.sparse`-Löser, Überlebenskurve per Matrix-Vektor-Produkt, Steady-state per Potenzmethode. Ein Schutz meldet „über 2·10⁵“ statt einer
   Zahl, wenn ARL₀ über 2·10⁵ liegt (ohne den Schutz liefert der Löser bei der unabhängigen Rechnung mit κ = 28 und H = 300 eine negative ARL₀; die Grenze, ab der die Zahl unbrauchbar wird, ist nicht bestimmt, die Reglergrenze ist vorsichtig).
2. **Vorgerechnet (Experimente 3 und 5):** Karten auf der Wartezeit per Simulation, Zufall nur aus SplitMix64 (`lwg_normal.py`, Stapel-Fassung), zwei Zufalls-Seeds (17001 und 17002), je 10 000 Läufe ohne Änderung
   (12 000 Lkw) und 8 000 Läufe nach der Änderung (6 000 Lkw, 400 Lkw Einschwingen, Karte startet bei ν mit 0). Schwelle für ARL₀ = 370 aus dem Lauf ohne Änderung; die Standardfehler enthalten die Unschärfe der Schwelle.
3. **Live:** Experiment 1 bis 4 rechnen live; die Live-Probe in Experiment 3 simuliert höchstens 200 Gates à 2 000 Lkw. Rechenzeiten dieser Maschine (Messung beim Bau, nicht im Test, kein Befund): Experiment 1 0,03 bis 0,07 s, die Tabelle über zwölf Schwellen 0,3 s, Experiment 2 0,03 bis 0,3 s, Experiment 4 beim ersten Aufruf 1,2 bis 1,5 s (danach im Cache), Live-Probe 0,3 s (50 Läufe) bis 0,7 s (200 Läufe); Streamlit Cloud ist nicht gemessen.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit überall angenommene Lkw. **exakt** = endliche Kette, kein Näherungsfehler (Tests: 1e-7 relativ); **gelb** = Simulation (Mittel ± Standardfehler). Alle Zahlen gelten für dieses Gate (M/M/2/6, λ₀ = 1,6), das exakt
bekannte Nullmodell, einseitige CUSUM-Karten und Zero-state, wenn nicht anders genannt.

| Frage | Befund | Art |
|---|---|---|
| Was bedeutet die Lehrbuch-Schwelle H = 97 (k = 2,5)? | Unter „unabhängig“ ARL₀ = 408,2; auf der Kette **51,2** (Faktor 8,0). Alarme in den ersten 100 Lkw: 19,6 % gegen **87,2 %**. | exakt |
| Wächst der Faktor mit der Schwelle? | Ja auf dem Raster der zwölf Schwellen (κ = 20) ab H = 30, bei H = 20 kehrt er sich um (0,57); zwischen den Rasterpunkten und für andere κ nicht gerechnet. Tabelle unten. | exakt |
| Dieselbe Erfahrung auf der Wartezeit | Naiver zweiseitiger CUSUM (k = 0,5, h = 4,77; nominell 368,6): ARL₀ **42,5 ± 0,4** (Zufalls-Seed 17001) und **42,2 ± 0,4** (17002), Median 32, 91,8 % beziehungsweise 92,2 % der Läufe schlagen in den ersten 100 Lkw an (Vormessung: 42,28 ± 0,38). | gelb |
| Schwelle auf die Kette angepasst (ARL₀ = 370) | Verzögerung **101,2** statt 37,2 Lkw (Faktor 2,7), beide zwischen den benachbarten ganzzahligen Schwellen log-linear auf ARL₀ = 370 interpoliert; Steady-state 89,2 an derselben Schwelle (Steady-ARL₀ dort 342,6) bzw. 93,0 bei Steady-ARL₀ = 370. Schwelle 24,7 statt 7,8 Standardabweichungen der gesehenen Zahl. | exakt |
| Verteilung der Verzögerung (Schwelle H = 310, ARL₀ = 378,1) | In 50 / 100 / 200 Lkw schlagen 26,7 / 61,9 / 89,9 % der Läufe an, Mittel 102,4, Median 80, 95-%-Quantil 254. | exakt |
| Holt ein anderes k den Wert der unabhängigen Rechnung? | Bei keinem der acht gemessenen k (2,25 bis 4,47) holt die Kette den Wert der unabhängigen Rechnung: sie ist 1,1- bis 2,7-fach langsamer, die Verhältnisse fallen mit wachsendem k Richtung 1; die beste der acht Karten der Kette (k = 2,5: 101,2) ist 2,7-mal langsamer als die beste der acht für unabhängige Lkw (37,2). k über 4,5 und feiner dazwischen sind nicht gerechnet. | exakt |
| Schiefe oder Abhängigkeit? (Wartezeit, Wechsel 1,6 → 2,0, ARL₀ = 370) | unabhängig Normal 50,5 → unabhängig mit der exakten schiefen Randverteilung 56,0 → Gate **134,7** (Rohkarte). | exakt / gelb |
| Helfen Blockmittel? (Zufalls-Seed 17001, Wechsel auf einer Blockgrenze) | b = 5 / 10 / 20 / 40: **135,2 / 135,9 / 135,4 / 136,2** (je ± 1,4) gegen Rohkarte 134,7 ± 1,4: der Unterschied ist nicht aufgelöst. Lag-1 der Blockmittel 0,36 / 0,18 / 0,08 / 0,04. | gelb |
| … und wenn die Änderung nicht auf einer Blockgrenze liegt? | 133,0 / 131,5 / 130,4 / 129,0 (b/2 Lkw des ersten Blocks vor dem Wechsel): 1,7 bis 5,6 Lkw schneller als die Rohkarte (bei b ≥ 10 aufgelöst, bei b = 5 nicht); gemessen sind nur zwei Phasen (Blockgrenze und Blockmitte), weitere nicht. Weit von 101 entfernt. | gelb |
| Hilft Ausdünnen? | Jeder 5. / 10. / 20.: 141,7 / 166,7 / 202,2 (Rohkarte 134,7). | gelb |
| Karte auf der gesehenen Zahl gegen Wartezeit-Rohkarte | 101,3 ± 1,0 gegen 134,7 ± 1,4: rund ein Viertel schneller (nur unter der Annahme, dass jeder Lkw die Zahl im Gate sieht). | gelb |
| Likelihood-Quotient-Karte der Übergänge (für 2,0 entworfen) bei ARL₀ = 370 | 102,3 gegen 101,2 der Rohkarte (Gitter 1/40); 107,4 bei Gitter 1/20, 102,2 bei 1/80. Merklich (5 % und mehr) gewinnt sie erst ab λ₁ = 3,0 und nur mit dafür entworfenem Score (λ₁ = 4,0: 22,4 gegen 26,6). | exakt |
| Information je Lkw | 0,01058 (Übergänge) gegen 0,06362 (Randverteilung der gesehenen Zahl); ln(370)/0,01058 = 559, das 5,2- bis 5,5-Fache des exakten Werts 102,3: keine Schranke bei endlichem ARL₀. | exakt |
| Falsches Nullmodell (Karten für 1,6 geeicht) | Wahres λ₀′ = 1,7: ARL₀ das 0,65-Fache (Rohkarte) bzw. 0,64-Fache (Modellkarte, Gitter 1/40; bei Gitter 1/20: 0,67); λ₀′ = 1,5: das 1,70- bzw. 1,73-Fache (Gitter 1/40). | exakt |
| Residuenkarte (Wartezeit gegeben die gesehene Zahl) | Ratenänderung 1,6 → 1,8 / 2,0 / 2,4: **312,0 / 269,6 / 215,5** (Rohkarte 203,9 / 134,7 / 78,9): 1,5- bis 2,7-mal langsamer, aber unter ARL₀ = 370 (nicht blind). Abfertigung 1,0 → 0,85: 107,8 ± 1,2 gegen Rohkarte 102,2 ± 1,0: rund 5 % langsamer (auch im zweiten Zufalls-Seed). | gelb |
| Simulator gegen Kette | Karte auf n (ARL₀ und ARL₁ nach 1,6 → 2,0 bei ARL₀ = 370) gleich der exakten Kette im Band; Rand, Übergänge und Autokorrelation (0,78 / 0,59 / 0,41 / 0,26 / 0,18) stimmen. | gelb |

**Der Faktor je Schwelle** (exakt, κ = 20, Zahl in Lkw; „über 2·10⁵“ heißt: keine verlässliche Zahl):

| Schwelle H | ARL₀ unabhängig | ARL₀ Kette | Faktor | Alarme in 100 Lkw (unabhängig / Kette) |
|---|---|---|---|---|
| 20 | 7,7 | 13,4 | 0,57 | 100,0 % / 99,9 % |
| 30 | 19,6 | 17,0 | 1,15 | 99,6 % / 99,7 % |
| 40 | 31,0 | 20,4 | 1,52 | 96,9 % / 99,3 % |
| 50 | 57,3 | 25,0 | 2,29 | 83,8 % / 98,4 % |
| 60 | 82,8 | 28,8 | 2,87 | 70,7 % / 97,4 % |
| 80 | 190,6 | 39,0 | 4,88 | 39,4 % / 93,3 % |
| 100 | 408,2 | 51,2 | 7,97 | 19,6 % / 87,2 % |
| 130 | 1 282,8 | 75,5 | 16,99 | 6,0 % / 74,5 % |
| 160 | 3 371,2 | 102,2 | 32,99 | 2,0 % / 62,5 % |
| 200 | 13 096,3 | 150,8 | 86,86 | 0,4 % / 46,9 % |
| 250 | 75 364,1 | 236,8 | 318,26 | 0,1 % / 31,0 % |
| 300 | über 2·10⁵ | 345,6 | – | 0,0 % / 20,6 % |

**Die fünf Referenzwerte bei ARL₀ = 370** (Wechsel 1,6 → 2,0, exakt; „Steady“ = Steady-state-ARL₁ an derselben Schwelle, in Klammern die Steady-state-ARL₀ dort, dann bei Steady-state-ARL₀ = 370):

| κ (k = κ/8) | Schwelle unabhängig | ARL₁ unabhängig | Schwelle Kette | ARL₁ Kette | Steady (ARL₀ dort) | Steady bei 370 | Kette / unabhängig |
|---|---|---|---|---|---|---|---|
| 20 (2,50) | 97,4 | 37,2 | 309,1 | 101,2 | 89,2 (342,6) | 93,0 | 2,72 |
| 22 (2,75) | 68,1 | 41,7 | 231,4 | 104,2 | 95,8 (351,0) | 98,7 | 2,50 |
| 24 (3,00) | 51,0 | 49,5 | 177,6 | 108,4 | 102,0 (355,6) | 104,5 | 2,19 |
| 26 (3,25) | 40,8 | 58,0 | 135,6 | 112,7 | 107,5 (358,3) | 109,8 | 1,94 |
| 28 (3,50) | 32,1 | 64,2 | 103,0 | 116,9 | 112,5 (360,3) | 114,5 | 1,82 |

Weitere k (Zero-state): k = 2,25 → 2,50 (103,2 gegen 41,3); k ≈ 3,97 → 1,31 (125,5 gegen 96,1); k ≈ 4,47 → 1,13 (141,2 gegen 125,1); auf dem groben Raster gleich (k = 4,0: 1,29; k = 4,5: 1,14), also kein Rasterartefakt.
k = 2,0 liegt unter dem Nullmittel der gesehenen Zahl (2,16): die Karte driftet schon ohne Änderung, das ist keine sinnvolle Karte und wird nicht gerechnet. Der Raster ist eine Treppe: alle Scores sind Vielfache von 2, die Schwellen
sind log-linear zwischen den beiden Treppenstufen auf 370 interpoliert.

**Größe der Änderung** (exakt, ARL₀ = 370, Rohkarte κ = 20 / Modellkarte für 2,0 / Modellkarte für das wahre λ₁; Gitter 1/40):

| wahres λ₁ | Rohkarte | Modellkarte (für 2,0) | Modellkarte (für das wahre λ₁) |
|---|---|---|---|
| 1,7 | 240,9 | 239,4 | 248,1 |
| 1,8 | 169,9 | 169,1 | 173,0 |
| 2,0 | 101,2 | 102,3 | 102,3 |
| 2,4 | 55,3 | 57,8 | 55,7 |
| 3,0 | 36,0 | 38,3 | 34,2 |
| 4,0 | 26,6 | 28,1 | 22,4 |

**Falsches Nullmodell** (exakt; Karten fest, Faktor = ARL₀ gegen nominelle ARL₀; Rohkarte H = 309, Modellkarte H = 74):

| wahres λ₀′ | 1,4 | 1,5 | 1,55 | 1,6 | 1,65 | 1,7 | 1,8 |
|---|---|---|---|---|---|---|---|
| Faktor Rohkarte | 3,23 | 1,70 | 1,29 | 1,00 | 0,80 | 0,65 | 0,45 |

Eine Abweichung der Rate im Nullzustand um 6 % (1,7 gegen 1,6) senkt ARL₀ auf das 0,65-Fache, nahe am Sprung selbst (2,0 gegen 1,6 sind 25 %). Beide Karten brauchen das Nullmodell auf wenige Prozent genau.

## Befunde und Korrekturen

Die Zahlen der Vormessung (V17) sind Erwartungen; im Bau sind alle neu gemessen (SplitMix64, zwei Zufalls-Seeds). Abweichungen:

- **Gate-Zahl.** Die Vormessung nannte 42,28 ± 0,38 (numpy-Strom); neu gemessen 42,5 ± 0,4 und 42,2 ± 0,4 (zwei Zufalls-Seeds, je 10 000 Läufe): im selben Band. Die ältere Messung (V2) 42,9 ± 0,7 liegt ebenfalls darin.
- **Blockgrenze.** Die Vormessung hielt „kein Gewinn durch Blöcke“ für die Aussage und die Phase der Änderung gegen den Block für nicht gemessen („begünstigt die Blockkarten“). Gemessen ist das Gegenteil: in den zwei gemessenen Phasen ist die Blockkarte
  auf der Blockgrenze langsamer als in der Blockmitte (weitere Phasen sind nicht gemessen); liegt die Grenze in der Mitte, sind es 1,7 bis 5,6 Lkw weniger (133,0 / 131,5 / 130,4 / 129,0 gegen Rohkarte 134,7), und das ist nicht beim Zufalls-Seed 17002 allein (130,3 / 129,6 / 128,8 / 127,2 gegen 134,9). Es bleibt bei „nicht in
  die Nähe von 101“.
- **Wartezeit-Karten, neu gemessen** (Zufalls-Seed 17001; Vormessung in Klammern): Rohkarte 134,7 (135,0), Ausdünnen jeder 5. / 10. / 20.: 141,7 / 166,7 / 202,2 (143,2 / 168,8 / 200,4), Blockmittel 135,2 bis 136,2 (135,2 bis 135,9), Karte auf n 101,3 (101,0),
  Residuenkarte 269,6 (274,9); Abfertigung: Rohkarte 102,2 (103,7), Karte auf n 133,6 (135,2), Residuenkarte 107,8 (109,2). Die Residuenkarte ist 1,5- bis 2,7-mal langsamer als die Rohkarte (Vormessung: 2,8).
- **Schwellen ohne grobes Raster.** In der Vormessung stützten sich Experiment 4 und Teile des Experiments 2 auf Rasterschwellen (ganzzahlig aufgerundet). Hier wird ARL₀ = 370 immer zwischen den beiden benachbarten ganzzahligen Schwellen log-linear interpoliert.
  Dadurch ändern sich Randzahlen: Rohkarte für 2,0 jetzt 101,2 (Vormessung 100,7 bei H = 308, dort ARL₀ 367,0), Modellkarte mit Gitter 1/20 jetzt 107,4 (107,65). Der Vormessungs-Wert 270,7 für die für λ₁ = 1,7 entworfene Karte war ein **Gitterartefakt**:
  fein gerechnet sind es 248,1 (Rohkarte 240,9); die Karte bleibt auch dort nicht besser als die Rohkarte.
- **Falsches Nullmodell.** Mit den ganzzahligen Schwellen H = 309 und H = 74 (Vormessung H = 308 / 74): Faktor 0,65 und 0,64 bei λ₀′ = 1,7 (Vormessung 0,651 und 0,644), 1,70 und 1,73 bei 1,5 (1,69 und 1,73).
- **Referenzwert κ = 16** (k = 2,0) ist nicht anwählbar (Drift schon ohne Änderung); der Regler läuft von κ = 18 (k = 2,25) bis 28.
- **Ausdünnen jeder 40.** ist nicht gemessen (nur jeder 5., 10. und 20.).
- **Nach unabhängiger Prüfung eingeschränkt:** „H = 309 ergibt ARL₀ 370“ ist ungenau, bei der ganzzahligen Schwelle H = 309 ist ARL₀ = 378,1 (H = 308: 367,0); 309,1 ist die interpolierte Schwelle. Die Voreinstellung H = 97 wirkt wie H = 100: unabhängig ARL₀ = 408,2, nicht 370 (370 gilt bei der interpolierten Schwelle 97,4). Die Residuenkarte ist bei der Abfertigungsänderung rund 5 % langsamer als die Rohkarte (nicht „nicht langsamer“).
- **Offener Codeverdacht (nicht geändert):** Die Blockkarten in `lwg_sim.json` (`tools/lwg_make_sim.py`) erhalten k = Sprung/Standardabweichung der Blockmittel/2, wobei der Sprung in Einheiten von σ₀ steht, die Standardabweichung der Blockmittel aber absolut; das k ist dadurch um den Faktor 1,114 (= 1/σ₀) zu groß (b = 20: 0,2985 statt 0,268). Eine unabhängige Simulation mit beiden k änderte ARL₁ um höchstens 0,8 Lkw (b = 20: 137,5 gegen 136,7 ± 1,4); die Schlüsse bleiben, die Zahlen der Blockkarten sind nicht neu gemessen.
- **Neu gegenüber der Vormessung:** zweiter Zufalls-Seed für alle Wartezeit-Zeilen (alle 44 Zellen beider Sätze liegen im 4-Standardfehler-Band der Differenz), Phase der Änderung gegen den Block, Live-Probe, Steady-state-Zahlen für κ = 18.

## 🚧 Wo die Annahmen enden

| Annahme | Was passiert, wenn sie verletzt ist | Wo steht es |
|---|---|---|
| **Das Gate ist ein Modell:** zwei Spuren, 6 Plätze insgesamt, exponentielle Zeiten | In den gelesenen Quellen gibt es keine reale Wartezeitreihe eines Gates mit Lastwechsel; andere Plätze, Spuren oder Verteilungen sind nicht gemessen. Keine Aussage über den Nutzen einer Alarmregel im Betrieb. | alle Experimente |
| **Das Nullmodell (λ₀ = 1,6) ist exakt bekannt** | Weicht die wahre Rate ohne Änderung ab, sinkt ARL₀ schon bei 1,7 statt 1,6 auf etwa das 0,65-Fache; mit **geschätztem** Nullmodell wird alles schlechter (nicht gemessen; Stück 4 zeigt die Empfindlichkeit). | Experiment 4 |
| **Die gesehene Zahl ist beobachtbar** (jeder Lkw sieht die Zahl der Lkw im Gate) | Das ist eine Modellannahme; der Vorsprung der Karte auf *n* gegen die Wartezeit-Rohkarte gilt nur unter ihr. | Experimente 1 bis 4 |
| **Einseitige CUSUM-Karten, Zero-state** | Zweiseitige Karten (Lastabfall), Steady-state-Maße außer in Experiment 2, das Lorden- und das Pollak-Maß für abhängige Beobachtungen, EWMA und Shewhart auf Blockmitteln sind nicht gemessen. | alle Experimente |
| **Nur die Ankunftsrate ändert sich** (Abfertigung nur bei der Residuenkarte) | Andere Änderungsarten (mehrere Klassen, andere Abfertigungszeiten) sind nicht gemessen. | Experiment 5 |
| **Karten auf der Wartezeit sind Simulation** (Zufalls-Seed fest, zwei Sätze) | Zahlen als Mittel ± Standardfehler; Unterschiede unter zwei Standardfehlern der Differenz gelten als nicht aufgelöst. | Experimente 3 und 5 |

**Bewusst nicht gemessen** (steht auch in der App):

- Zweiseitige Karten und das Lorden-/Pollak-Maß für abhängige Beobachtungen (nur Zero-state und Steady-state der Karte; das Maximum über die Vorgeschichte ist nicht gemessen).
- Andere Kartenfamilien (EWMA, Shewhart auf Blockmitteln, Shiryaev–Roberts, Western-Electric) und andere Gates, Änderungsarten, Verteilungen.
- Geschätzte Parameter (Phase I aus dem Gate selbst); μ₀, σ₀ und λ₀ sind exakt bekannt.
- Das Segmentieren der Gate-Reihe nach dem Wechsel (Fehlbrüche auf der stationären Gate-Reihe; Stück 7).
- Echte Daten: keine.
- Linux-Lauf der Tests (`demo_linux_check.py`) und Rechenzeiten auf Streamlit Cloud.
- Keine Aussage über Chen/Zhou 2014 und Jana/Dey 2000: nicht gelesen.

## Quellen und Lesestufen

Stufen aus der Vorarbeit übernommen (W = gelesen), beim Bau nicht erneut geprüft:

- Vanhatalo/Kulahci 2015, QREI 31 (**W**): drei Ansätze bei Autokorrelation (Rohdaten mit theoretischer Grenze, Rohdaten mit per Simulation angepasster Grenze, Residuen eines VAR(1)); die Residuenkarte stabilisiere ARL₀ nur bei perfekt bekanntem Modell.
- Larsen u. a., arXiv 1907.05453 (**W**, Einleitung): Residuen eines AR(1) erkennen die Verschiebung zuerst, dann immer schlechter.
- Zwetsloot/Jones-Farmer/Woodall 2024 (**W**, über eine Sekundärquelle): Aggregation über die Zeit kostet Auflösung.
- Baron/Malov, arXiv 2502.09185 (**W**, Abstract): die CUSUM-Rekursion ist die Lindley-Rekursion (der Wartezeit); ein Ein-Server-System steht im Abstract nicht, Abschnitt 1 ist nicht eingesehen.
- Chen/Zhou 2014 (Technometrics; nur S) und Jana/Dey 2000 (nur A): **nicht gelesen**; sie sind Hinweise, dass die Idee der Modellkarte vermutlich nicht neu ist. Lorden 1971, Moustakides 1986: **nicht gelesen**; die Näherung ln γ/I wird nie als Schranke geführt.
- Alle Zahlen dieser README sind eigene Nachrechnungen in diesem Modell. Keine Neuheit: alles Lehrbuch; neu ist die Vorführbarkeit mit Zähleinheit und exakter Kette für die Gate-Beobachtung.

## Zahl → Test

Jede Zahl der README steht in `tests/test_claims.py` (exakt oder im Band, in der Spalte Art); die Messreihen der Wartezeit-Karten sind in `tests/test_tables.py` gegen die exakte Kette und zwischen den beiden Zufalls-Seeds geprüft.

| Zahl | Test | Art |
|---|---|---|
| 408,2 / 51,2 / 8,0 / 19,6 % / 87,2 %; H = 20: 0,57; Tabelle je Schwelle | `test_claims.py::test_zusammenbruch_*`, `test_tabelle_je_schwelle` | exakt, 1e-7 relativ (Tabelle auf die angezeigten Stellen) |
| 101,2 / 37,2 / 2,7 / 89,2 / 93,0 / 24,7 / 7,8; 26,7 / 61,9 / 89,9 % | `test_claims.py::test_schwelle_angepasst_*` | exakt |
| fünf κ, acht k, Kontrollzeilen | `test_claims.py::test_referenzwerte_*`, `test_tables.py` | exakt |
| 50,5 / 56,0 | `test_claims.py::test_bezugsgroessen_*`, `test_oracle_*` | exakt (Integralgleichung, Zellenkette, Monte-Carlo) |
| 42,5 / 42,2 / 134,7 / Blockmittel / Ausdünnen / Residuen | `test_claims.py::test_wartezeit_*`, `test_tables.py` | gelb: Wert aus der Datei, Band gegen zweiten Satz und Kette |
| Modellkarte, Information, Näherung | `test_claims.py::test_modellkarte_*` | exakt |
| Falsches Nullmodell | `test_claims.py::test_falsches_nullmodell_*` | exakt |
| Handbeispiele (49/6, 103/24, 9, 63/16 …) | `test_handbeispiele.py` | Brüche |

## Tests

**255 Tests**, seriell rund 145 s (Messung am 2026-11-19, andere Läufe liefen zeitweise parallel; diese Maschine, Windows, Python 3.14, ein BLAS-Thread; `pytest tests/ -q`). Zusammensetzung: Handbeispiele als Brüche, Orakel (Resolvente gegen Integral der Matrixexponentialfunktion, Geburts-Sterbe-Formel,
eigene 1-D-Kette, Summe der Überlebenswahrscheinlichkeiten, Monte-Carlo, Ereignissimulation des Gates, Nyström gegen Zellenkette), Eigenschaften der Kette und der Karten, Simulator gegen Kette, gefrorene Daten, Claims, Verdikte (je Zweig, je Zahl),
Schnellstart-Verdikte, Randfälle, App (AppTest), Layout, Footer, Laufzeit-Importe, Zufallsquelle, Literaturwerte (nur die V2-Zahlen mit Quelle, Rest „offen“).
Die Neu-Rechnen-Tests (`--check` beider Messskripte) vergleichen im Band; exakte Zahlen mit relativer Toleranz 1e-7. Der Lauf unter Linux (`demo_linux_check.py`) wurde **nicht** ausgeführt.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (fünf Experimente) |
| `lwg_chain.py` | Kette der gesehenen Zahl: M, R, Randverteilung, Momente |
| `lwg_cards.py` | Karten als Ketten (n, C): ARL, Überlebenskurve, Steady-state, Schwellen, Schutz |
| `lwg_iid.py` | Bezugsgrößen für unabhängige Beobachtungen (Normalverteilung, schiefe Wartezeit) |
| `lwg_sim.py` | Gate-Simulator und Karten auf der Wartezeit, Live-Probe |
| `lwg_normal.py`, `lwg_rng.py` | Zufall: SplitMix64 (Vorlage unverändert) und Stapel-Fassung |
| `lwg_reference.py`, `lwg_kennlinien.json`, `lwg_sim.json` | gefrorene Daten (Quelle und Stand im Kopf) |
| `lwg_evaluation.py` | Kennzahlen und Verdikte (reine Funktionen) |
| `lwg_texts.py`, `lwg_presets.py`, `lwg_constants.py`, `lwg_layout.py`, `lwg_visualization.py` | Texte, Presets, Konstanten, fester Kopf, Abbildungen |
| `tools/lwg_make_kennlinien.py` | erzeugt `lwg_kennlinien.json` (`--voll` rund 3 Minuten, `--check`) |
| `tools/lwg_make_sim.py` | erzeugt `lwg_sim.json` (`--voll` rund 6 Minuten, `--check` rund 25 s) |
| `tests/` | Tests (siehe oben); `tests/mini_gate.py`: Handrechnung mit Brüchen |

## Bewusst nicht umgesetzt

- **Kein Nachbau** von `niveausprung-demo` (Detektor auf Regressionsresiduen, Zurücksetzen, Bestätigung, Kosten): hier das Gate-Modell, das Maß ist die Lauflänge bis zum Alarm. Dort: Anwendung; hier: Prüfstand.
- **Keine Wiederholung** von `markov-queue-demo` (Gleichgewicht, Einschwingzeit der Schlange; dort: wie lange das Gate zum Einschwingen braucht, hier: was eine Karte auf der Reihe tut) und `output-analysis-demo` (Batch Means als Intervall der mittleren Wartezeit; hier Blockmittel als Beobachtungen einer Karte).
- **Kein eigener Löser außerhalb von numpy/scipy**, kein PDF-Export (die Seite hat keine Einzelinstanz, die man exportieren würde).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
