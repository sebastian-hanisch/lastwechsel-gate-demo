"""Alle längeren Texte der App (Einleitung, Erklärung, Kernfragen, Grenzen, Formeln). Zahlen stehen hier nur, wenn `tests/test_claims.py` sie prüft;
Zahlen aus Rechnungen kommen als Verdikte aus `lwg_evaluation.py`."""

INTRO = """
Stück 8 der Reihe „Änderungserkennung“. Die Karten der vorigen Stücke rechnen ihre Schwelle unter der Annahme, dass die Beobachtungen **unabhängig** sind. Hier läuft eine Karte
auf einem **Lkw-Tor** (zwei Spuren, sechs Plätze insgesamt, Poisson-Ankünfte, exponentielle Abfertigung), dessen Beobachtungen abhängig sind, und die **Ankunftsrate springt** von
1,6 auf 2,0 (Last 80 % auf 100 %). Gefragt wird: Was bedeutet die Schwelle „ARL₀ = 370“ dort wirklich, was **funktioniert danach noch** (Schwelle neu bestimmen, Blockmittel,
Ausdünnen, Modellkarte, Residuenkarte) und was nicht? Die gesehene Zahl der Lkw im Gate ist eine **exakte Markov-Kette**: dort sind alle Zahlen exakt; die Karten auf der
Wartezeit sind vorgerechnete **Simulationen** (gelb).

Weiter unten: die Experimente **🔬 1 bis 5**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

VEHIKEL = (
    "Das Gate ist ein Modell (zwei Spuren, sechs Plätze insgesamt, exponentielle Zeiten); in den gelesenen Quellen gibt es keine reale Wartezeitreihe eines Gates mit einem Lastwechsel. "
    "Gemessen wird, wie lange eine Karte nach einem Anstieg der Ankunftsrate von 1,6 auf 2,0 bis zum Alarm braucht und wie oft sie vorher fälschlich anspricht. "
    "Was das für einen echten Betrieb heißt, wird hier nicht behauptet."
)

SO_FUNKTIONIERT = """
1. **Das Gate.** Lkw kommen mit der Rate λ an (λ = 1,6 sind 80 % Last, 2,0 sind 100 %); zwei Spuren, jede fertigt in der Zeit 1 (= 3 Minuten) ab; höchstens **6 Lkw insgesamt** sind im Gate, ein weiterer wird abgewiesen und zählt nicht. Gezählt wird in **angenommenen Lkw**.
2. **Die Beobachtung.** Jeder angenommene Lkw sieht bei der Ankunft *n* Lkw im Gate (0 bis 5) und wartet danach die Zeit *W*. Beide Folgen sind abhängig: *n* ist eine endliche Markov-Kette (hier exakt gerechnet), *W* nicht.
3. **Die Karte.** Ein einseitiger CUSUM *C = max(0, C + s)* mit Score *s* je Lkw schlägt bei *C ≥ H* an (Anstieg nach oben gesucht). Auf *n* ist *(n, C)* eine endliche Kette: Lauflänge und Verzögerung sind **exakt**.
4. **Die Maße.** **ARL₀** = mittlere Zahl Lkw bis zum Alarm ohne Änderung (Karte bei 0, Gate stationär). **ARL₁** = mittlere Zahl Lkw von der Änderung bis zum Alarm; ν ist der letzte Lkw vor der Änderung, der Alarm-Lkw zählt mit (**Zero-state**: Karte startet bei ν mit 0). **Steady-state** heißt hier quasistationär (Karte lange ohne Alarm gelaufen). Das ist weder das Lorden- noch das Pollak-Maß.
5. **Das Ziel „370“.** Die Linie eicht alle Karten auf ARL₀ = 370 Lkw (die Shewhart-Karte mit Grenze 3 σ hat 370,398; die kleine Differenz ist Absicht und steht in der README).
"""

FRAGEN = """
**Die fünf Kernfragen dieser Seite:** (1) Was bedeutet „370“ am Gate? (2) Was bringt es, die Schwelle anzupassen? (3) Helfen Blockmittel oder Ausdünnen? (4) Was bringt ein Modell der Schlange?
(5) Welche Änderung sieht eine Residuenkarte? Jede Frage hat ein Experiment mit einem Verdikt, das live aus den Reglern berechnet wird.
"""

EXP1_FRAGE = """
**Kernfrage 1: Was bedeutet „370“ am Gate?** Die Schwelle der Rohkarte auf der gesehenen Zahl wird einmal so gerechnet, wie die Lehrbuch-Rechnung es tut (Lkw **unabhängig**, mit der
richtigen Randverteilung der gesehenen Zahl), und einmal auf der **tatsächlichen Kette**. Beide Zahlen sind exakt (Gleichungssystem über *(n, C)*); Regler: Schwelle *H* und Referenzwert κ
(*k = κ/8*) in der Seitenleiste. Die Voreinstellung H = 97 ist die Lehrbuch-Schwelle: unabhängig ARL₀ = 408,2.
"""

EXP1_NEGATIV = """
**Was das nicht zeigt.** Der Faktor hängt an der Schwelle: bei kleinen Schwellen kehrt er sich um (bei H = 20 liegt ARL₀ der Kette mit 13,4 Lkw über den 7,7 der Rechnung für unabhängige Lkw), bei großen wächst er
schnell (Tabelle in der README). Die Randverteilung der gesehenen Zahl allein verrät den Zusammenbruch nicht; er kommt aus der Abhängigkeit zwischen aufeinanderfolgenden Lkw.
Dieselbe Erfahrung auf der **Wartezeit** (Simulation): ein naiver zweiseitiger CUSUM mit k = 0,5 und h = 4,77, für unabhängige Werte auf ARL₀ ≈ 370 gestellt, schlägt am Gate nach im Mittel
{v2_a} Lkw an ({v2_b} aus dem zweiten Zufalls-Seed; Vormessung 42,28 ± 0,38) statt nach 370; {v2_p} % der Läufe schlagen in den ersten 100 Lkw an.
"""

EXP2_FRAGE = """
**Kernfrage 2: Was bringt es, die Schwelle auf die Kette anzupassen?** Die Schwelle wird so gerechnet, dass die **tatsächliche** ARL₀ das Ziel trifft (Regler: Ziel-ARL₀, κ, Größe des Wechsels λ₁).
Die Kennlinien stammen aus einer gefrorenen exakten Rechnung (`lwg_kennlinien.json`), die Verteilung der Verzögerung an der ganzzahligen Schwelle wird live gerechnet.
"""

EXP2_NEGATIV = """
**Ehrlicher Negativbefund.** Die angepasste Schwelle stellt ARL₀ wieder her, aber die Verzögerung bleibt ein Mehrfaches der Rechnung für unabhängige Lkw; die Tabelle zeigt es für fünf Referenzwerte κ,
darunter die Reihe über acht k. Steady-state (quasistationär) ist in allen Zeilen um 2 bis 19 % schneller als Zero-state (an gleicher Schwelle mit niedrigerer ARL₀); die beiden Steady-state-Spalten stehen getrennt, weil an derselben
Schwelle die Steady-state-ARL₀ nicht 370 ist.
"""

EXP3_FRAGE = """
**Kernfrage 3: Helfen Blockmittel oder Ausdünnen?** Idee: Das Mittel von *b* aufeinanderfolgenden Wartezeiten (oder nur jeder *b*-te Lkw) ist fast unabhängig, also müsste die Lehrbuch-Karte dort
passen. Gemessen (Simulation der Wartezeit, gelb, vorgerechnet): Verzögerung bei gleicher ARL₀ = 370 Lkw. *b = 1* ist die Rohkarte. Dazu eine kleine **Live-Probe** des Gates (Zufalls-Seed und Läufe in der
Seitenleiste): Autokorrelation, Blockmittel und der naive CUSUM aus Experiment 1.
"""

EXP3_NEGATIV = """
**Was das nicht zeigt.** Die Änderung liegt in der Hauptmessung auf einer Blockgrenze; ein zweiter Lauf mit der Änderung in der Blockmitte steht im Verdikt, weitere Phasen sind nicht gemessen.
Gemessen sind nur CUSUM-Karten, einseitig, Zero-state und die Wechsel 1,6 → 1,8 / 2,0 / 2,4. Die Standardfehler enthalten die Unschärfe der Schwelle; ein zweiter Zufalls-Seed liegt in der Tabelle daneben.
Bezugsgrößen für unabhängige Beobachtungen: Normalverteilung und exakte schiefe Randverteilung der Wartezeit (rechte Spalte der Tabelle) trennen Schiefe von Abhängigkeit.
"""

EXP4_FRAGE = """
**Kernfrage 4: Was bringt ein Modell der Schlange?** Die Likelihood-Quotient-Karte der Übergänge (Score ln R₁(n, m)/R₀(n, m), auf das Gitter 1/q_s gerundet) nutzt die Kettenstruktur. Verglichen wird sie
bei gleicher ARL₀ = 370 mit der Rohkarte (κ = 20) und dann unter **falschen Annahmen**: anderer wahrer Sprung als der Entwurf, anderes Nullmodell. Alles exakt (Ketten). Regler: wahres λ₁, Entwurf-λ₁, wahres λ₀′ und Gitter q_s.
"""

EXP4_NEGATIV = """
**Hinweis.** Die Idee „Likelihood-Quotient-Karte auf einer Warteschlange“ ist vermutlich nicht neu (Chen/Zhou 2014 und Jana/Dey 2000 sind Hinweise, beide nicht gelesen: keine Aussage über sie).
Die Zahlen unten gelten für dieses Gate und diese Karte. Bei λ₁ = 1,7 ist die für das wahre λ₁ entworfene Karte nicht besser als die Rohkarte; eine gröbere Vorab-Rechnung hatte dort 270,7 (Gitterartefakt der Schwellen), die feinere Rechnung hier nicht (README, Befunde und Korrekturen).
"""

EXP5_FRAGE = """
**Kernfrage 5: Welche Änderung sieht eine Residuenkarte?** Die Wartezeit gegeben die gesehene Zahl, *U = F(W | n)*, ist für **jede** Ankunftsrate gleichverteilt (die Wartezeit hängt gegeben *n* nicht von λ ab);
die Karte auf *z = Φ⁻¹(U)* rechnet das Modell heraus, in dem die Ratenänderung steckt. Gemessen (Simulation, gelb, vorgerechnet): Verzögerung bei ARL₀ = 370 Lkw für drei Ratenänderungen und eine Abfertigungsänderung.
"""

EXP5_NEGATIV = """
**Was das nicht zeigt.** „Die Residuenkarte ist blind für die Ratenänderung“ wäre falsch: ihre Verzögerung liegt unter ARL₀ (die Abhängigkeit der z-Folge ändert sich mit der Rate), nur deutlich über der Rohkarte.
Die Zuordnung der Ursache („sie rechnet das Modell heraus“) ist eine Deutung; gemessen ist das Verhalten dieser Regel, mit bekanntem Modell (bester Fall, vgl. Vanhatalo/Kulahci 2015 laut Plan).
"""

ASSUMPTIONS_MARKDOWN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wo steht es |
|---|---|---|
| **Das Gate ist ein Modell:** zwei Spuren, 6 Plätze insgesamt, exponentielle Zeiten | In den gelesenen Quellen gibt es keine reale Wartezeitreihe eines Gates mit Lastwechsel; andere Plätze, Spuren oder Verteilungen sind nicht gemessen. Keine Aussage über den Nutzen einer Alarmregel im Betrieb. | alle Experimente |
| **Das Nullmodell (λ₀ = 1,6) ist exakt bekannt** | Weicht die wahre Rate ohne Änderung ab, sinkt ARL₀ schon bei 1,7 statt 1,6 auf etwa das 0,65-Fache; mit **geschätztem** Nullmodell wird alles schlechter (nicht gemessen; Stück 4 zeigt die Empfindlichkeit). | Experiment 4 |
| **Die gesehene Zahl ist beobachtbar** (jeder Lkw sieht die Zahl der Lkw im Gate) | Das ist eine Modellannahme; der Vorsprung der Karte auf *n* gegen die Wartezeit-Rohkarte gilt nur unter ihr. | Experimente 1 bis 4 |
| **Einseitige CUSUM-Karten, Zero-state** | Zweiseitige Karten (Lastabfall), Steady-state-Maße außer in Experiment 2, das Lorden- und das Pollak-Maß für abhängige Beobachtungen, EWMA und Shewhart auf Blockmitteln sind nicht gemessen. | alle Experimente |
| **Nur die Ankunftsrate ändert sich** (Abfertigung nur bei der Residuenkarte) | Andere Änderungsarten (mehrere Klassen, andere Abfertigungszeiten) sind nicht gemessen. | Experiment 5 |
| **Karten auf der Wartezeit sind Simulation** (Zufalls-Seed fest, zwei Sätze) | Zahlen als Mittel ± Standardfehler; Unterschiede unter zwei Standardfehlern der Differenz gelten als nicht aufgelöst. | Experimente 3 und 5 |
"""

CONVENTIONS_MARKDOWN = """
**Konventionen.** Zähleinheit: angenommene Lkw (Abgewiesene zählen nicht). ν = letzter Lkw vor der Änderung; die Verzögerung *T − ν* ist mindestens 1 (der Alarm-Lkw zählt mit). Zero-state: Karte startet bei ν mit
0, das Gate ist im alten Zustand stationär; Steady-state: quasistationär (lim für lange Läufe ohne Alarm). ARL₀ = 370 ist das Ziel der Linie (gegen 370,398 der Shewhart-Karte). „Fehlalarm“ heißt Alarm vor der Änderung, je Zeit gezählt.
"""

MATH_INTRO = """
**Lesehilfe.** Gegeben: ein Gate mit *c* Spuren und *K* Plätzen, Ankunftsrate λ, Abfertigungsrate μ. Gesucht: die Verteilung der Lauflänge einer CUSUM-Karte auf der Folge der gesehenen Zahlen, vor und nach einem
Sprung von λ, **exakt**, und mit der Schwelle als Stellgröße.

| Symbol | Bedeutung |
|---|---|
| *c = 2*, *K = 6*, μ = 1 | Spuren, Plätze insgesamt, Abfertigungsrate je Spur (Zeiteinheit = mittlere Abfertigungsdauer) |
| λ₀ = 1,6, λ₁ | Ankunftsrate vor und nach dem Wechsel |
| *M*, *R* | Übergangsmatrix der Zahl im System bei der nächsten Ankunft; Übergangsmatrix der von angenommenen Lkw gesehenen Zahl |
| *n_i*, *W_i* | gesehene Zahl und Wartezeit des *i*-ten angenommenen Lkw |
| π | stationäre Verteilung von *R* |
| *s*, *q*, κ, *H* | Score je Lkw, Skala (q = 8), Referenzwert, Schwelle |
| *Q*, *L* | Übergangsmatrix der Karte auf *(n, C)*, mittlere Zahl Lkw bis zum Alarm je Startzustand |
"""

MATH_MARKDOWN = r"""
**Vollständiges Modell** (alle Bereiche und Größen sind genannt; die Rechnung selbst steht in `lwg_chain.py` und `lwg_cards.py`).

**1. Gate und Kette der gesehenen Zahl.** Zahl im System *j ∈ {0, …, K}*. Zwischen zwei Ankünften sterben nur Lkw: Todesprozess mit Generator *G*, *G[j, j−1] = min(j, c)·μ*, *G[j, j] = −G[j, j−1]*, sonst 0. Die Lücke bis zur nächsten Ankunft ist exponentialverteilt mit Rate λ, also

$$M = \lambda\,(\lambda I - G)^{-1}, \qquad M[j, m] = P(\text{nächste Ankunft sieht } m \mid j \text{ unmittelbar nach einer Ankunft}).$$

Abgewiesene Lkw sehen *K* und werden übersprungen; für *n, m ∈ {0, …, K−1}*

$$R[n, m] = M[n{+}1, m] + \frac{M[n{+}1, K]\; M[K, m]}{1 - M[K, K]}, \qquad \pi R = \pi,\ \ \textstyle\sum_n \pi_n = 1 .$$

Gegeben *n* ist die Wartezeit *W* gleich 0 für *n < c* und Erlang-verteilt mit Form *n − c + 1* und Rate *c·μ* sonst; *E[W] = Σₙ πₙ · max(0, n − c + 1)/(c·μ)*.

**2. Karte und Kette (n, C).** *C₀ = 0*, *Cᵢ = max(0, Cᵢ₋₁ + sᵢ)*, Alarmzeit *T = min{i : Cᵢ ≥ H}* (Zähleinheit angenommene Lkw). Rohkarte: *sᵢ = q·nᵢ − κ* (ganzzahlig, *H* ganzzahlig, *C ∈ {0, …, H−1}*).
Modellkarte: *sᵢ = round(q_s · ln R₁(nᵢ₋₁, nᵢ)/R₀(nᵢ₋₁, nᵢ))*. Dann ist *(nᵢ, Cᵢ)* eine endliche Kette mit der substochastischen Matrix

$$Q\big((n, C), (m, C')\big) = R[n, m] \quad\text{falls } C' = \max(0, C + s(n, m)) < H,\ \text{sonst } 0 .$$

**3. Maße.** Mit *1* dem Einsvektor ist *L = (I − Q)⁻¹ 1* die mittlere Zahl Lkw bis zum Alarm je Startzustand. Zero-state: Start *(n, 0)* mit *n ~ π₀*.

$$\mathrm{ARL}_0 = \sum_n \pi_0(n)\, L^{(0)}(n, 0), \qquad \mathrm{ARL}_1 = \sum_n \pi_0(n)\, L^{(1)}(n, 0), \qquad P(T > t) = v_0^{\top} Q^{t}\, \mathbf 1 .$$

Dabei ist *L⁽⁰⁾* das *L* zur Matrix *R₀* (vor dem Wechsel) und *L⁽¹⁾* das zu *R₁* (nach dem Wechsel); der Start ist immer *n ~ π₀*. Steady-state: Start mit der quasistationären Verteilung von *Q* (linker Eigenvektor zum größten Eigenwert).
Die „unabhängige“ Rechnung setzt *R[n, m] = π(m)* in jeder Zeile (Randverteilung, Lkw unabhängig).

**4. Karten auf der Wartezeit (Simulation).** *zᵢ = (Wᵢ − μ₀)/σ₀* mit exakt gerechnetem *μ₀ = 0,6590*, *σ₀ = 0,8977*; Rohkarte *Cᵢ = max(0, Cᵢ₋₁ + zᵢ − k)*, *k =* halber Sprung (*0,1393*). Blockmittel: *z* aus dem Mittel von *b* Lkw (standardisiert mit der
Standardabweichung der Blockmittel). Ausdünnen: nur Lkw *b, 2b, …*. Naiv zweiseitig: *C⁺*, *C⁻* mit *k = 0,5*, Alarm bei *C > 4,77*. Residuenkarte: *Uᵢ = F(Wᵢ | nᵢ)* (bei *n < c* gleichverteilt zufällig), *zᵢ = Φ⁻¹(Uᵢ)*, *k = 0,25*.

**5. Kullback-Information je Lkw.** $I = \sum_{n,m} \pi_1(n)\,R_1[n, m]\,\ln\frac{R_1[n,m]}{R_0[n,m]}$ (Likelihood-Quotient der Übergänge) gegen $\sum_n \pi_1(n)\ln\frac{\pi_1(n)}{\pi_0(n)}$ (Randverteilung).
Die Näherung *ln(370)/I* wird nicht als Schranke benutzt (README, Befunde).

**Quellen und Stufen** (Lesestufen aus der Vorarbeit übernommen, beim Bau nicht erneut geprüft): Vanhatalo/Kulahci 2015 (QREI 31; W): Ansätze bei Autokorrelation. Larsen u. a. (arXiv 1907.05453; W, Einleitung): Residuen eines AR(1).
Zwetsloot/Jones-Farmer/Woodall 2024 (W, über eine Sekundärquelle): Aggregation über die Zeit kostet Auflösung. Baron/Malov (arXiv 2502.09185; W, Abstract): die CUSUM-Rekursion ist die Lindley-Rekursion (der Wartezeit); ein Ein-Server-System steht im Abstract nicht.
Chen/Zhou 2014 (Technometrics; nur S) und Jana/Dey 2000 (nur A): **nicht gelesen**. Lorden 1971, Moustakides 1986: **nicht gelesen**.
"""
