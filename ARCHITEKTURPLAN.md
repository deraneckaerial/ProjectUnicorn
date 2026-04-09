# Trading View Analyst — Architekturplan

## 1. Vision & Scope

### Was wir bauen
Ein lokales Trading-Analyse-Dashboard, das wissenschaftlich fundierte technische Analyse mit systematischem Risikomanagement und Anti-Bias-Mechanismen kombiniert. Der Trader erhält für ein gewähltes Asset eine strukturierte Analyse mit klarer Handlungsempfehlung (Long/Short/Neutral) samt Konfidenzwert — und wird dabei aktiv vor seinen eigenen kognitiven Verzerrungen geschützt.

### Zielgruppe
Privat-Trader mit Grundkenntnissen, die:
- Emotionale Entscheidungen durch regelbasierte Analyse ersetzen wollen
- Ein kompaktes Tool statt einer überladenen Plattform suchen
- Ihre eigene Disziplin durch systematische Checks erzwingen wollen

### Was es NICHT ist
- **Kein Auto-Trader.** Das Tool analysiert und empfiehlt — es platziert keine Orders.
- **Keine Anlageberatung.** Jede Ausgabe enthält einen Disclaimer. Keine Haftung für Handelsentscheidungen.
- **Kein HFT-System.** Kein Millisekundenhandel, kein Latenz-Optimierung. Fokus liegt auf Swing-Trading (1-7 Tage Halteperiode).
- **Kein Daten-Vendor.** Wir nutzen kostenlose APIs und deren Limits.

---

## 2. Wissenschaftliche Fundierung

### Indikatoren mit Evidenz

Die Recherche zeigt klar: **Einzelne Indikatoren allein sind nicht zuverlässig.** Die Kombination mehrerer Indikatoren über verschiedene Kategorien liefert die besten Ergebnisse.

**Trend-Indikatoren (Primär):**
| Indikator | Evidenz | Einsatz |
|-----------|---------|---------|
| MACD | Validiert in Kombination mit DMI/KST (ScienceDirect 2024). GRU-Modelle mit 5-Tage-Lookback zeigen Outperformance. | Trend-Richtung + Momentum |
| DMI/ADX | Teil der validierten Kombination. ADX > 25 = starker Trend. | Trendstärke |
| EMA (20/50/200) | Breit akzeptiert als Trend-Filter. Nicht allein handelbar. | Kontext / Filter |

**Momentum-Indikatoren (Sekundär):**
| Indikator | Evidenz | Einsatz |
|-----------|---------|---------|
| RSI | Überverkauft/Überkauft als Mean-Reversion-Signal in Range-Märkten. | Extremwerte-Erkennung |
| KST | Teil der validierten Kombination (10-Tage-Lookback). | Langfristige Momentum-Bestätigung |

**Volatilität:**
| Indikator | Evidenz | Einsatz |
|-----------|---------|---------|
| ATR | Standard für Stop-Loss-Berechnung und Position Sizing. | Risikomanagement |
| Bollinger Bands | Nützlich für Volatilitäts-Regime-Erkennung. | Kontext |

**Volumen:**
| Indikator | Evidenz | Einsatz |
|-----------|---------|---------|
| OBV | Volumen-Bestätigung von Preisbewegungen. | Bestätigungssignal |

### Kritische Bewertung der Evidenz

**Was funktioniert (mit Einschränkungen):**
- Kombination MACD + DMI + KST schlägt Buy-and-Hold in bestimmten Märkten
- Optimale Halteperiode für Swing-Trading: 1-7 Tage (Forex-Studie, 22 Jahre Daten)
- Fractional Kelly Criterion für Position Sizing reduziert Drawdowns messbar

**Was NICHT funktioniert (trotz Papers):**
- Die "beste kürzlich gefundene Regel" performt out-of-sample schlechter als Buy-and-Hold (Springer 2023, 6406 Rules). Das bedeutet: **Wir dürfen Parameter nicht overfitt en.**
- 99.3% Accuracy bei CNN-Candlestick-Erkennung ist In-Sample-Overfitting. Realistische Accuracy liegt bei ~70%.
- Einzelne Indikatoren ohne Kontext liefern zu viele Fehlsignale.

**Konsequenz für unser Design:**
- Feste, konservative Parameter statt Optimierung (kein Curve-Fitting)
- Multi-Indikator-Scoring statt einzelner Signale
- Jede Empfehlung zeigt die historische Trefferquote mit an

### ML für v1 — Ehrliche Einschätzung

**Nein.** ML ist für den MVP nicht nötig und kontraproduktiv:
- Training braucht saubere, umfangreiche Daten (die wir erst sammeln müssen)
- Overfitting-Risiko bei kleinen Datensätzen ist enorm
- Ein regelbasiertes Scoring-Modell ist transparent, erklärbar und in 1-2 Tagen baubar
- ML wird erst in Phase 3 relevant, wenn wir genug historische Signaldaten haben

**Stattdessen für v1:** Gewichtetes Scoring-Modell mit festen Regeln.

### Kelly Criterion — Position Sizing

Formel: `f* = (bp - q) / b` bzw. `K% = W - [(1-W) / R]`

Für die Praxis verwenden wir **Quarter Kelly** (f*/4):
- Deutlich geringere Drawdowns als Full Kelly
- Geringfügig niedrigere Returns, aber massiv weniger Risiko
- Benötigt als Input: Win-Rate (W) und durchschnittliches Reward/Risk-Ratio (R)
- Der Trader gibt sein Gesamtkapital und maximales Risiko pro Trade ein

### Anti-Bias-Maßnahmen

| Bias | Gegenmaßnahme im Tool |
|------|----------------------|
| Confirmation Bias | **Forced Disconfirmation**: Für jedes Bullish-Signal werden explizit Bearish-Argumente angezeigt und umgekehrt |
| Loss Aversion | **Automatische Stop-Loss-Berechnung** basierend auf ATR. Wird VOR dem Entry angezeigt |
| Anchoring | **Multi-Timeframe-Analyse** (Daily + Weekly + Monthly) statt Fixierung auf einen Zeitrahmen |
| Overconfidence | **Historische Accuracy** des Modells für dieses Setup wird angezeigt |
| Herding | **Sentiment-Kontraindikator**: Extreme Retail-Sentiment-Werte als Warnsignal |
| Disposition Effect | **Regelbasierte Exit-Signale**: Tool sagt wann raus, nicht das Bauchgefühl |
| Recency Bias | **Langzeit-Performance-Stats** werden immer neben kurzfristigen angezeigt |

---

## 3. Technologie-Stack

### Core
| Komponente | Technologie | Begründung |
|------------|-------------|------------|
| UI | **Streamlit** | Schnellster Weg zu einem funktionierenden Dashboard. Keine Frontend-Erfahrung nötig. |
| Charts | **lightweight-charts-python** | TradingView-Qualität direkt in Streamlit. Interaktive Candlestick-Charts. |
| Technische Analyse | **pandas-ta** | Einfache Installation (pip), 150+ Indikatoren, aktive Community. TA-Lib hat C-Dependencies die auf Windows Probleme machen. |
| Daten | **yfinance** (primär) + **Finnhub** (Sentiment) | yfinance: Kostenlos, kein Key, breite Abdeckung. Finnhub: 60 Req/Min, Real-Time-News + Sentiment. |
| Dataframes | **pandas** | Standard. |
| Berechnungen | **numpy** | Standard für numerische Operationen. |

### Warum diese Entscheidungen?

**pandas-ta statt TA-Lib:** TA-Lib ist 2-4x schneller, aber: (a) C-Dependencies sind auf Windows ein bekanntes Problem, (b) wir berechnen Indikatoren für ein einzelnes Asset, nicht für Tausende — Performance ist irrelevant, (c) pandas-ta installiert sich mit `pip install pandas-ta` ohne Komplikationen.

**yfinance statt Alpha Vantage:** Alpha Vantage hat nur 25 Calls/Tag im Free Tier. Das reicht für einen einzigen Analyse-Durchlauf kaum. yfinance hat keine harten Limits und deckt Aktien, ETFs und Crypto ab.

**Finnhub für Sentiment (Phase 2):** 60 Req/Min Free Tier ist großzügig genug. Liefert News-Sentiment das wir als Kontraindikator nutzen können.

**Kein Datenbank-Layer für v1:** Daten werden bei jedem Lauf frisch gezogen und im Memory gehalten. Historische Signale landen als CSV in `.tmp/`. Eine DB lohnt sich erst wenn wir Backtesting über lange Zeiträume machen.

---

## 4. Architektur (WAT-Framework)

### Datenpipeline

```
[User Input: Ticker + Timeframe]
         │
         ▼
[Tool: fetch_market_data.py]  ──→  OHLCV-Daten via yfinance
         │
         ▼
[Tool: calculate_indicators.py]  ──→  MACD, DMI, RSI, KST, ATR, EMA, OBV, BB
         │
         ▼
[Tool: generate_signal.py]  ──→  Multi-Indikator-Scoring → Empfehlung (Long/Short/Neutral)
         │
         ▼
[Tool: calculate_risk.py]  ──→  Stop-Loss (ATR), Position Size (Kelly), R/R-Ratio
         │
         ▼
[Tool: forced_disconfirmation.py]  ──→  Gegenargumente zur Empfehlung
         │
         ▼
[Streamlit Dashboard]  ──→  Chart + Empfehlung + Risiko + Anti-Bias-Checks
```

### Workflows

**`workflows/analyze_asset.md`** — Haupt-Workflow
- Objective: Komplettanalyse eines Assets von Datenabruf bis Empfehlung
- Inputs: Ticker-Symbol, Timeframe (daily/weekly)
- Steps: fetch → calculate → score → risk → disconfirmation → display
- Edge Cases: API-Fehler, Daten zu kurz für Indikatoren, Ticker nicht gefunden

**`workflows/position_sizing.md`** — Kelly Criterion Workflow
- Objective: Berechne optimale Position Size
- Inputs: Kapital, Win-Rate, Avg Win/Loss Ratio, Risk Tolerance
- Steps: Kelly berechnen → auf Quarter Kelly reduzieren → Max-Risk-Cap anwenden
- Edge Cases: Win-Rate < 50% (Kelly sagt "nicht handeln"), ungültige Inputs

**`workflows/backtest_signal.md`** (Phase 2)
- Objective: Historische Performance des Scoring-Modells testen
- Inputs: Ticker, Zeitraum, Signal-Parameter
- Steps: Historische Daten laden → Signale generieren → Trades simulieren → Performance messen

### Tools

```
tools/
├── fetch_market_data.py      # yfinance-Wrapper: OHLCV-Daten holen
├── calculate_indicators.py   # pandas-ta: Alle Indikatoren berechnen
├── generate_signal.py        # Scoring-Modell: Indikatoren → Empfehlung
├── calculate_risk.py         # ATR-Stop-Loss + Kelly Position Sizing
├── forced_disconfirmation.py # Gegenargumente generieren
└── app.py                    # Streamlit Dashboard (UI-Layer)
```

**Tool-Details:**

**`fetch_market_data.py`**
- Input: Ticker (str), Period (str, z.B. "1y"), Interval (str, z.B. "1d")
- Output: pandas DataFrame mit OHLCV
- Error Handling: Ticker nicht gefunden, API-Timeout, leere Daten

**`calculate_indicators.py`**
- Input: OHLCV DataFrame
- Output: DataFrame mit allen Indikator-Spalten
- Indikatoren: MACD(12,26,9), DMI(14), ADX(14), RSI(14), KST, ATR(14), EMA(20,50,200), OBV, Bollinger(20,2)
- Feste Parameter — kein Tuning, kein Overfitting

**`generate_signal.py`**
- Input: DataFrame mit Indikatoren
- Output: Signal-Dict mit `direction` (long/short/neutral), `confidence` (0-100), `reasons` (list), `contra_reasons` (list)
- Scoring-Logik:

```
Score = 0 (Range: -100 bis +100)

Trend (Gewicht: 40%)
  +20  MACD > Signal Line UND steigend
  +10  ADX > 25 (starker Trend)
  +10  Kurs > EMA50 > EMA200

Momentum (Gewicht: 30%)
  +15  RSI zwischen 30-70 (gesund) ODER RSI < 30 (überverkauft = long)
  +15  KST > Signal Line

Volumen (Gewicht: 15%)
  +15  OBV steigend bei Kurs steigend (Bestätigung)

Volatilität (Gewicht: 15%)
  +15  Kurs innerhalb Bollinger Bands (kein Extrem)
  ODER
  -15  Kurs außerhalb BB + hoher ADX (Trend-Erschöpfung möglich)

Interpretation:
  Score > 40:   Strong Long
  Score 20-40:  Lean Long
  Score -20-20: Neutral (kein Trade)
  Score -40--20: Lean Short
  Score < -40:  Strong Short
```

**`calculate_risk.py`**
- Input: Signal, aktueller Preis, ATR, User-Kapital, Win-Rate, Avg R/R
- Output: Stop-Loss-Preis, Take-Profit-Preis, Position Size (Stück + Euro), Risk per Trade
- Stop-Loss: Entry ± 2 * ATR (Standard, anpassbar)
- Position Size: Quarter Kelly, gedeckelt bei max. 5% des Kapitals

**`forced_disconfirmation.py`**
- Input: Signal-Dict (Richtung + Gründe)
- Output: Liste von Gegenargumenten
- Logik: Für jedes Bullish-Argument wird das entsprechende Bearish-Szenario formuliert. Bei "MACD bullish crossover" → "MACD kann in Seitwärtsmärkten viele Fehlsignale liefern". Fest kodierte Gegenargumente pro Indikator-Signal — kein LLM nötig.

### .env

```
# Phase 1 (MVP) - kein Key nötig für yfinance
# FINNHUB_API_KEY=        # Phase 2: News-Sentiment

# User-spezifische Defaults (optional, überschreibbar in UI)
DEFAULT_CAPITAL=10000
DEFAULT_MAX_RISK_PCT=2
DEFAULT_WIN_RATE=50
DEFAULT_RR_RATIO=2
```

---

## 5. Features nach Priorität

### MVP (Phase 1) — Ziel: 1-2 Tage

**Scope:** Ein funktionierendes Dashboard das für einen Ticker eine komplette Analyse liefert.

- [ ] Ticker-Eingabe + Timeframe-Auswahl (Daily/Weekly)
- [ ] OHLCV-Daten via yfinance laden
- [ ] Candlestick-Chart mit lightweight-charts-python
- [ ] Indikatoren berechnen (MACD, DMI/ADX, RSI, EMA 20/50/200, ATR, OBV, Bollinger)
- [ ] Scoring-Modell → Long/Short/Neutral mit Konfidenz
- [ ] Stop-Loss + Take-Profit auf Basis ATR
- [ ] Position Sizing mit Quarter Kelly
- [ ] Forced Disconfirmation (Gegenargumente-Box)
- [ ] Multi-Timeframe-Übersicht (Daily + Weekly Signal nebeneinander)
- [ ] Disclaimer: "Keine Anlageberatung"

### Phase 2 — Erweiterungen

- [ ] Watchlist: Mehrere Ticker speichern und auf einen Blick sehen
- [ ] News-Sentiment via Finnhub als zusätzlicher Faktor
- [ ] Backtesting: Historische Performance des Scoring-Modells
- [ ] Signal-History: Vergangene Signale speichern und Accuracy tracken
- [ ] Alert-System: Benachrichtigung wenn ein Ticker das Scoring-Threshold erreicht
- [ ] Crypto-Support via CoinGecko/Binance API
- [ ] Indikatoren als Overlay im Chart einblenden

### Phase 3 — Advanced / ML

- [ ] ML-Modell (GRU) trainiert auf eigenen historischen Signaldaten
- [ ] Pattern Recognition (Candlestick-Muster via CNN, wenn genug Daten)
- [ ] Portfolio-View: Korrelationen zwischen gehaltenen Positionen
- [ ] Regime Detection: Erkennung ob Markt Trend/Range/Volatile
- [ ] Automatische Reports (PDF/Slides)

---

## 6. Anti-Bias-Features im Detail

### Forced Disconfirmation (MVP)
Jede Analyse hat zwei Spalten:
- **Dafür (Bull Case):** Die Gründe für das Signal
- **Dagegen (Bear Case):** Aktiv generierte Gegenargumente

Der Trader MUSS beide lesen bevor er die Position Size sieht. UI-technisch: Gegenargumente werden zuerst angezeigt, Position Sizing erst nach Scroll/Click.

### Position Sizing Calculator (MVP)
- Eingabe: Gesamtkapital, max. Risiko pro Trade (%), historische Win-Rate
- Quarter Kelly berechnet die optimale Größe
- Hard Cap: Nie mehr als 5% des Kapitals in einem Trade
- Output: Exakte Stückzahl + Euro-Betrag + "Wenn der Trade scheitert, verlierst du X Euro"

### Regelbasierte Entry/Exit (MVP)
- Entry nur wenn Score-Threshold erreicht UND Multi-Timeframe-Bestätigung
- Exit bei: (a) Stop-Loss erreicht, (b) Take-Profit erreicht, (c) Signal dreht auf Neutral
- Kein "Bauchgefühl"-Override — das Tool zeigt die Regeln und der Trader entscheidet

### Overconfidence-Check (MVP)
- Anzeige: "Dieses Signal-Setup hatte historisch eine Trefferquote von X%"
- Anfangs auf Basis allgemeiner Indikator-Statistiken, später auf eigenen Daten

### Sentiment-Kontraindikator (Phase 2)
- Extreme Bullish-Sentiment (>80%) → Warnung "Markt möglicherweise überkauft"
- Extreme Bearish-Sentiment (<20%) → Warnung "Markt möglicherweise überverkauft"
- News-Sentiment von Finnhub als zusätzlicher Datenpunkt

---

## 7. Risiken & Limitationen

### Technische Grenzen
- **yfinance ist inoffiziell.** Yahoo kann die Schnittstelle jederzeit ändern oder blockieren. Fallback: Alpha Vantage (mit strengen Rate Limits).
- **Keine Echtzeit-Daten.** yfinance liefert End-of-Day oder 15-Min-Delayed. Für Intraday-Trading ungeeignet.
- **Kein Backtesting in v1.** Die Scoring-Parameter sind auf Basis der Forschung gewählt, nicht auf historischen Daten validiert. Backtesting kommt in Phase 2.

### Methodische Grenzen
- **Out-of-Sample-Degradation.** Die Forschung zeigt: Optimale Regeln performen in der Zukunft schlechter. Unsere Antwort: Feste, konservative Parameter statt Optimierung.
- **Kein heiliger Gral.** Technische Analyse allein kann Märkte nicht konsistent schlagen. Das Tool ist ein Entscheidungs-Support, kein Orakel.
- **Regime-Blindheit in v1.** Das Scoring unterscheidet nicht zwischen Trend- und Seitwärtsmärkten. ADX hilft teilweise, aber ein echtes Regime-Modell kommt erst in Phase 3.

### Regulatorisch
- **Keine Anlageberatung.** Muss in der UI prominent stehen. Jede Empfehlung ist eine technische Analyse, keine Finanz-Beratung.
- **Kein MiFID/WpHG-Relevanz** solange wir keine Orders platzieren und keine personalisierten Empfehlungen aussprechen.

### Was das Tool NICHT kann
- Fundamentalanalyse (Earnings, P/E, Cashflow)
- Makro-Events vorhersagen (Fed-Entscheidungen, Black Swans)
- Marktmanipulation erkennen
- Garantierte Gewinne liefern

---

## 8. Umsetzungsplan

### Phase 1: MVP (1-2 Tage)

**Tag 1: Backend + Core-Logic**
| Schritt | Tool | Aufwand |
|---------|------|---------|
| Projekt-Setup (venv, requirements.txt) | — | 15 min |
| `fetch_market_data.py` | yfinance | 30 min |
| `calculate_indicators.py` | pandas-ta | 1h |
| `generate_signal.py` (Scoring) | — | 1.5h |
| `calculate_risk.py` (Kelly + ATR) | — | 1h |
| `forced_disconfirmation.py` | — | 30 min |
| Unit Tests für Scoring-Logik | pytest | 1h |

**Tag 2: UI + Integration**
| Schritt | Tool | Aufwand |
|---------|------|---------|
| `app.py` Streamlit Dashboard | Streamlit | 2h |
| Chart-Integration | lightweight-charts-python | 1h |
| Multi-Timeframe-View | — | 1h |
| UI-Polish + Anti-Bias-UX | — | 1h |
| End-to-End-Test | — | 30 min |

### Phase 2: Erweiterungen (1-2 Wochen, bei Bedarf)
- Watchlist + Signal-History
- Finnhub-Integration (Sentiment)
- Backtesting-Modul
- Indikator-Overlays im Chart

### Phase 3: ML/Advanced (nur wenn Phase 2 stabil)
- GRU-Modell auf eigenen Signaldaten
- Regime Detection
- Portfolio-Korrelation

---

## Kritische Selbstbewertung

### Was wurde gestrichen (Overengineering)
- **Datenbank für v1:** Nicht nötig. CSVs in `.tmp/` reichen.
- **ML in jeglicher Form für v1:** Kein GRU, kein CNN, kein LSTM. Ein gewichtetes Scoring-Modell mit festen Regeln ist transparenter, erklärbarer und in Stunden statt Wochen baubar.
- **WebSocket-Echtzeit-Daten:** Für Swing-Trading (1-7 Tage) braucht man keine Echtzeit-Daten. End-of-Day reicht.
- **Eigenes Backtesting-Framework:** Kommt in Phase 2. Für MVP reicht die Analyse des aktuellen Zustands.
- **Multi-Asset-Portfolio-Optimierung:** Viel zu komplex für v1. Wir analysieren ein Asset nach dem anderen.

### Ist der Tech-Stack die einfachste Lösung?
Ja. Streamlit + yfinance + pandas-ta + lightweight-charts-python ist der kürzeste Weg von Null zu einem funktionierenden Dashboard. Alle Libraries installieren sich via pip. Kein Build-Step, kein Frontend-Framework, keine Datenbank.

### Ist der MVP in 1-2 Tagen baubar?
Ja, wenn wir strikt bei den definierten Features bleiben:
- 5 Python-Scripts mit klarer Verantwortung
- 1 Streamlit-App die alles zusammenbringt
- Feste Parameter, kein Tuning
- Kein ML, kein Backtesting, keine Datenbank

### Verbleibende Risiken
- yfinance-Stabilität (Mitigation: Try/Except + Fallback-Meldung)
- Scoring-Parameter sind theoretisch fundiert aber nicht empirisch validiert (Mitigation: Backtesting in Phase 2, konservative Thresholds)
- lightweight-charts-python Streamlit-Integration kann Quirks haben (Mitigation: Falls Probleme, Fallback auf plotly Candlestick-Charts)
