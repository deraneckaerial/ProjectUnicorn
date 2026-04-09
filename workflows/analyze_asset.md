# Workflow: Analyse eines Assets

## Objective
Komplettanalyse eines Trading-Assets von Datenabruf bis Handlungsempfehlung mit Anti-Bias-Checks.

## Required Inputs
- **Ticker Symbol** (z.B. AAPL, BTC-USD, MSFT)
- **Timeframe** (Daily oder Weekly)
- **Risiko-Parameter** (Kapital, Max-Risk, Win-Rate, R/R-Ratio)

## Pipeline

### 1. Daten holen
**Tool:** `tools/fetch_market_data.py`
- `fetch_ohlcv(ticker, period, interval)` → OHLCV DataFrame
- `fetch_multi_timeframe(ticker)` → Daily + Weekly DataFrames

### 2. Indikatoren berechnen
**Tool:** `tools/calculate_indicators.py`
- `calculate_all_indicators(df)` → DataFrame mit 25+ Indikator-Spalten
- Feste Parameter (kein Tuning!)

### 3. Signal generieren
**Tool:** `tools/generate_signal.py`
- `generate_signal(df)` → Scoring (-100 bis +100) → Direction + Confidence
- Kategorien: Trend (40%), Momentum (30%), Volumen (15%), Volatilität (15%)

### 4. Risiko berechnen
**Tool:** `tools/calculate_risk.py`
- Stop-Loss: Entry ± 2 × ATR
- Position Sizing: Quarter Kelly, max 5% des Kapitals
- Take-Profit: Entry ± (ATR × Multiplier × R/R-Ratio)

### 5. Forced Disconfirmation
**Tool:** `tools/forced_disconfirmation.py`
- Gegenargumente zu jedem Signal
- Bias-Warnung basierend auf Signalstärke
- Wird VOR der Position Size angezeigt

### 6. Anzeige
**Tool:** `tools/app.py` (Streamlit)
- Candlestick-Chart mit EMAs, Bollinger, SL/TP
- MACD + RSI Sub-Charts
- Anti-Bias-Box mit Bull/Bear Case
- Position Sizing Metriken
- Scoring-Breakdown

## Edge Cases
- **Ticker nicht gefunden:** ValueError mit Hinweis
- **Zu wenig Daten (<50 Bars):** Signal = Neutral, Hinweis anzeigen
- **yfinance API-Fehler:** Try/Except, User-Meldung
- **ATR = 0:** Fallback auf Standardabweichung der letzten 5 Tage
- **Kelly sagt "kein Trade":** Message anzeigen, kein Position Sizing

## Starten
```bash
cd B:\PlanB\Projekte\TradingViewAnalyst
venv\Scripts\streamlit.exe run tools/app.py
```
