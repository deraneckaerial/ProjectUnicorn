# Trading View Analyst — Agent Instructions

You're working inside the **WAT framework** (Workflows, Agents, Tools). This architecture separates concerns so that probabilistic AI handles reasoning while deterministic code handles execution.

## Project Overview

**Trading View Analyst** is a local trading analysis dashboard that combines scientifically validated technical analysis with systematic risk management and anti-bias mechanisms. It provides structured analysis with clear recommendations (Long/Short/Neutral) plus confidence scores — while actively protecting traders from cognitive biases.

**What it is:** A decision-support tool for swing traders (1-7 day hold periods).
**What it is NOT:** An auto-trader, financial advisor, or HFT system.

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- Markdown SOPs stored in `workflows/`
- Each workflow defines the objective, required inputs, which tools to use, expected outputs, and how to handle edge cases

**Layer 2: Agents (The Decision-Maker)**
- This is your role. Read the relevant workflow, run tools in the correct sequence, handle failures gracefully.

**Layer 3: Tools (The Execution)**
- Python scripts in `tools/` that do the actual work
- Credentials and API keys are stored in `.env`

## Tool Overview

| Tool | Purpose |
|------|---------|
| `tools/fetch_market_data.py` | Fetch OHLCV data via yfinance |
| `tools/calculate_indicators.py` | Calculate technical indicators via pandas-ta |
| `tools/generate_signal.py` | Multi-indicator scoring → Long/Short/Neutral |
| `tools/calculate_risk.py` | ATR-based Stop-Loss + Quarter Kelly position sizing |
| `tools/forced_disconfirmation.py` | Generate counter-arguments for every signal |
| `tools/app.py` | Streamlit dashboard (UI layer) |

## Key Design Decisions

1. **No ML in v1.** Weighted scoring model with fixed, conservative parameters. No curve-fitting.
2. **Anti-bias is a first-class feature.** Forced disconfirmation is shown BEFORE position sizing.
3. **Quarter Kelly for position sizing.** Hard cap at 5% of capital per trade.
4. **Multi-timeframe required.** Daily + Weekly signals shown side-by-side.
5. **Fixed indicator parameters.** No optimization/tuning to prevent overfitting.

## How to Operate

**1. Look for existing tools first.** Check `tools/` before building anything new.

**2. Learn and adapt when things fail.**
- Read the full error message
- Fix the script and retest (check with me before paid API calls)
- Document learnings in the workflow

**3. Keep workflows current.** Don't create or overwrite workflows without asking.

## File Structure

```
.tmp/           # Temporary files, CSVs, intermediate data. Disposable.
tools/          # Python scripts for deterministic execution
workflows/      # Markdown SOPs
.env            # API keys (NEVER store secrets anywhere else)
```

## Disclaimer

This tool provides technical analysis only. It is NOT financial advice. No liability for trading decisions.
