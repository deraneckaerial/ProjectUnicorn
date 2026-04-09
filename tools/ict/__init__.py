"""ICT Unicorn Model Detection Engine.

Deterministische Erkennung von ICT Smart Money Konzepten:
- Swing Levels (lokale Hochs/Tiefs)
- Liquidity Sweeps (Stop Hunts)
- Displacement + Market Structure Shift
- Fair Value Gaps (Preis-Ineffizienzen)
- Breaker Blocks (invalidierte Order Blocks)
- Unicorn Zones (FVG-BB-Konfluenz)
- Kill Zone Zeitfilter

Alle Module nehmen ein pandas DataFrame mit OHLCV-Spalten entgegen
und geben strukturierte Ergebnisse zurueck. Kein LLM, keine API-Calls,
keine Schaetzungen — nur exakte mathematische Bedingungen.
"""
