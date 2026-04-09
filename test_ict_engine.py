"""End-to-End Test der ICT Unicorn Detection Engine mit echten Marktdaten.

Laedt Daten via yfinance, fuehrt die komplette Pipeline durch und
gibt alle Zwischenergebnisse aus. Kein Schritt wird uebersprungen,
keine Daten werden erfunden.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd

from tools.ict.swing_levels import get_swing_levels
from tools.ict.liquidity_sweep import detect_liquidity_sweeps
from tools.ict.displacement import detect_displacement, detect_mss, calculate_atr
from tools.ict.config import DISPLACEMENT_MULT_BY_TIMEFRAME, ATR_DISPLACEMENT_MULT
from tools.ict.fvg import detect_fvgs
from tools.ict.breaker_block import detect_breaker_blocks
from tools.ict.unicorn_zone import calculate_unicorn_zones
from tools.ict.signal_engine import analyze


def run_test(ticker: str, period: str, interval: str):
    """Fuehrt einen vollstaendigen Testdurchlauf durch."""
    print(f"\n{'='*70}")
    print(f"ICT UNICORN ENGINE TEST — {ticker} ({interval}, {period})")
    print(f"{'='*70}\n")

    # --- Daten laden ---
    print("[1/8] Lade Marktdaten via yfinance...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)

    if data.empty:
        print(f"FEHLER: Keine Daten fuer {ticker} erhalten.")
        return

    # yfinance gibt MultiIndex-Spalten zurück bei single ticker — flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    print(f"  >{len(data)} Kerzen geladen")
    print(f"  >Zeitraum: {data.index[0]} bis {data.index[-1]}")
    print(f"  >Aktueller Preis (letzter Close): {data['Close'].iloc[-1]:.2f}")
    print()

    # --- Swing Levels ---
    print("[2/8] Erkenne Swing Levels...")
    swings = get_swing_levels(data)
    print(f"  >{len(swings)} Swing Levels erkannt")
    if not swings.empty:
        highs = swings[swings["type"] == "high"]
        lows = swings[swings["type"] == "low"]
        print(f"     {len(highs)} Swing Highs, {len(lows)} Swing Lows")
        if not highs.empty:
            print(f"     Letztes Swing High: {highs.iloc[-1]['price']:.2f} bei {highs.iloc[-1]['index']}")
        if not lows.empty:
            print(f"     Letztes Swing Low:  {lows.iloc[-1]['price']:.2f} bei {lows.iloc[-1]['index']}")
    print()

    # --- Liquidity Sweeps ---
    print("[3/8] Suche Liquidity Sweeps...")
    sweeps = detect_liquidity_sweeps(data, swing_levels=swings)
    print(f"  >{len(sweeps)} Sweeps gefunden")
    for s in sweeps[-3:]:  # Letzte 3 anzeigen
        print(f"     [{s['sweep_type'].upper()}] bei {s['sweep_index']}: "
              f"Level {s['swept_price']:.2f}")
    print()

    # --- Displacement ---
    atr_mult = DISPLACEMENT_MULT_BY_TIMEFRAME.get(interval, ATR_DISPLACEMENT_MULT)
    print(f"[4/8] Suche Displacement-Kerzen (Schwellenwert: {atr_mult}x ATR fuer {interval})...")
    disp_df = detect_displacement(data, atr_mult=atr_mult)
    disp_count = int(disp_df["is_displacement"].sum())
    print(f"  >{disp_count} Displacement-Kerzen erkannt")
    if disp_count > 0:
        last_disp = disp_df[disp_df["is_displacement"]].iloc[-1]
        print(f"     Letzte: Richtung={last_disp['displacement_direction']}, "
              f"ATR-Ratio={last_disp['body_atr_ratio']:.1f}x")
    print()

    # --- Market Structure Shift ---
    print("[5/8] Suche Market Structure Shifts (MSS)...")
    mss_events = detect_mss(data, disp_df, swings, atr_mult)
    print(f"  >{len(mss_events)} MSS erkannt")
    for mss in mss_events[-3:]:
        print(f"     [{mss['direction'].upper()}] bei {mss['mss_index']}: "
              f"brach Level {mss['broken_level_price']:.2f}, "
              f"ATR-Ratio={mss['body_atr_ratio']:.1f}x")
    print()

    # --- Fair Value Gaps ---
    print("[6/8] Suche Fair Value Gaps...")
    fvgs = detect_fvgs(data)
    print(f"  >{len(fvgs)} FVGs gefunden")
    for fvg in fvgs[-3:]:
        print(f"     [{fvg['direction'].upper()}] "
              f"{fvg['fvg_low']:.2f} – {fvg['fvg_high']:.2f} "
              f"(Groesse: {fvg['gap_atr_ratio']:.1f}x ATR)")
    print()

    # --- Breaker Blocks ---
    print("[7/8] Suche Breaker Blocks...")
    bbs = detect_breaker_blocks(data, sweeps, mss_events)
    print(f"  >{len(bbs)} Breaker Blocks gefunden")
    for bb in bbs[-3:]:
        print(f"     [{bb['direction'].upper()}] "
              f"{bb['bb_low']:.2f} – {bb['bb_high']:.2f}")
    print()

    # --- Unicorn Zones ---
    print("[8/8] Berechne Unicorn Zones (FVG + BB)...")
    zones = calculate_unicorn_zones(fvgs, bbs)
    print(f"  >{len(zones)} Unicorn Zones gefunden")
    for z in zones[:3]:
        print(f"     [{z['direction'].upper()}] "
              f"{z['zone_low']:.2f} – {z['zone_high']:.2f} "
              f"(Ueberlappung: {z['overlap_pct']:.0f}%)")
    print()

    # --- Vollstaendige Analyse ---
    print(f"\n{'='*70}")
    print("VOLLSTAENDIGE ANALYSE (signal_engine.analyze)")
    print(f"{'='*70}\n")

    result = analyze(data, capital=10000.0, timeframe=interval)

    print(f"Status: {result['status']}")
    print(f"\nZusammenfassung:\n{result['zusammenfassung']}")

    if result["begruendung"]:
        print(f"\nBegruendung:")
        for step in result["begruendung"]:
            print(f"  {step}")

    if result["warnungen"]:
        print(f"\nWarnungen:")
        for w in result["warnungen"]:
            print(f"  [!]{w}")

    if result["signal"]:
        sig = result["signal"]
        print(f"\n--- SIGNAL DETAILS ---")
        print(f"  Richtung:     {sig['direction_display']}")
        print(f"  Entry-Zone:   {sig['entry_zone_low']:.2f} – {sig['entry_zone_high']:.2f}")
        print(f"  Optimaler Entry: {sig['entry_optimal']:.2f}")
        print(f"  Stop-Loss:    {sig['stop_loss']}")
        print(f"  Take-Profit:  {sig['take_profit']}")
        print(f"  R:R:          1:{sig['rr_ratio']}")
        print(f"  Risiko:       {sig['risk_amount']} EUR")
        print(f"  Position:     {sig['position_size_units']} Einheiten")

    # Detail-Statistiken
    d = result["details"]
    print(f"\n--- DETAIL-STATISTIKEN ---")
    print(f"  Swing Levels:       {d.get('swing_level_count', 0)}")
    print(f"  Liquidity Sweeps:   {d.get('sweep_count', 0)}")
    print(f"  Displacements:      {d.get('displacement_count', 0)}")
    print(f"  MSS Events:         {d.get('mss_count', 0)}")
    print(f"  Fair Value Gaps:    {d.get('fvg_count', 0)}")
    print(f"  Breaker Blocks:     {d.get('bb_count', 0)}")
    print(f"  Unicorn Zones:      {d.get('unicorn_zone_count', 0)}")


if __name__ == "__main__":
    # Test mit verschiedenen Instrumenten und Zeitrahmen
    # NQ Futures (Nasdaq 100)
    run_test("NQ=F", period="60d", interval="1d")

    # S&P 500 ETF (hohe Liquiditaet, gute Datenqualitaet)
    run_test("SPY", period="6mo", interval="1d")

    # EUR/USD (Forex)
    run_test("EURUSD=X", period="3mo", interval="1d")
