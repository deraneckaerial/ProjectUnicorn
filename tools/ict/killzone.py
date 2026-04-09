"""Kill Zone und ICT Macro Zeitfilter.

Die Wahrscheinlichkeit eines erfolgreichen ICT-Setups steigt erheblich,
wenn es innerhalb definierter Liquiditaetsfenster auftritt. Ausserhalb
dieser Zeiten ist der Markt oft von geringem Volumen und algorithmischem
Rauschen gepraegt.

Kill Zones sind breite Zeitfenster (2-3 Stunden) rund um die Oeffnung
grosser Finanzmaerkte. ICT Macros sind praezise 20-30-Minuten-Fenster,
in denen institutionelle Algorithmen Preisauslieferungen vornehmen.

Alle Zeiten sind in EST (Eastern Standard Time / New York Time) definiert.
Der Filter konvertiert automatisch wenn die Daten in UTC vorliegen.

HINWEIS: Dieser Filter ist OPTIONAL. Bei Daily/Weekly-Charts oder
Maerkten ohne klare Session-Struktur kann er deaktiviert werden.
"""

from datetime import datetime, time
from typing import Optional

import pandas as pd
import numpy as np

from tools.ict.config import KILLZONES, MACROS


def is_in_killzone(
    timestamp: datetime,
    killzone_name: Optional[str] = None,
) -> dict:
    """Prueft ob ein Zeitpunkt in einer ICT Kill Zone liegt.

    Args:
        timestamp: Der zu pruefende Zeitpunkt (muss EST sein oder
            wird als EST interpretiert).
        killzone_name: Optional — nur eine bestimmte Kill Zone pruefen.
            Wenn None, werden alle Kill Zones geprueft.

    Returns:
        Dict mit:
            'in_killzone': Boolean — liegt der Zeitpunkt in einer Kill Zone?
            'active_killzone': Name der aktiven Kill Zone oder None,
            'erklarung': Verstaendliche Erklaerung,
    """
    t = timestamp.time() if isinstance(timestamp, datetime) else timestamp

    zones_to_check = (
        {killzone_name: KILLZONES[killzone_name]}
        if killzone_name and killzone_name in KILLZONES
        else KILLZONES
    )

    for name, (sh, sm, eh, em) in zones_to_check.items():
        start = time(sh, sm)
        end = time(eh, em)

        if _time_in_range(t, start, end):
            label = _killzone_label(name)
            return {
                "in_killzone": True,
                "active_killzone": name,
                "erklarung": (
                    f"Aktive Kill Zone: {label}. "
                    f"In diesem Zeitfenster ist die institutionelle "
                    f"Liquiditaet hoch — Setups haben eine erhoehte "
                    f"Erfolgswahrscheinlichkeit."
                ),
            }

    return {
        "in_killzone": False,
        "active_killzone": None,
        "erklarung": (
            "Keine Kill Zone aktiv. Setups ausserhalb der Kill Zones "
            "haben eine deutlich niedrigere Erfolgswahrscheinlichkeit, "
            "da weniger institutionelle Liquiditaet vorhanden ist."
        ),
    }


def is_in_macro(timestamp: datetime) -> dict:
    """Prueft ob ein Zeitpunkt in einem ICT Macro-Fenster liegt.

    Macros sind praezise 20-30-Minuten-Fenster, in denen institutionelle
    Algorithmen Preisauslieferungen vornehmen. Setups innerhalb von
    Macros haben die hoechste statistische Zuverlaessigkeit.

    Args:
        timestamp: Der zu pruefende Zeitpunkt (EST).

    Returns:
        Dict mit:
            'in_macro': Boolean,
            'active_macro': Name des aktiven Macros oder None,
            'erklarung': Verstaendliche Erklaerung,
    """
    t = timestamp.time() if isinstance(timestamp, datetime) else timestamp

    for name, (sh, sm, eh, em) in MACROS.items():
        start = time(sh, sm)
        end = time(eh, em)

        if _time_in_range(t, start, end):
            label = _macro_label(name)
            return {
                "in_macro": True,
                "active_macro": name,
                "erklarung": (
                    f"Aktives ICT Macro: {label}. "
                    f"In diesem praezisen Zeitfenster finden algorithmische "
                    f"Preisauslieferungen statt. Setups die hier entstehen "
                    f"haben die hoechste Zuverlaessigkeit."
                ),
            }

    return {
        "in_macro": False,
        "active_macro": None,
        "erklarung": "Kein ICT Macro aktiv.",
    }


def filter_dataframe_by_killzone(
    df: pd.DataFrame,
    killzone_names: Optional[list] = None,
) -> pd.DataFrame:
    """Filtert einen DataFrame auf Kerzen innerhalb von Kill Zones.

    Nuetzlich um nur Kerzen zu analysieren, die in Hochliquiditaets-
    phasen entstanden sind.

    Args:
        df: OHLCV DataFrame mit DatetimeIndex.
        killzone_names: Optional — nur bestimmte Kill Zones beruecksichtigen.

    Returns:
        Gefilterter DataFrame (nur Kerzen in Kill Zones).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df  # Kann nicht filtern ohne Zeitstempel

    zones = (
        {k: KILLZONES[k] for k in killzone_names if k in KILLZONES}
        if killzone_names
        else KILLZONES
    )

    mask = np.zeros(len(df), dtype=bool)

    for _, (sh, sm, eh, em) in zones.items():
        start = time(sh, sm)
        end = time(eh, em)

        for i, idx in enumerate(df.index):
            t = idx.time()
            if _time_in_range(t, start, end):
                mask[i] = True

    return df[mask]


def _time_in_range(t: time, start: time, end: time) -> bool:
    """Prueft ob eine Uhrzeit in einem Bereich liegt.

    Beruecksichtigt Mitternachts-Ueberschreitung (z.B. Asian Kill Zone
    19:00 - 22:00 ist unkritisch, aber 22:00 - 02:00 wuerde
    Mitternacht ueberqueren).
    """
    if start <= end:
        return start <= t <= end
    else:
        # Mitternachts-Ueberschreitung
        return t >= start or t <= end


def _killzone_label(name: str) -> str:
    """Menschenlesbarer Name fuer Kill Zones."""
    labels = {
        "asian": "Asian Session (19:00-22:00 EST)",
        "london_open": "London Open (02:00-05:00 EST)",
        "new_york": "New York Session (07:00-10:00 EST)",
        "london_close": "London Close (10:00-12:00 EST)",
    }
    return labels.get(name, name)


def _macro_label(name: str) -> str:
    """Menschenlesbarer Name fuer ICT Macros."""
    labels = {
        "london_1": "London Macro 1 (02:33-03:00 EST)",
        "london_2": "London Macro 2 (04:03-04:30 EST)",
        "ny_am_1": "NY AM Macro 1 (08:50-09:10 EST)",
        "ny_am_2": "NY AM Macro 2 (09:50-10:10 EST)",
        "ny_am_3": "NY AM Macro 3 (10:50-11:10 EST)",
        "ny_lunch": "NY Lunch Macro (11:50-12:10 EST)",
        "ny_pm_1": "NY PM Macro 1 (13:10-13:40 EST)",
        "ny_last_hour": "Last Hour Macro (15:15-15:45 EST)",
    }
    return labels.get(name, name)
