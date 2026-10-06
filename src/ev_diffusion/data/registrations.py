"""Registration counts: loading, a coverage audit and a coverage-consistent yearly series.

State registration files start in different years. If you add every row of every
year, a part of the growth comes from new states that join the data, not from
adoption. This module builds the yearly series from one of two consistent sources:

* ``national``: the national rows, if they cover every year of the window.
* ``balanced``: the sum of the states that report in every year of the window.

National rows and state rows are never added together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import Column, SchemaError, conform, read_table

REGISTRATION_COLUMNS = (
    Column("year", "int", aliases=("registration_year", "yr")),
    Column("count", "float", aliases=("vehicle_count", "registrations", "vehicles"), min_value=0),
    Column("state", "str", required=False, aliases=("region", "geography", "state_code")),
    Column("make_model", "str", required=False, aliases=("vehicle_make_&_model", "vehicle_make_model", "model")),
)

NATIONAL_CODES = frozenset({"US", "USA", "NATIONAL", "UNITED STATES", "ALL STATES"})
UNKNOWN_STATE = "UNKNOWN"
POLICIES = ("auto", "national", "balanced")


@dataclass
class AdoptionSeries:
    """One yearly series of new registrations, with the decisions that made it."""

    years: np.ndarray
    counts: np.ndarray
    source: str
    states: tuple[str, ...]
    market: str = "US"
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"year": self.years, "count": self.counts})


def load_registrations(path) -> pd.DataFrame:
    """Read a registration file and return the columns year, state, make_model and count."""
    frame = conform(read_table(path), REGISTRATION_COLUMNS, source=str(path))
    return normalize_registrations(frame)


def normalize_registrations(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "state" not in frame:
        frame["state"] = UNKNOWN_STATE
    frame["state"] = frame["state"].astype("string").str.strip().str.upper()
    if "make_model" not in frame:
        frame["make_model"] = "ALL"
    frame["year"] = frame["year"].astype(int)
    if frame.empty:
        raise SchemaError("registration table is empty")
    return frame[["year", "state", "make_model", "count"]]


def is_national(state: str) -> bool:
    return str(state).upper() in NATIONAL_CODES


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """States in rows, years in columns, total registrations in cells (NaN = no data)."""
    pivot = frame.pivot_table(index="state", columns="year", values="count", aggfunc="sum")
    return pivot.sort_index()


def coverage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per state: first year, last year, years present, gap years and the total."""
    table = coverage_table(frame)
    rows = []
    all_years = list(table.columns)
    for state, values in table.iterrows():
        present = [y for y in all_years if not pd.isna(values[y])]
        first, last = min(present), max(present)
        gaps = [y for y in range(first, last + 1) if y not in present]
        rows.append(
            {
                "state": state,
                "national": is_national(state),
                "first_year": first,
                "last_year": last,
                "years_present": len(present),
                "gap_years": len(gaps),
                "total": float(np.nansum(values.to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def is_partial_last_year(totals: pd.Series, ratio: float = 0.6, lookback: int = 3) -> bool:
    """True if the final year falls below ``ratio`` x the year before, after growth.

    "After growth" means the year before the final year is at least as high as the
    year ``lookback`` years earlier. A sudden drop after growth is the typical sign
    of a year that the source has not finished. The rule is a heuristic. Use ``last_complete_year``
    to override it.
    """
    values = totals.sort_index().to_numpy(dtype=float)
    if len(values) < lookback + 2:
        return False
    growing = values[-2] >= values[-(lookback + 2)]
    return bool(growing and values[-1] < ratio * values[-2])


def best_balanced_window(table: pd.DataFrame, end_year: int, min_years: int) -> tuple[int, list[str]]:
    """Choose the start year that maximises (states x years) for states present in every year."""
    best: tuple[float, int, list[str]] | None = None
    years = [y for y in table.columns if y <= end_year]
    for start in years:
        span = [y for y in years if y >= start]
        if len(span) < min_years:
            continue
        complete = table[span].notna().all(axis=1)
        states = sorted(table.index[complete].tolist())
        if not states:
            continue
        score = len(states) * len(span)
        if best is None or score > best[0]:
            best = (score, start, states)
    if best is None:
        raise ValueError(f"no state reports in every year of any window with at least {min_years} years")
    return best[1], best[2]


def build_series(
    frame: pd.DataFrame,
    policy: str = "auto",
    last_complete_year: int | None = None,
    min_years: int = 8,
    market: str = "US",
) -> AdoptionSeries:
    """Build a coverage-consistent yearly series from registration rows."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    notes: list[str] = []
    frame = frame.copy()

    if last_complete_year is not None:
        frame = frame[frame["year"] <= last_complete_year]
        notes.append(f"years after {last_complete_year} removed (last complete year given)")
    else:
        totals = frame.groupby("year")["count"].sum()
        if is_partial_last_year(totals):
            dropped = int(totals.index.max())
            frame = frame[frame["year"] < dropped]
            notes.append(f"year {dropped} removed: it looks partial (sharp drop after a growth run)")

    national_mask = frame["state"].map(is_national)
    national = frame[national_mask]
    states = frame[~national_mask]
    if national_mask.any() and (~national_mask).any():
        notes.append("national rows and state rows are present: they are never added together")

    if policy in ("auto", "national") and not national.empty:
        yearly = national.groupby("year")["count"].sum().sort_index()
        full = range(int(yearly.index.min()), int(yearly.index.max()) + 1)
        missing = [y for y in full if y not in yearly.index]
        if not missing and len(yearly) >= min_years:
            notes.append(f"source: national rows, {yearly.index.min()}-{yearly.index.max()}")
            return AdoptionSeries(
                years=yearly.index.to_numpy(dtype=int),
                counts=yearly.to_numpy(dtype=float),
                source="national",
                states=tuple(sorted(national["state"].unique())),
                market=market,
                notes=notes,
            )
        if policy == "national":
            raise ValueError(f"national rows do not cover a full window: missing years {missing}")
        notes.append("national rows are incomplete: the balanced state panel is used")
    elif policy == "national":
        raise ValueError("policy 'national' needs rows with state code US or NATIONAL")

    if states.empty:
        raise ValueError("no state rows to build a balanced panel")
    if (states["state"] == UNKNOWN_STATE).all():
        notes.append("the file has no state column: coverage cannot be checked")

    table = coverage_table(states)
    end_year = int(table.columns.max())
    start, chosen = best_balanced_window(table, end_year, min_years)
    window = [y for y in table.columns if start <= y <= end_year]
    yearly = table.loc[chosen, window].sum(axis=0)
    dropped_states = sorted(set(table.index) - set(chosen))
    notes.append(f"source: balanced panel of {len(chosen)} state(s), {start}-{end_year}")
    if dropped_states:
        notes.append(f"states left out (not present in every year): {', '.join(dropped_states)}")
    return AdoptionSeries(
        years=np.array(window, dtype=int),
        counts=yearly.to_numpy(dtype=float),
        source="balanced",
        states=tuple(chosen),
        market=market,
        notes=notes,
    )
