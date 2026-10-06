"""Column contracts for input tables.

Every loader maps known header aliases onto one canonical name, checks that the
required columns exist, converts types and rejects rows that break the contract.
A broken file stops the run with a clear message. It never produces a silent NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SchemaError(ValueError):
    """The input table does not match its column contract."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "int", "float" or "str"
    required: bool = True
    aliases: tuple[str, ...] = ()
    min_value: float | None = None
    allow_missing: bool = False


def _key(label: str) -> str:
    return "".join(ch for ch in str(label).strip().lower() if ch.isalnum())


def conform(frame: pd.DataFrame, columns: tuple[Column, ...], source: str) -> pd.DataFrame:
    """Return a copy of ``frame`` with canonical column names and checked types."""
    lookup: dict[str, str] = {}
    for col in frame.columns:
        lookup.setdefault(_key(col), col)

    out = pd.DataFrame(index=frame.index)
    for spec in columns:
        found = None
        for candidate in (spec.name, *spec.aliases):
            if _key(candidate) in lookup:
                found = lookup[_key(candidate)]
                break
        if found is None:
            if spec.required:
                names = ", ".join((spec.name, *spec.aliases))
                raise SchemaError(f"{source}: missing required column {spec.name!r} (accepted names: {names})")
            continue
        out[spec.name] = _convert(frame[found], spec, source)
    return out.reset_index(drop=True)


def _convert(series: pd.Series, spec: Column, source: str) -> pd.Series:
    if spec.kind == "str":
        values = series.astype("string").str.strip()
        bad = values.isna() | (values == "")
        if bad.any() and not spec.allow_missing:
            rows = (np.flatnonzero(bad.to_numpy()) + 2)[:5].tolist()
            raise SchemaError(f"{source}: column {spec.name!r} is empty in {int(bad.sum())} row(s), first file rows {rows}")
        return values

    numbers = pd.to_numeric(series, errors="coerce")
    bad = numbers.isna() & ~(series.isna() & spec.allow_missing)
    if bad.any():
        rows = (np.flatnonzero(bad.to_numpy()) + 2)[:5].tolist()
        raise SchemaError(f"{source}: column {spec.name!r} is not numeric in {int(bad.sum())} row(s), first file rows {rows}")
    if spec.min_value is not None:
        low = numbers < spec.min_value
        if low.any():
            rows = (np.flatnonzero(low.to_numpy()) + 2)[:5].tolist()
            raise SchemaError(f"{source}: column {spec.name!r} is below {spec.min_value} in rows {rows}")
    if spec.kind == "int":
        if not np.allclose(numbers.dropna(), np.round(numbers.dropna())):
            raise SchemaError(f"{source}: column {spec.name!r} must hold whole numbers")
        return numbers.round().astype("Int64")
    return numbers.astype(float)


def read_table(path) -> pd.DataFrame:
    """Read a CSV file, or an Excel file when the optional ``excel`` extra is installed."""
    text = str(path).lower()
    if text.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(path)
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("Excel input needs the optional extra: pip install 'ev-diffusion[excel]'") from exc
    return pd.read_csv(path)
