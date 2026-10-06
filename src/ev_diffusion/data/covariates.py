"""Yearly market covariates for the Generalized Bass Model (GBM).

The GBM (Bass, Krishnan and Jain, 1994) changes the speed of the diffusion clock
with the *percentage change* of each covariate. This module turns a table of
positive yearly covariate levels into the design matrix ``Z`` of log changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import SchemaError, read_table


def load_covariates(path) -> pd.DataFrame:
    """Read a table with a ``year`` column and one column for each positive covariate."""
    frame = read_table(path)
    frame.columns = [str(c).strip() for c in frame.columns]
    if "year" not in frame.columns:
        raise SchemaError(f"{path}: covariate table needs a 'year' column")
    names = [c for c in frame.columns if c != "year"]
    if not names:
        raise SchemaError(f"{path}: covariate table has no covariate column")
    for name in names:
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().any() or (values <= 0).any():
            raise SchemaError(f"{path}: covariate {name!r} must be a positive number in every row")
        frame[name] = values.astype(float)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    if frame["year"].duplicated().any():
        raise SchemaError(f"{path}: covariate table has duplicate years")
    return frame.sort_values("year").reset_index(drop=True)


def design_matrix(covariates: pd.DataFrame, years) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``Z`` (len(years) x J) of log changes, the covariate names and notes.

    ``Z[i, j] = ln(c_j(year_i) / c_j(year_i - 1))``. If the year before the first
    year is not in the table, the first row is 0 and a note says so.
    """
    years = [int(y) for y in years]
    table = covariates.set_index("year")
    names = [c for c in table.columns]
    missing = [y for y in years if y not in table.index]
    if missing:
        raise SchemaError(f"covariates are missing for years {missing}")
    notes: list[str] = []
    rows = []
    for i, year in enumerate(years):
        prev = year - 1
        if prev in table.index:
            rows.append(np.log(table.loc[year, names].to_numpy(float) / table.loc[prev, names].to_numpy(float)))
        else:
            rows.append(np.zeros(len(names)))
            if i == 0:
                notes.append(f"no covariate row for {prev}: the first log change is set to 0")
    return np.vstack(rows), names, notes
