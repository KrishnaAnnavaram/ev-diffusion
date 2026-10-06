"""Rolling-origin backtest.

For each origin ``o`` (from ``min_train`` to ``n - 1``) every model is fitted on
periods ``0..o-1`` only and forecasts periods ``o..o+h-1``. The errors are
grouped by model and horizon. A model never sees a value at or after its origin.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .diffusion.baselines import Forecaster


def rolling_origin(
    y,
    factories: dict[str, Callable[[], Forecaster]],
    years=None,
    z: np.ndarray | None = None,
    min_train: int = 8,
    horizon: int = 3,
) -> pd.DataFrame:
    """Return one row per (origin, model, horizon) with the actual and the forecast value."""
    y = np.asarray(y, dtype=float)
    n = y.size
    years = np.arange(n) if years is None else np.asarray(years)
    if min_train < 4 or min_train >= n:
        raise ValueError(f"min_train must be in [4, {n - 1}], got {min_train}")
    rows = []
    for origin in range(min_train, n):
        steps = min(horizon, n - origin)
        train = y[:origin].copy()
        z_train = None if z is None else z[:origin]
        z_future = None if z is None else z[origin : origin + steps]
        for name, make in factories.items():
            status, forecast = "ok", np.full(steps, np.nan)
            try:
                model = make().fit(train, z_train)
                forecast = np.asarray(model.predict(steps, z_future), dtype=float)
            except Exception as exc:  # a failed fit is a result, not a crash
                status = f"failed: {type(exc).__name__}"
            for h in range(steps):
                actual = y[origin + h]
                rows.append(
                    {
                        "origin_year": int(years[origin - 1]),
                        "target_year": int(years[origin + h]),
                        "model": name,
                        "horizon": h + 1,
                        "actual": actual,
                        "forecast": forecast[h],
                        "error": forecast[h] - actual,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE, MAPE and sMAPE per model and horizon, plus the number of failed fits."""

    def _agg(group: pd.DataFrame) -> pd.Series:
        ok = group[group["status"] == "ok"]
        err = ok["error"].to_numpy(float)
        act = ok["actual"].to_numpy(float)
        fc = ok["forecast"].to_numpy(float)
        nonzero = act != 0
        denom = np.abs(act) + np.abs(fc)
        return pd.Series(
            {
                "n": int(len(ok)),
                "failed": int((group["status"] != "ok").sum()),
                "mae": float(np.mean(np.abs(err))) if err.size else np.nan,
                "rmse": float(np.sqrt(np.mean(err**2))) if err.size else np.nan,
                "mape_pct": float(100 * np.mean(np.abs(err[nonzero] / act[nonzero]))) if nonzero.any() else np.nan,
                "smape_pct": float(100 * np.mean(2 * np.abs(err) / np.where(denom > 0, denom, 1))) if err.size else np.nan,
            }
        )

    rows = []
    for (model, horizon), group in results.groupby(["model", "horizon"], sort=True):
        row = _agg(group)
        row["model"], row["horizon"] = model, horizon
        rows.append(row)
    table = pd.DataFrame(rows)
    return table[["model", "horizon", "n", "failed", "mae", "rmse", "mape_pct", "smape_pct"]]


def ranking(summary: pd.DataFrame, metric: str = "mape_pct") -> pd.DataFrame:
    """Mean of ``metric`` over horizons, best model first."""
    table = summary.groupby("model")[metric].mean().sort_values().reset_index()
    table.columns = ["model", f"mean_{metric}"]
    return table
