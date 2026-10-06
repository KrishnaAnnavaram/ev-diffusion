"""Forecasters with one interface: ``fit(y, z=None)`` then ``predict(h, z_future=None)``.

All forecasters predict *new adoptions per period*. The backtest compares the
diffusion models with these simple baselines, so a diffusion model must beat a
naive rule before its forecast has any value.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .bass import fit_bass
from .gbm import fit_gbm


class Forecaster:
    name = "base"

    def fit(self, y: np.ndarray, z: np.ndarray | None = None) -> "Forecaster":
        raise NotImplementedError

    def predict(self, h: int, z_future: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class NaiveForecaster(Forecaster):
    """Repeat the last observed value."""

    name = "naive"

    def fit(self, y, z=None):
        self.last_ = float(np.asarray(y, float)[-1])
        return self

    def predict(self, h, z_future=None):
        return np.full(h, self.last_)


class DriftForecaster(Forecaster):
    """Continue the average change per period (the random walk with drift)."""

    name = "drift"

    def fit(self, y, z=None):
        y = np.asarray(y, float)
        self.last_ = float(y[-1])
        self.slope_ = float((y[-1] - y[0]) / (y.size - 1)) if y.size > 1 else 0.0
        return self

    def predict(self, h, z_future=None):
        return np.maximum(self.last_ + self.slope_ * np.arange(1, h + 1), 0.0)


class ExponentialForecaster(Forecaster):
    """Log-linear trend on the last ``window`` positive values."""

    name = "exponential"

    def __init__(self, window: int = 6):
        self.window = window

    def fit(self, y, z=None):
        y = np.asarray(y, float)
        t = np.arange(y.size, dtype=float)
        keep = y > 0
        t, logs = t[keep][-self.window :], np.log(y[keep][-self.window :])
        if t.size < 2:
            raise ValueError("exponential trend needs at least 2 positive values")
        self.slope_, self.intercept_ = np.polyfit(t, logs, 1)
        self.n_ = y.size
        return self

    def predict(self, h, z_future=None):
        t = np.arange(self.n_, self.n_ + h, dtype=float)
        return np.exp(self.intercept_ + self.slope_ * t)


class LogisticForecaster(Forecaster):
    """Three-parameter logistic curve on cumulative adoptions."""

    name = "logistic"

    def fit(self, y, z=None):
        y = np.asarray(y, float)
        cum = np.cumsum(y)
        total = cum[-1]
        t = np.arange(1, y.size + 1, dtype=float)

        def resid(theta):
            scale, rate, mid = theta
            return scale * total / (1 + np.exp(-rate * (t - mid))) - cum

        best = None
        for scale in (1.5, 4.0, 15.0):
            for mid in (y.size, 2 * y.size):
                result = least_squares(
                    resid, [scale, 0.5, mid], bounds=([1.0, 1e-3, -50.0], [1000.0, 5.0, 500.0]), x_scale="jac"
                )
                if best is None or result.cost < best.cost:
                    best = result
        self.params_ = (best.x[0] * total, best.x[1], best.x[2])
        self.n_ = y.size
        return self

    def _cum(self, t):
        k, r, mid = self.params_
        return k / (1 + np.exp(-r * (np.asarray(t, float) - mid)))

    def predict(self, h, z_future=None):
        t = np.arange(self.n_, self.n_ + h + 1, dtype=float)
        return np.diff(self._cum(t))


class BassForecaster(Forecaster):
    name = "bass"

    def __init__(self, mode: str = "period"):
        self.mode = mode

    def fit(self, y, z=None):
        self.fit_ = fit_bass(y, mode=self.mode)
        self.n_ = len(y)
        return self

    def predict(self, h, z_future=None):
        return self.fit_.predict_period(self.n_ + h)[self.n_ :]


class GBMForecaster(Forecaster):
    """GBM forecast. It needs the covariate rows of the forecast periods."""

    name = "gbm"

    def __init__(self, names: list[str] | None = None):
        self.names = names

    def fit(self, y, z=None):
        if z is None:
            raise ValueError("GBMForecaster needs covariates")
        self.z_ = np.asarray(z, float)
        self.fit_ = fit_gbm(y, self.z_, names=self.names)
        self.n_ = len(y)
        return self

    def predict(self, h, z_future=None):
        if z_future is None or len(z_future) < h:
            raise ValueError("GBM forecast needs covariate rows for every forecast period")
        z_all = np.vstack([self.z_, np.asarray(z_future, float)[:h]])
        return self.fit_.predict_period(self.n_ + h, z_all=z_all)[self.n_ :]


def default_forecasters(with_gbm: bool = False, names: list[str] | None = None) -> dict:
    """Factories for the models in the backtest. Each call gives a new, unfitted model."""
    factories = {
        "naive": NaiveForecaster,
        "drift": DriftForecaster,
        "exponential": ExponentialForecaster,
        "logistic": LogisticForecaster,
        "bass": BassForecaster,
        "bass_cumulative": lambda: BassForecaster(mode="cumulative"),
    }
    if with_gbm:
        factories["gbm"] = lambda: GBMForecaster(names=names)
    return factories
