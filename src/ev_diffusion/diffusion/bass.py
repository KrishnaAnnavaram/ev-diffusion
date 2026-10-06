"""The Bass diffusion model.

``F(t) = (1 - exp(-(p+q) t)) / (1 + (q/p) exp(-(p+q) t))`` is the share of the
final market ``m`` that adopted by time ``t``. The adoptions in period ``i`` are
``m (F(i) - F(i-1))``. That is the exact discrete form, not the density at a point.

The fit supports two targets:

* ``period``: the new adoptions of each year (default, less autocorrelated).
* ``cumulative``: the running total of adoptions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import inference

MODES = ("period", "cumulative")
BASS_NAMES = ("p", "q", "m")


def bass_cdf(t, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    decay = np.exp(-(p + q) * t)
    return (1.0 - decay) / (1.0 + (q / p) * decay)


def bass_pdf(t, p: float, q: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    decay = np.exp(-(p + q) * t)
    return ((p + q) ** 2 / p) * decay / (1.0 + (q / p) * decay) ** 2


def bass_period(p: float, q: float, m: float, n_periods: int) -> np.ndarray:
    """New adoptions in periods 1..n_periods (period ``i`` covers time ``i-1`` to ``i``)."""
    grid = np.arange(0, n_periods + 1, dtype=float)
    return m * np.diff(bass_cdf(grid, p, q))


def peak_time(p: float, q: float) -> float:
    """Time of the highest adoption rate (the inflection of F). 0 if q <= p."""
    return math.log(q / p) / (p + q) if q > p else 0.0


def as_counts(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4:
        raise ValueError("a diffusion fit needs at least 4 periods")
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("adoption counts must be finite and not negative")
    if y.sum() <= 0:
        raise ValueError("adoption counts are all zero")
    return y


def to_target(period_values: np.ndarray, mode: str) -> np.ndarray:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return period_values if mode == "period" else np.cumsum(period_values)


@dataclass
class DiffusionFit:
    """The result of one least-squares diffusion fit."""

    model: str
    mode: str
    names: tuple[str, ...]
    x: np.ndarray
    observed: np.ndarray  # in the target space (period or cumulative)
    fitted: np.ndarray  # in the target space
    jacobian: np.ndarray
    curve: Callable[..., np.ndarray] = field(repr=False)
    success: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.observed.size)

    @property
    def k(self) -> int:
        return int(self.x.size)

    @property
    def params(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, self.x)}

    @property
    def residuals(self) -> np.ndarray:
        return self.observed - self.fitted

    @property
    def sse(self) -> float:
        return float(self.residuals @ self.residuals)

    def criteria(self) -> dict[str, float]:
        return inference.information_criteria(self.sse, self.n, self.k)

    def covariance(self, kind: str = "hac") -> np.ndarray:
        if kind == "ols":
            return inference.ols_covariance(self.jacobian, self.residuals)
        if kind == "hac":
            return inference.hac_covariance(self.jacobian, self.residuals)
        raise ValueError("kind must be 'ols' or 'hac'")

    def parameter_table(self, kind: str = "hac") -> list[dict]:
        return inference.parameter_table(self.names, self.x, self.covariance(kind), self.n)

    def diagnostics(self) -> dict[str, float]:
        out = inference.residual_diagnostics(self.residuals)
        out["condition_number"] = inference.condition_number(self.jacobian)
        return out

    def predict_period(self, n_periods: int, **kwargs) -> np.ndarray:
        """New adoptions for periods 1..n_periods with the fitted parameters."""
        return self.curve(self.x, n_periods, **kwargs)


def _bass_curve(x: np.ndarray, n_periods: int) -> np.ndarray:
    p, q, m = x
    return bass_period(p, q, m, n_periods)


def _bass_starts(total: float) -> list[np.ndarray]:
    return [
        np.array([p, q, s])
        for p in (0.002, 0.02)
        for q in (0.25, 0.7)
        for s in (1.5, 5.0, 25.0)
    ]


def fit_bass(y, mode: str = "period", x0=None, m_upper_factor: float = 1000.0) -> DiffusionFit:
    """Fit p, q and m by bounded least squares from several start points.

    ``m`` is optimised as a multiple of the observed total so all three
    parameters have a similar scale. ``m`` cannot be smaller than the observed total.
    """
    y = as_counts(y)
    n = y.size
    total = float(y.sum())
    target = to_target(y, mode)

    def resid(theta: np.ndarray) -> np.ndarray:
        p, q, scale = theta
        return to_target(bass_period(p, q, scale * total, n), mode) - target

    lower = np.array([1e-6, 1e-6, 1.0])
    upper = np.array([0.5, 3.0, m_upper_factor])
    if x0 is not None:
        x0 = np.asarray(x0, float)
        starts = [np.array([x0[0], x0[1], x0[2] / total])]
    else:
        starts = _bass_starts(total)

    best = None
    for start in starts:
        start = np.clip(start, lower * 1.0001, upper * 0.9999)
        result = least_squares(resid, start, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=800)
        if best is None or result.cost < best.cost:
            best = result

    x = np.array([best.x[0], best.x[1], best.x[2] * total])
    fitted = to_target(_bass_curve(x, n), mode)
    jac = inference.numeric_jacobian(lambda v: to_target(_bass_curve(v, n), mode), x)
    fit = DiffusionFit(
        model="bass",
        mode=mode,
        names=BASS_NAMES,
        x=x,
        observed=target,
        fitted=fitted,
        jacobian=jac,
        curve=_bass_curve,
        success=bool(best.success),
    )
    if best.x[2] >= 0.99 * m_upper_factor:
        fit.notes.append(f"m is at its upper bound ({m_upper_factor:g} x the observed total): the data do not identify m")
    fit.notes.extend(identification_notes(fit, n))
    return fit


def identification_notes(fit: DiffusionFit, n_periods: int) -> list[str]:
    """Warn when the data cannot pin down the market size ``m``."""
    notes = []
    p, q = fit.x[0], fit.x[1]
    peak = peak_time(p, q)
    if peak > n_periods:
        notes.append(
            f"the fitted peak is at period {peak:.1f}, after the last observed period {n_periods}: "
            "the series has not reached its inflection, so m is weakly identified and forecasts extrapolate"
        )
    table = fit.parameter_table("hac")
    m_row = next(r for r in table if r["parameter"] == "m")
    if m_row["std_error"] > 0.5 * abs(m_row["estimate"]):
        notes.append("the HAC standard error of m is more than 50 % of m")
    return notes
