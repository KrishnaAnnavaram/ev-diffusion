"""Inference helpers for least-squares curve fits.

A diffusion curve fitted to about 15 yearly points has autocorrelated residuals.
The classic covariance ``s^2 (J'J)^-1`` then gives standard errors that are too
small. This module adds the Newey-West (HAC) covariance and residual diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import stats


def numeric_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rel_step: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of ``fun`` at ``x`` (rows = observations)."""
    x = np.asarray(x, dtype=float)
    base = np.asarray(fun(x), dtype=float)
    jac = np.empty((base.size, x.size))
    for j in range(x.size):
        step = rel_step * max(abs(x[j]), 1e-8)
        up, down = x.copy(), x.copy()
        up[j] += step
        down[j] -= step
        jac[:, j] = (np.asarray(fun(up), float) - np.asarray(fun(down), float)) / (2 * step)
    return jac


def information_criteria(sse: float, n: int, k: int) -> dict[str, float]:
    """Gaussian AIC, AICc and BIC for a least-squares fit with ``k`` parameters."""
    sse = max(float(sse), 1e-300)
    loglik_part = n * math.log(sse / n)
    aic = loglik_part + 2 * k
    aicc = aic + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
    bic = loglik_part + k * math.log(n)
    return {"aic": aic, "aicc": aicc, "bic": bic}


def _column_scale(jac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale each Jacobian column to unit length. Parameters such as p (about 0.001)
    and m (about 1e6) differ by many orders, and J'J is then badly scaled."""
    norms = np.linalg.norm(jac, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    return jac / norms, norms


def ols_covariance(jac: np.ndarray, resid: np.ndarray) -> np.ndarray:
    n, k = jac.shape
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    scaled, norms = _column_scale(jac)
    cov_scaled = s2 * np.linalg.pinv(scaled.T @ scaled)
    return cov_scaled / np.outer(norms, norms)


def newey_west_lags(n: int) -> int:
    return max(1, int(math.floor(4 * (n / 100) ** (2 / 9))))


def hac_covariance(jac: np.ndarray, resid: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West heteroscedasticity and autocorrelation consistent covariance."""
    n, k = jac.shape
    lags = newey_west_lags(n) if lags is None else int(lags)
    scaled, norms = _column_scale(jac)
    bread = np.linalg.pinv(scaled.T @ scaled)
    scores = scaled * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    correction = n / max(n - k, 1)
    return correction * (bread @ meat @ bread) / np.outer(norms, norms)


def acf(resid: np.ndarray, nlags: int) -> np.ndarray:
    resid = np.asarray(resid, float) - np.mean(resid)
    denom = float(resid @ resid)
    if denom == 0:
        return np.zeros(nlags)
    return np.array([float(resid[lag:] @ resid[:-lag]) / denom for lag in range(1, nlags + 1)])


def durbin_watson(resid: np.ndarray) -> float:
    resid = np.asarray(resid, float)
    denom = float(resid @ resid)
    return float(np.sum(np.diff(resid) ** 2) / denom) if denom else float("nan")


def ljung_box(resid: np.ndarray, lags: int | None = None, fitted_params: int = 0) -> tuple[float, float, int]:
    """Ljung-Box Q statistic, its p-value and the degrees of freedom."""
    resid = np.asarray(resid, float)
    n = resid.size
    lags = lags or max(1, min(10, n // 4))
    rho = acf(resid, lags)
    q_stat = n * (n + 2) * float(np.sum(rho**2 / (n - np.arange(1, lags + 1))))
    dof = max(lags - fitted_params, 1)
    return q_stat, float(stats.chi2.sf(q_stat, dof)), dof


def residual_diagnostics(resid: np.ndarray) -> dict[str, float]:
    q_stat, p_value, dof = ljung_box(resid)
    return {
        "lag1_autocorrelation": float(acf(resid, 1)[0]),
        "durbin_watson": durbin_watson(resid),
        "ljung_box_q": q_stat,
        "ljung_box_p": p_value,
        "ljung_box_dof": float(dof),
    }


def parameter_table(names, estimates, cov: np.ndarray, n: int) -> list[dict[str, float | str]]:
    """Estimate, standard error, t value, two-sided p-value and 95 % interval for each parameter."""
    k = len(names)
    dof = max(n - k, 1)
    crit = float(stats.t.ppf(0.975, dof))
    rows = []
    for i, name in enumerate(names):
        se = float(math.sqrt(max(cov[i, i], 0.0)))
        est = float(estimates[i])
        t_val = est / se if se > 0 else math.inf
        rows.append(
            {
                "parameter": name,
                "estimate": est,
                "std_error": se,
                "t_value": t_val,
                "p_value": float(2 * stats.t.sf(abs(t_val), dof)) if se > 0 else 0.0,
                "ci_low": est - crit * se,
                "ci_high": est + crit * se,
            }
        )
    return rows


def condition_number(jac: np.ndarray) -> float:
    """Condition number of the column-scaled Jacobian. A large value means weak identification."""
    norms = np.linalg.norm(jac, axis=0)
    norms[norms == 0] = 1.0
    singular = np.linalg.svd(jac / norms, compute_uv=False)
    return float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
