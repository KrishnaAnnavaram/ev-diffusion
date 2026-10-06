"""The Generalized Bass Model (Bass, Krishnan and Jain, 1994).

The GBM keeps the Bass shape but runs the diffusion clock at speed
``x(i) = 1 + sum_j beta_j * z_j(i)``, where ``z_j(i)`` is the log change of
covariate ``j`` (for example charging ports or an incentive index) in period ``i``:

``X(i) = X(i-1) + x(i)``, ``F_GBM(i) = F_Bass(X(i))``.

A multiplier on ``m`` is *not* a GBM: ``g * m`` is one parameter. Two more
cases are not identified and the fit refuses them:

* no covariate at all;
* a covariate with a constant log change. Then ``x(i)`` is constant and only
  rescales ``p`` and ``q``.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from . import inference
from .bass import DiffusionFit, as_counts, bass_cdf, fit_bass, to_target


class GBMIdentificationError(ValueError):
    """The covariates cannot identify the GBM parameters."""


def check_identifiable(z: np.ndarray, names: list[str] | None = None) -> None:
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] == 0:
        raise GBMIdentificationError("the GBM needs at least one covariate; without one it is the Bass model")
    names = names or [f"z{j}" for j in range(z.shape[1])]
    for j, name in enumerate(names):
        if np.std(z[:, j]) < 1e-9:
            raise GBMIdentificationError(
                f"covariate {name!r} has a constant log change: its beta is confounded with p and q"
            )
    design = np.column_stack([np.ones(z.shape[0]), z])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise GBMIdentificationError("the covariate log changes are collinear with each other or with a constant")


def gbm_period(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """New adoptions for each row of ``z`` with parameters ``(p, q, m, beta_1..beta_J)``."""
    p, q, m = x[0], x[1], x[2]
    betas = np.asarray(x[3:], dtype=float)
    speed = np.maximum(1.0 + np.asarray(z, float) @ betas, 1e-6)
    clock = np.concatenate([[0.0], np.cumsum(speed)])
    return m * np.diff(bass_cdf(clock, p, q))


def fit_gbm(y, z, names: list[str] | None = None, mode: str = "period", beta_bound: float = 10.0) -> DiffusionFit:
    """Fit p, q, m and one beta per covariate by bounded least squares."""
    y = as_counts(y)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if z.shape[0] != y.size:
        raise ValueError(f"z has {z.shape[0]} rows but y has {y.size} periods")
    names = list(names) if names else [f"z{j}" for j in range(z.shape[1])]
    check_identifiable(z, names)

    n, n_cov = y.size, z.shape[1]
    total = float(y.sum())
    target = to_target(y, mode)

    def resid(theta: np.ndarray) -> np.ndarray:
        x = np.concatenate([[theta[0], theta[1], theta[2] * total], theta[3:]])
        return to_target(gbm_period(x, z), mode) - target

    lower = np.concatenate([[1e-6, 1e-6, 1.0], np.full(n_cov, -beta_bound)])
    upper = np.concatenate([[0.5, 3.0, 1000.0], np.full(n_cov, beta_bound)])
    bass = fit_bass(y, mode=mode)
    base = np.array([bass.x[0], bass.x[1], bass.x[2] / total])
    starts = [np.concatenate([base, np.full(n_cov, b)]) for b in (0.0, 0.5, -0.5, 1.5)]
    starts += [np.concatenate([[p, q, s], np.zeros(n_cov)]) for p in (0.003, 0.02) for q in (0.3, 0.7) for s in (2.0, 8.0)]

    best = None
    for start in starts:
        start = np.clip(start, lower + 1e-9, upper - 1e-9)
        result = least_squares(resid, start, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=4000)
        if best is None or result.cost < best.cost:
            best = result

    x = np.concatenate([[best.x[0], best.x[1], best.x[2] * total], best.x[3:]])

    def curve(params: np.ndarray, n_periods: int, z_all: np.ndarray | None = None) -> np.ndarray:
        design = z if z_all is None else np.asarray(z_all, float)
        if design.shape[0] < n_periods:
            raise ValueError("GBM prediction needs covariate rows for every period")
        return gbm_period(params, design[:n_periods])

    jac = inference.numeric_jacobian(lambda v: to_target(gbm_period(v, z), mode), x)
    fit = DiffusionFit(
        model="gbm",
        mode=mode,
        names=("p", "q", "m", *[f"beta_{name}" for name in names]),
        x=x,
        observed=target,
        fitted=to_target(gbm_period(x, z), mode),
        jacobian=jac,
        curve=curve,
        success=bool(best.success),
    )
    cond = inference.condition_number(jac)
    if cond > 1e6:
        fit.notes.append(f"Jacobian condition number {cond:.2e}: the parameters are weakly identified")
    return fit


def compare_nested(bass: DiffusionFit, gbm: DiffusionFit) -> dict[str, float]:
    """F test of the GBM against the Bass model (the Bass model is the GBM with all beta = 0)."""
    if bass.n != gbm.n or bass.mode != gbm.mode:
        raise ValueError("the two fits must use the same data and the same mode")
    extra = gbm.k - bass.k
    dof = gbm.n - gbm.k
    if dof <= 0:
        raise ValueError("not enough periods for the F test")
    sse_b, sse_g = bass.sse, gbm.sse
    f_stat = ((sse_b - sse_g) / extra) / (sse_g / dof) if sse_g > 0 else float("inf")
    p_value = float(stats.f.sf(max(f_stat, 0.0), extra, dof))
    cb, cg = bass.criteria(), gbm.criteria()
    return {
        "f_statistic": float(f_stat),
        "f_p_value": p_value,
        "delta_aic": cg["aic"] - cb["aic"],
        "delta_aicc": cg["aicc"] - cb["aicc"],
        "delta_bic": cg["bic"] - cb["bic"],
    }
