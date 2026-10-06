"""Uncertainty from the data: a moving-block residual bootstrap for the Bass model.

1. Fit the Bass model and keep the residuals.
2. Make a new series: fitted values plus residual blocks drawn with replacement.
   Blocks keep the autocorrelation of neighbouring years.
3. Fit the Bass model again on the new series. Keep p, q, m and the forecast path.
4. Add resampled residuals to the forecast path for the prediction interval.

``relative`` residuals (``(y - fit) / fit``) scale with the level of the series.
That suits counts that grow by orders of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bass import DiffusionFit, fit_bass

RESIDUAL_KINDS = ("relative", "additive")


@dataclass
class BootstrapResult:
    params: np.ndarray  # (n_ok, 3): p, q, m
    mean_paths: np.ndarray  # (n_ok, horizon): forecast of the expected adoptions
    prediction_paths: np.ndarray  # (n_ok, horizon): forecast plus resampled noise
    n_requested: int
    n_failed: int
    block: int
    residual_kind: str

    def param_intervals(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        lo, hi = 50 * (1 - level), 50 * (1 + level)
        out = {}
        for i, name in enumerate(("p", "q", "m")):
            a, b = np.percentile(self.params[:, i], [lo, hi])
            out[name] = (float(a), float(b))
        return out

    def forecast_intervals(self, level: float = 0.95) -> dict[str, np.ndarray]:
        lo, hi = 50 * (1 - level), 50 * (1 + level)
        return {
            "mean_low": np.percentile(self.mean_paths, lo, axis=0),
            "mean_high": np.percentile(self.mean_paths, hi, axis=0),
            "pred_low": np.percentile(self.prediction_paths, lo, axis=0),
            "pred_high": np.percentile(self.prediction_paths, hi, axis=0),
        }


def _block_sample(resid: np.ndarray, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n = resid.size
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(length / block)))
    return np.concatenate([resid[s : s + block] for s in starts])[:length]


def bootstrap_bass(
    y,
    fit: DiffusionFit,
    horizon: int,
    n_boot: int = 200,
    block: int = 3,
    seed: int = 42,
    residual_kind: str = "relative",
) -> BootstrapResult:
    if fit.model != "bass" or fit.mode != "period":
        raise ValueError("bootstrap_bass needs a Bass fit in 'period' mode")
    if residual_kind not in RESIDUAL_KINDS:
        raise ValueError(f"residual_kind must be one of {RESIDUAL_KINDS}")
    y = np.asarray(y, float)
    n = y.size
    rng = np.random.default_rng(seed)
    fitted = fit.fitted
    future_mean = fit.predict_period(n + horizon)[n:]

    if residual_kind == "relative":
        safe = np.where(fitted > 0, fitted, np.nan)
        resid = (y - fitted) / safe
        resid = resid[np.isfinite(resid)]
    else:
        resid = y - fitted
    resid = resid - resid.mean()

    params, means, preds, failed = [], [], [], 0
    for _ in range(n_boot):
        draw = _block_sample(resid, n, block, rng)
        y_star = fitted * (1 + draw) if residual_kind == "relative" else fitted + draw
        y_star = np.maximum(y_star, 0.0)
        try:
            refit = fit_bass(y_star, mode="period", x0=fit.x)
        except (ValueError, FloatingPointError):
            failed += 1
            continue
        path = refit.predict_period(n + horizon)[n:]
        noise = _block_sample(resid, horizon, block, rng)
        pred = path * (1 + noise) if residual_kind == "relative" else path + noise
        params.append(refit.x)
        means.append(path)
        preds.append(np.maximum(pred, 0.0))

    if not params:
        raise RuntimeError("every bootstrap refit failed")
    return BootstrapResult(
        params=np.vstack(params),
        mean_paths=np.vstack(means),
        prediction_paths=np.vstack(preds),
        n_requested=n_boot,
        n_failed=failed,
        block=block,
        residual_kind=residual_kind,
    )


def point_forecast(fit: DiffusionFit, n_observed: int, horizon: int) -> np.ndarray:
    return fit.predict_period(n_observed + horizon)[n_observed:]
