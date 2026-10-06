"""Reference problem 3: the uncertainty band used hand-picked standard deviations."""

import numpy as np
import pytest

from ev_diffusion.diffusion.bass import bass_period, fit_bass
from ev_diffusion.diffusion.uncertainty import bootstrap_bass, point_forecast


def _noisy(sd, seed=0, n=16):
    clean = bass_period(0.005, 0.45, 2e5, n)
    return clean * np.exp(np.random.default_rng(seed).normal(0, sd, n))


def _width(sd):
    y = _noisy(sd)
    fit = fit_bass(y)
    boot = bootstrap_bass(y, fit, horizon=3, n_boot=60, seed=1)
    iv = boot.forecast_intervals()
    return float(np.mean(iv["pred_high"] - iv["pred_low"]))


def test_interval_width_comes_from_the_residuals():
    assert _width(0.20) > 2 * _width(0.02)


def test_noiseless_series_gives_a_narrow_interval():
    y = bass_period(0.005, 0.45, 2e5, 16)
    fit = fit_bass(y)
    boot = bootstrap_bass(y, fit, horizon=2, n_boot=30, seed=1)
    lo, hi = boot.param_intervals()["m"]
    assert (hi - lo) / fit.params["m"] < 1e-3


def test_point_forecast_lies_inside_the_prediction_interval():
    y = _noisy(0.08)
    fit = fit_bass(y)
    boot = bootstrap_bass(y, fit, horizon=3, n_boot=80, seed=2)
    iv = boot.forecast_intervals()
    point = point_forecast(fit, y.size, 3)
    assert np.all(iv["pred_low"] <= point) and np.all(point <= iv["pred_high"])
    assert np.all(iv["pred_low"] >= 0)


def test_bootstrap_is_reproducible_with_a_seed():
    y = _noisy(0.08)
    fit = fit_bass(y)
    a = bootstrap_bass(y, fit, horizon=2, n_boot=25, seed=5)
    b = bootstrap_bass(y, fit, horizon=2, n_boot=25, seed=5)
    assert np.array_equal(a.params, b.params)


def test_bootstrap_needs_a_period_bass_fit():
    y = _noisy(0.05)
    with pytest.raises(ValueError):
        bootstrap_bass(y, fit_bass(y, mode="cumulative"), horizon=2, n_boot=20)
