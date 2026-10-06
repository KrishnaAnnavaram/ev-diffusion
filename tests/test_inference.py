"""Reference problem 9: t-tests used the naive covariance of autocorrelated residuals."""

import numpy as np
import pytest

from ev_diffusion.diffusion.inference import (
    acf,
    durbin_watson,
    hac_covariance,
    information_criteria,
    ljung_box,
    newey_west_lags,
    numeric_jacobian,
    ols_covariance,
    parameter_table,
)


def _ar1(n, rho, rng):
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = rho * e[i - 1] + rng.normal()
    return e


def test_hac_standard_error_is_larger_under_positive_autocorrelation():
    rng = np.random.default_rng(0)
    n = 200
    x = np.cumsum(rng.normal(size=n)) / 5
    jac = np.column_stack([np.ones(n), x])
    ratios = []
    for _ in range(20):
        resid = _ar1(n, 0.8, rng)
        resid -= jac @ np.linalg.lstsq(jac, resid, rcond=None)[0]
        ratios.append(np.sqrt(hac_covariance(jac, resid)[1, 1] / ols_covariance(jac, resid)[1, 1]))
    assert np.mean(ratios) > 1.3


def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(1)
    assert ljung_box(_ar1(300, 0.7, rng))[1] < 0.001
    assert ljung_box(rng.normal(size=300))[1] > 0.01


def test_durbin_watson_and_acf_on_white_noise():
    noise = np.random.default_rng(2).normal(size=2000)
    assert durbin_watson(noise) == pytest.approx(2.0, abs=0.15)
    assert abs(acf(noise, 1)[0]) < 0.06


def test_information_criteria_penalise_parameters():
    a = information_criteria(100.0, 20, 3)
    b = information_criteria(100.0, 20, 4)
    assert b["aic"] - a["aic"] == pytest.approx(2.0)
    assert b["bic"] - a["bic"] == pytest.approx(np.log(20))
    assert b["aicc"] > b["aic"]


def test_numeric_jacobian_of_a_linear_map():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert np.allclose(numeric_jacobian(lambda v: a @ v, np.array([1.0, 2.0])), a)


def test_parameter_table_fields():
    rows = parameter_table(["a"], [2.0], np.array([[0.25]]), n=30)
    assert rows[0]["std_error"] == pytest.approx(0.5)
    assert rows[0]["ci_low"] < 2.0 < rows[0]["ci_high"]
    assert newey_west_lags(14) >= 1
