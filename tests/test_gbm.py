"""Reference problem 1: the old "GBM" only multiplied m and was not identified."""

import numpy as np
import pytest

from ev_diffusion.data import synthetic
from ev_diffusion.data.covariates import design_matrix
from ev_diffusion.diffusion.bass import bass_period, fit_bass
from ev_diffusion.diffusion.gbm import GBMIdentificationError, check_identifiable, compare_nested, fit_gbm, gbm_period
from ev_diffusion.diffusion.inference import condition_number


def _design(years):
    return design_matrix(synthetic.covariate_frame(), years)


def test_gbm_with_zero_betas_is_the_bass_curve():
    z = np.random.default_rng(1).normal(0, 0.2, size=(15, 2))
    x = np.array([0.01, 0.4, 1000.0, 0.0, 0.0])
    assert np.allclose(gbm_period(x, z), bass_period(0.01, 0.4, 1000.0, 15))


def test_gbm_without_covariates_is_refused():
    with pytest.raises(GBMIdentificationError):
        check_identifiable(np.empty((10, 0)))


def test_constant_log_change_is_refused():
    z = np.full((12, 1), 0.2)
    with pytest.raises(GBMIdentificationError, match="confounded"):
        check_identifiable(z, ["ports"])


def test_collinear_covariates_are_refused():
    base = np.random.default_rng(2).normal(0, 0.1, 12)
    with pytest.raises(GBMIdentificationError, match="collinear"):
        check_identifiable(np.column_stack([base, 2 * base]))


def test_gbm_parameters_are_jointly_identified():
    years = synthetic.YEARS[:-1]
    z, names, _ = _design(years)
    y = synthetic.true_national_series(years=years)
    fit = fit_gbm(y, z, names)
    assert condition_number(fit.jacobian) < 1e6
    assert np.linalg.matrix_rank(fit.jacobian) == fit.k


def test_gbm_recovers_true_parameters_on_clean_data():
    years = synthetic.YEARS[:-1]
    z, names, _ = _design(years)
    y = synthetic.true_national_series(years=years)
    fit = fit_gbm(y, z, names)
    true = synthetic.TRUE_PARAMS
    assert fit.params["q"] == pytest.approx(true["q"], rel=0.02)
    assert fit.params["beta_charging_ports"] == pytest.approx(true["beta_charging_ports"], rel=0.05)
    assert fit.params["beta_incentive_index"] == pytest.approx(true["beta_incentive_index"], rel=0.05)


def test_gbm_beats_bass_on_data_with_covariate_effects():
    years = synthetic.YEARS[:-1]
    z, names, _ = _design(years)
    y = synthetic.true_national_series(years=years) * np.exp(np.random.default_rng(3).normal(0, 0.02, len(years)))
    comparison = compare_nested(fit_bass(y), fit_gbm(y, z, names))
    assert comparison["f_p_value"] < 0.01
    assert comparison["delta_aicc"] < 0


def test_gbm_prediction_needs_future_covariates():
    years = synthetic.YEARS[:-1]
    z, names, _ = _design(years)
    fit = fit_gbm(synthetic.true_national_series(years=years), z, names)
    with pytest.raises(ValueError):
        fit.predict_period(len(years) + 3)
