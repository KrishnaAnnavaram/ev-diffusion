import numpy as np
import pytest

from ev_diffusion.diffusion.bass import as_counts, bass_cdf, bass_pdf, bass_period, fit_bass, peak_time


def test_cdf_starts_at_zero_and_tends_to_one():
    assert bass_cdf(0.0, 0.01, 0.4) == pytest.approx(0.0)
    assert bass_cdf(500.0, 0.01, 0.4) == pytest.approx(1.0)
    values = bass_cdf(np.arange(50), 0.01, 0.4)
    assert np.all(np.diff(values) > 0)


def test_pdf_is_the_derivative_of_cdf():
    t, h = 7.3, 1e-5
    numeric = (bass_cdf(t + h, 0.02, 0.3) - bass_cdf(t - h, 0.02, 0.3)) / (2 * h)
    assert bass_pdf(t, 0.02, 0.3) == pytest.approx(numeric, rel=1e-6)


def test_period_adoptions_sum_to_the_cumulative_share():
    periods = bass_period(0.01, 0.4, 1000.0, 12)
    assert periods.sum() == pytest.approx(1000.0 * bass_cdf(12, 0.01, 0.4))


def test_peak_time_is_the_maximum_of_the_pdf():
    t = np.linspace(0, 40, 40001)
    assert t[np.argmax(bass_pdf(t, 0.01, 0.4))] == pytest.approx(peak_time(0.01, 0.4), abs=1e-3)
    assert peak_time(0.5, 0.1) == 0.0


@pytest.mark.parametrize("mode", ["period", "cumulative"])
def test_fit_recovers_parameters_of_a_clean_series(clean_bass, mode):
    y, (p, q, m) = clean_bass
    fit = fit_bass(y, mode=mode)
    assert fit.params["p"] == pytest.approx(p, rel=1e-3)
    assert fit.params["q"] == pytest.approx(q, rel=1e-3)
    assert fit.params["m"] == pytest.approx(m * bass_cdf(1e6, p, q), rel=1e-3)
    assert fit.sse < 1e-3 * float(y @ y)


def test_market_size_is_never_below_the_observed_total(rng):
    y = bass_period(0.003, 0.5, 50_000.0, 10) * np.exp(rng.normal(0, 0.05, 10))
    fit = fit_bass(y)
    assert fit.params["m"] >= y.sum() * 0.999


def test_series_before_its_peak_gets_an_identification_note():
    y = bass_period(0.001, 0.3, 1e6, 8)
    fit = fit_bass(y)
    assert any("weakly identified" in note for note in fit.notes)


def test_standard_errors_are_finite_for_every_parameter(clean_bass, rng):
    y, _ = clean_bass
    fit = fit_bass(y * np.exp(rng.normal(0, 0.05, y.size)))
    for kind in ("ols", "hac"):
        table = fit.parameter_table(kind)
        assert all(np.isfinite(r["std_error"]) and r["std_error"] > 0 for r in table)


@pytest.mark.parametrize("bad", [[1, 2, 3], [1, -2, 3, 4], [0, 0, 0, 0], [1, np.nan, 2, 3]])
def test_invalid_counts_are_rejected(bad):
    with pytest.raises(ValueError):
        as_counts(bad)
