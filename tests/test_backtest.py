"""Reference problem 2: no out-of-sample validation and no baselines."""

import numpy as np
import pytest

from ev_diffusion import backtest
from ev_diffusion.diffusion.baselines import (
    DriftForecaster,
    ExponentialForecaster,
    Forecaster,
    LogisticForecaster,
    NaiveForecaster,
    default_forecasters,
)
from ev_diffusion.diffusion.bass import bass_period


class Spy(Forecaster):
    seen: list[int] = []

    def fit(self, y, z=None):
        Spy.seen.append(len(y))
        self.last_ = y[-1]
        return self

    def predict(self, h, z_future=None):
        return np.full(h, self.last_)


class Broken(Forecaster):
    def fit(self, y, z=None):
        raise RuntimeError("no fit")


def test_models_only_see_data_before_the_origin():
    Spy.seen = []
    y = np.arange(1.0, 13.0)
    backtest.rolling_origin(y, {"spy": Spy}, min_train=8, horizon=2)
    assert Spy.seen == [8, 9, 10, 11]


def test_forecasts_do_not_change_when_future_values_change():
    y = bass_period(0.01, 0.4, 1e5, 14)
    changed = y.copy()
    changed[10:] *= 3
    a = backtest.rolling_origin(y, default_forecasters(), min_train=10, horizon=3)
    b = backtest.rolling_origin(changed, default_forecasters(), min_train=10, horizon=3)
    first = (a["origin_year"] == a["origin_year"].min())
    assert np.allclose(a.loc[first, "forecast"], b.loc[first, "forecast"])


def test_summary_has_metrics_per_model_and_horizon():
    y = bass_period(0.01, 0.4, 1e5, 14)
    results = backtest.rolling_origin(y, default_forecasters(), min_train=8, horizon=3)
    summary = backtest.summarize(results)
    assert set(summary["model"]) == {"naive", "drift", "exponential", "logistic", "bass", "bass_cumulative"}
    assert set(summary["horizon"]) == {1, 2, 3}
    assert {"mae", "rmse", "mape_pct", "smape_pct"} <= set(summary.columns)
    ranking = backtest.ranking(summary)
    assert ranking.iloc[0]["model"] in {"bass", "bass_cumulative"}


def test_failed_fit_is_recorded_and_does_not_stop_the_backtest():
    results = backtest.rolling_origin(np.arange(1.0, 11.0), {"broken": Broken, "naive": NaiveForecaster}, min_train=8)
    assert (results[results["model"] == "broken"]["status"] != "ok").all()
    assert (results[results["model"] == "naive"]["status"] == "ok").all()
    assert backtest.summarize(results).set_index("model").loc["broken", "failed"].sum() > 0


def test_min_train_is_checked():
    with pytest.raises(ValueError):
        backtest.rolling_origin(np.arange(1.0, 6.0), {"naive": NaiveForecaster}, min_train=5)


def test_baselines_give_expected_shapes_and_values():
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    assert NaiveForecaster().fit(y).predict(2).tolist() == [32.0, 32.0]
    assert DriftForecaster().fit(y).predict(1)[0] == pytest.approx(32 + 31 / 5)
    assert ExponentialForecaster().fit(y).predict(1)[0] == pytest.approx(64.0, rel=1e-6)
    logistic = LogisticForecaster().fit(np.diff(np.concatenate([[0], 1000 / (1 + np.exp(-0.6 * (np.arange(1, 13) - 8))) ])))
    assert logistic.predict(3).shape == (3,)
