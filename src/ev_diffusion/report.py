"""The full study: adoption fit, backtest, bootstrap forecast, sentiment and aspects.

``run_study`` returns plain Python objects and DataFrames. ``write_outputs``
saves them as CSV, JSON and one Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest
from .config import Settings
from .data.covariates import design_matrix, load_covariates
from .data.registrations import build_series, coverage_summary, load_registrations
from .data.reviews import load_reviews, prepare_reviews
from .diffusion.baselines import default_forecasters
from .diffusion.bass import fit_bass, peak_time
from .diffusion.gbm import GBMIdentificationError, compare_nested, fit_gbm
from .diffusion.uncertainty import bootstrap_bass, point_forecast
from .sentiment.aspects import aspect_table
from .sentiment.evaluate import cross_validate
from .sentiment.models import MODEL_NAMES


def adoption_study(
    registrations_path,
    covariates_path=None,
    settings: Settings | None = None,
    horizon: int = 5,
    policy: str = "auto",
    min_train: int = 8,
    backtest_horizon: int = 3,
) -> dict:
    settings = settings or Settings()
    frame = load_registrations(registrations_path)
    series = build_series(frame, policy=policy, last_complete_year=settings.last_complete_year, market=settings.market)
    y, years = series.counts, series.years
    out: dict = {"series": series, "coverage": coverage_summary(frame)}

    bass = fit_bass(y, mode="period")
    out["bass"] = bass
    out["bass_cumulative"] = fit_bass(y, mode="cumulative")

    z = names = None
    out["gbm"] = out["gbm_comparison"] = None
    out["notes"] = list(series.notes)
    if covariates_path is not None:
        cov = load_covariates(covariates_path)
        z, names, cov_notes = design_matrix(cov, years)
        out["notes"] += cov_notes
        try:
            gbm = fit_gbm(y, z, names)
            out["gbm"] = gbm
            out["gbm_comparison"] = compare_nested(bass, gbm)
        except GBMIdentificationError as exc:
            out["notes"].append(f"GBM not fitted: {exc}")
            z = None
        future_years = list(range(int(years[-1]) + 1, int(years[-1]) + 1 + horizon))
        if out["gbm"] is not None and set(future_years) <= set(cov["year"]):
            z_future, _, _ = design_matrix(cov, future_years)
            z_all = np.vstack([z, z_future])
            out["gbm_forecast"] = out["gbm"].predict_period(len(y) + horizon, z_all=z_all)[len(y) :]
        else:
            out["gbm_forecast"] = None

    results = backtest.rolling_origin(
        y,
        default_forecasters(with_gbm=z is not None, names=names),
        years=years,
        z=z,
        min_train=min(min_train, len(y) - 2),
        horizon=backtest_horizon,
    )
    out["backtest"] = results
    out["backtest_summary"] = backtest.summarize(results)
    out["backtest_ranking"] = backtest.ranking(out["backtest_summary"])

    boot = bootstrap_bass(y, bass, horizon=horizon, n_boot=settings.n_boot, seed=settings.seed)
    intervals = boot.forecast_intervals()
    future = np.arange(int(years[-1]) + 1, int(years[-1]) + 1 + horizon)
    forecast = pd.DataFrame(
        {
            "year": future,
            "bass_point": point_forecast(bass, len(y), horizon),
            "mean_low_95": intervals["mean_low"],
            "mean_high_95": intervals["mean_high"],
            "pred_low_95": intervals["pred_low"],
            "pred_high_95": intervals["pred_high"],
        }
    )
    if out.get("gbm_forecast") is not None:
        forecast["gbm_point"] = out["gbm_forecast"]
    out["forecast"] = forecast
    out["bootstrap"] = boot
    return out


def sentiment_study(reviews_path, settings: Settings | None = None, models=MODEL_NAMES, market: str | None = None) -> dict:
    settings = settings or Settings()
    labelled = prepare_reviews(load_reviews(reviews_path), market=market)
    cv = cross_validate(
        labelled.texts,
        labelled.labels,
        models=models,
        n_repeats=settings.cv_repeats,
        seed=settings.seed,
        transformer_model=settings.transformer_model,
    )
    return {"reviews": labelled, "cv": cv, "aspects": aspect_table(labelled.texts, labelled.labels)}


def market_note(adoption_market: str | None, review_market: str | None) -> str:
    if not review_market or not adoption_market:
        return "The review market is unknown: aspect results are not linked to the adoption curve."
    if review_market.upper() != adoption_market.upper():
        return (
            f"The reviews come from market {review_market} and the registrations from market {adoption_market}: "
            "the report does not link review aspects to the adoption curve."
        )
    return "The reviews and the registrations come from the same market. The aspect table is descriptive, not causal."


def _fmt(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}g}"


def _table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) if isinstance(v, (int, float, np.number)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def render_markdown(adoption: dict | None, sentiment: dict | None) -> str:
    parts = ["# ev-diffusion report", ""]
    if adoption:
        series, bass = adoption["series"], adoption["bass"]
        parts += ["## Adoption series", "", f"Source: `{series.source}`, years {series.years[0]}-{series.years[-1]}.", ""]
        parts += [f"- {note}" for note in adoption["notes"] + bass.notes] + [""]
        parts += ["## Bass fit (period mode, HAC standard errors)", "", _table(pd.DataFrame(bass.parameter_table("hac"))), ""]
        diag = bass.diagnostics()
        parts += [
            f"Peak period: {peak_time(bass.x[0], bass.x[1]):.1f}. Lag-1 residual autocorrelation: {diag['lag1_autocorrelation']:.3f}. "
            f"Ljung-Box p-value: {diag['ljung_box_p']:.3f}.",
            "",
        ]
        if adoption.get("gbm") is not None:
            parts += ["## GBM fit", "", _table(pd.DataFrame(adoption["gbm"].parameter_table("hac"))), ""]
            cmp = adoption["gbm_comparison"]
            parts += [f"F test against Bass: F = {cmp['f_statistic']:.2f}, p = {cmp['f_p_value']:.3g}. Delta AICc = {cmp['delta_aicc']:.2f}.", ""]
        parts += ["## Rolling-origin backtest (mean MAPE over horizons)", "", _table(adoption["backtest_ranking"]), ""]
        parts += ["## Forecast with bootstrap 95 % intervals", "", _table(adoption["forecast"]), ""]
        boot = adoption["bootstrap"]
        parts += [f"Bootstrap: {boot.params.shape[0]} refits, {boot.n_failed} failed, block length {boot.block}, {boot.residual_kind} residuals.", ""]
    if sentiment:
        rev, cv = sentiment["reviews"], sentiment["cv"]
        parts += ["## Review sentiment", "", f"Input reviews: {rev.n_input}. Labelled: {len(rev.frame)}. Class counts: {rev.class_counts()}.", ""]
        parts += [f"- {note}" for note in rev.notes] + [""]
        cols = ["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "balanced_accuracy", "accuracy"]
        parts += [f"Repeated stratified CV: {cv.n_splits} folds x {cv.n_repeats} repeats.", "", _table(cv.summary[cols]), ""]
        parts += ["## Aspect sentiment", "", _table(sentiment["aspects"][["aspect", "mentions", "negative", "positive", "negative_share", "negative_share_ci_low", "concern"]]), ""]
    if adoption and sentiment:
        parts += ["## Market alignment", "", market_note(adoption["series"].market, sentiment["reviews"].market), ""]
    return "\n".join(parts)


def write_outputs(out_dir, adoption: dict | None, sentiment: dict | None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    summary: dict = {}
    if adoption:
        adoption["backtest"].to_csv(out / "backtest.csv", index=False)
        adoption["backtest_summary"].to_csv(out / "backtest_summary.csv", index=False)
        adoption["forecast"].to_csv(out / "forecast.csv", index=False)
        adoption["coverage"].to_csv(out / "coverage.csv", index=False)
        summary["series_source"] = adoption["series"].source
        summary["bass"] = adoption["bass"].params
        summary["bass_diagnostics"] = adoption["bass"].diagnostics()
        summary["bootstrap_param_ci95"] = adoption["bootstrap"].param_intervals()
        summary["backtest_ranking"] = adoption["backtest_ranking"].to_dict(orient="records")
        if adoption.get("gbm") is not None:
            summary["gbm"] = adoption["gbm"].params
            summary["gbm_comparison"] = adoption["gbm_comparison"]
        paths.update({k: out / f"{k}.csv" for k in ("backtest", "backtest_summary", "forecast", "coverage")})
    if sentiment:
        sentiment["cv"].folds.to_csv(out / "cv_folds.csv", index=False)
        sentiment["cv"].summary.to_csv(out / "cv_summary.csv", index=False)
        sentiment["aspects"].to_csv(out / "aspects.csv", index=False)
        summary["cv_summary"] = sentiment["cv"].summary.to_dict(orient="records")
        summary["class_counts"] = sentiment["reviews"].class_counts()
        paths.update({k: out / f"{k}.csv" for k in ("cv_folds", "cv_summary", "aspects")})
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(adoption, sentiment), encoding="utf-8")
    paths["json"], paths["markdown"] = out / "report.json", out / "report.md"
    return paths
