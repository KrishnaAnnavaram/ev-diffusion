"""Command line interface: ``ev-diffusion <command>``."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

from . import __version__, backtest
from .config import Settings
from .data.covariates import design_matrix, load_covariates
from .data.registrations import build_series, coverage_summary, load_registrations
from .data.reviews import load_reviews, prepare_reviews
from .data.synthetic import write_synthetic
from .diffusion.baselines import default_forecasters
from .diffusion.bass import fit_bass, peak_time
from .diffusion.gbm import compare_nested, fit_gbm
from .report import adoption_study, render_markdown, sentiment_study, write_outputs
from .sentiment.aspects import aspect_table
from .sentiment.models import MODEL_NAMES, OPTIONAL_MODELS


def _settings(args) -> Settings:
    settings = Settings.from_env()
    changes = {}
    for field in ("seed", "n_boot", "cv_repeats", "last_complete_year", "market"):
        value = getattr(args, field, None)
        if value is not None:
            changes[field] = value
    return dataclasses.replace(settings, **changes)


def _series(args, settings: Settings):
    frame = load_registrations(args.registrations)
    return frame, build_series(frame, policy=args.policy, last_complete_year=settings.last_complete_year, market=settings.market)


def _print_frame(frame: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(frame.to_string(index=False))


def cmd_synth(args) -> int:
    paths = write_synthetic(args.out, seed=args.seed or 42)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def cmd_coverage(args) -> int:
    settings = _settings(args)
    frame, series = _series(args, settings)
    _print_frame(coverage_summary(frame))
    print()
    _print_frame(series.to_frame())
    for note in series.notes:
        print(f"note: {note}")
    return 0


def cmd_fit(args) -> int:
    settings = _settings(args)
    _, series = _series(args, settings)
    bass = fit_bass(series.counts, mode=args.mode)
    print(f"Bass fit ({args.mode} mode), {series.years[0]}-{series.years[-1]}, {args.se.upper()} standard errors")
    _print_frame(pd.DataFrame(bass.parameter_table(args.se)))
    print(f"peak period: {peak_time(bass.x[0], bass.x[1]):.2f}   criteria: {bass.criteria()}")
    print(f"diagnostics: {bass.diagnostics()}")
    for note in series.notes + bass.notes:
        print(f"note: {note}")
    if args.covariates:
        z, names, notes = design_matrix(load_covariates(args.covariates), series.years)
        gbm = fit_gbm(series.counts, z, names, mode=args.mode)
        print("\nGBM fit")
        _print_frame(pd.DataFrame(gbm.parameter_table(args.se)))
        print(f"comparison with Bass: {compare_nested(bass, gbm)}")
        for note in notes + gbm.notes:
            print(f"note: {note}")
    return 0


def cmd_backtest(args) -> int:
    settings = _settings(args)
    _, series = _series(args, settings)
    z = names = None
    if args.covariates:
        z, names, _ = design_matrix(load_covariates(args.covariates), series.years)
    results = backtest.rolling_origin(
        series.counts,
        default_forecasters(with_gbm=z is not None, names=names),
        years=series.years,
        z=z,
        min_train=args.min_train,
        horizon=args.horizon,
    )
    summary = backtest.summarize(results)
    _print_frame(summary)
    print()
    _print_frame(backtest.ranking(summary))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    return 0


def cmd_forecast(args) -> int:
    settings = _settings(args)
    study = adoption_study(args.registrations, args.covariates, settings=settings, horizon=args.horizon, policy=args.policy)
    _print_frame(study["forecast"])
    boot = study["bootstrap"]
    print(f"bootstrap: {boot.params.shape[0]} refits, {boot.n_failed} failed; parameter 95 % intervals {boot.param_intervals()}")
    return 0


def cmd_sentiment(args) -> int:
    settings = _settings(args)
    study = sentiment_study(args.reviews, settings=settings, models=tuple(args.models), market=args.review_market)
    rev = study["reviews"]
    print(f"reviews: {rev.n_input} in, {len(rev.frame)} labelled, classes {rev.class_counts()}")
    for note in rev.notes:
        print(f"note: {note}")
    cols = ["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "balanced_accuracy", "accuracy"]
    extra = [c for c in study["cv"].summary.columns if c.startswith("recall_") or c.startswith("gain")]
    _print_frame(study["cv"].summary[cols + extra].round(3))
    return 0


def cmd_aspects(args) -> int:
    labelled = prepare_reviews(load_reviews(args.reviews), market=args.review_market)
    _print_frame(aspect_table(labelled.texts, labelled.labels, min_mentions=args.min_mentions).round(3))
    return 0


def cmd_report(args) -> int:
    settings = _settings(args)
    data_dir = Path(args.data_dir or settings.data_dir)
    out_dir = Path(args.out or settings.output_dir)
    registrations = data_dir / "registrations.csv"
    covariates = data_dir / "covariates.csv"
    reviews = data_dir / "reviews.csv"
    adoption = None
    if registrations.exists():
        adoption = adoption_study(registrations, covariates if covariates.exists() else None, settings=settings, horizon=args.horizon)
    sentiment = sentiment_study(reviews, settings=settings, market=args.review_market) if reviews.exists() else None
    if adoption is None and sentiment is None:
        print(f"no registrations.csv or reviews.csv in {data_dir}", file=sys.stderr)
        return 2
    paths = write_outputs(out_dir, adoption, sentiment)
    print(f"wrote {paths['markdown']} and {len(paths) - 1} other file(s) in {out_dir}")
    return 0


def cmd_demo(args) -> int:
    settings = _settings(args)
    out_dir = Path(args.out)
    data_dir = out_dir / "synthetic_data"
    paths = write_synthetic(data_dir, seed=settings.seed)
    adoption = adoption_study(paths["registrations"], paths["covariates"], settings=settings, horizon=args.horizon)
    sentiment = sentiment_study(paths["reviews"], settings=settings)
    written = write_outputs(out_dir, adoption, sentiment)
    print(render_markdown(adoption, sentiment))
    print(f"\nSYNTHETIC DATA. Wrote {written['markdown']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ev-diffusion", description=__doc__)
    parser.add_argument("--version", action="version", version=f"ev-diffusion {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, registrations=False, reviews=False):
        p.add_argument("--seed", type=int)
        if registrations:
            p.add_argument("--registrations", required=True, help="CSV or XLSX with year, count and optional state")
            p.add_argument("--policy", choices=("auto", "national", "balanced"), default="auto")
            p.add_argument("--last-complete-year", dest="last_complete_year", type=int)
            p.add_argument("--market")
        if reviews:
            p.add_argument("--reviews", required=True, help="CSV with text and rating columns")
            p.add_argument("--review-market", dest="review_market")

    p = sub.add_parser("synth", help="write synthetic registrations, covariates and reviews")
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("coverage", help="coverage audit and the chosen yearly series")
    common(p, registrations=True)
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("fit", help="fit the Bass model (and the GBM with --covariates)")
    common(p, registrations=True)
    p.add_argument("--covariates")
    p.add_argument("--mode", choices=("period", "cumulative"), default="period")
    p.add_argument("--se", choices=("hac", "ols"), default="hac")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("backtest", help="rolling-origin backtest against baselines")
    common(p, registrations=True)
    p.add_argument("--covariates")
    p.add_argument("--min-train", dest="min_train", type=int, default=8)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--out")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("forecast", help="Bass forecast with bootstrap intervals")
    common(p, registrations=True)
    p.add_argument("--covariates")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n-boot", dest="n_boot", type=int)
    p.set_defaults(func=cmd_forecast)

    p = sub.add_parser("sentiment", help="repeated stratified CV of the sentiment models")
    common(p, reviews=True)
    p.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES + OPTIONAL_MODELS))
    p.add_argument("--cv-repeats", dest="cv_repeats", type=int)
    p.set_defaults(func=cmd_sentiment)

    p = sub.add_parser("aspects", help="aspect sentiment table")
    common(p, reviews=True)
    p.add_argument("--min-mentions", dest="min_mentions", type=int, default=5)
    p.set_defaults(func=cmd_aspects)

    p = sub.add_parser("report", help="full study from a data folder")
    p.add_argument("--data-dir", dest="data_dir")
    p.add_argument("--out")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int)
    p.add_argument("--n-boot", dest="n_boot", type=int)
    p.add_argument("--cv-repeats", dest="cv_repeats", type=int)
    p.add_argument("--last-complete-year", dest="last_complete_year", type=int)
    p.add_argument("--market")
    p.add_argument("--review-market", dest="review_market")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("demo", help="synthetic data + full study, no download and no network")
    p.add_argument("--out", default="reports/demo")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int)
    p.add_argument("--n-boot", dest="n_boot", type=int)
    p.add_argument("--cv-repeats", dest="cv_repeats", type=int)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
