"""Synthetic data for the offline demo and the tests. All values are generated.

* ``registrations.csv``: state x year x model rows plus national rows. The states
  enter the data in different years, and the final year is partial. These are
  the two coverage traps of real registration files.
* ``covariates.csv``: two positive yearly covariates with an uneven growth.
* ``reviews.csv``: imbalanced owner reviews with negations, duplicates and
  unrated (rating 0) reviews.

The national series comes from a GBM with known parameters (``TRUE_PARAMS``), so
the tests can check that the fit gets them back.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..diffusion.gbm import gbm_period

YEARS = list(range(2010, 2025))  # 2024 is partial
STATES = [f"S{i:02d}" for i in range(1, 13)]
ENTRY_YEAR = {**{s: 2010 for s in STATES[:5]}, **{s: 2013 for s in STATES[5:8]}, **{s: 2016 for s in STATES[8:10]}, **{s: 2019 for s in STATES[10:]}}
MODELS = ["Model A", "Model B", "Model C", "Model D"]
TRUE_PARAMS = {"p": 0.001, "q": 0.33, "m": 3_000_000.0, "beta_charging_ports": 0.9, "beta_incentive_index": 0.6}
PARTIAL_SHARE = 0.45
LISTED_STATE_SHARE = 0.8


def covariate_frame() -> pd.DataFrame:
    years = list(range(2009, 2031))
    growth = [0.0, 0.62, 0.48, 0.55, 0.30, 0.41, 0.22, 0.35, 0.18, 0.31, 0.16, 0.27, 0.12, 0.24, 0.20, 0.10, 0.18, 0.15, 0.12, 0.14, 0.10, 0.09]
    ports = 400 * np.exp(np.cumsum(growth))
    incentive = []
    for year in years:
        if year <= 2012:
            incentive.append(1.0)
        elif year <= 2016:
            incentive.append(1.5)
        elif year <= 2019:
            incentive.append(1.2)
        elif year <= 2022:
            incentive.append(1.8)
        else:
            incentive.append(2.2)
    return pd.DataFrame({"year": years, "charging_ports": np.round(ports, 1), "incentive_index": incentive})


def true_national_series(covariates: pd.DataFrame | None = None, years=YEARS) -> np.ndarray:
    covariates = covariate_frame() if covariates is None else covariates
    table = covariates.set_index("year")
    names = ["charging_ports", "incentive_index"]
    z = np.vstack([np.log(table.loc[y, names].to_numpy(float) / table.loc[y - 1, names].to_numpy(float)) for y in years])
    x = np.array([TRUE_PARAMS["p"], TRUE_PARAMS["q"], TRUE_PARAMS["m"], TRUE_PARAMS["beta_charging_ports"], TRUE_PARAMS["beta_incentive_index"]])
    return gbm_period(x, z)


def registration_frame(seed: int = 42, noise_sd: float = 0.06, rho: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    expected = true_national_series()
    shock = np.zeros(len(YEARS))
    for i in range(len(YEARS)):
        shock[i] = (rho * shock[i - 1] if i else 0.0) + rng.normal(0, noise_sd)
    national = expected * np.exp(shock)
    national[-1] *= PARTIAL_SHARE

    shares = rng.dirichlet(np.full(len(STATES), 2.0)) * LISTED_STATE_SHARE
    rows = []
    for i, year in enumerate(YEARS):
        rows.append({"state": "US", "year": year, "make_model": "ALL", "count": int(round(national[i]))})
        mix = rng.dirichlet(np.full(len(MODELS), 3.0))
        for state, share in zip(STATES, shares):
            if year < ENTRY_YEAR[state]:
                continue
            state_total = national[i] * share * np.exp(rng.normal(0, 0.03))
            for model, part in zip(MODELS, mix):
                rows.append({"state": state, "year": year, "make_model": model, "count": int(round(state_total * part))})
    return pd.DataFrame(rows)


_ASPECT_TERMS = {
    "charging": ["charging network", "home charger", "charging station"],
    "service": ["dealer service", "after sales service", "service centre"],
    "price": ["price", "running cost", "value for money"],
    "range": ["range", "battery range", "highway range"],
    "performance": ["pickup", "acceleration", "power delivery"],
    "comfort": ["ride", "seats", "cabin space"],
    "design": ["interior", "build quality", "exterior design"],
}
_POS = ["good", "great", "excellent", "smooth", "impressive", "superb"]
_NEG = ["poor", "terrible", "disappointing", "bad", "slow", "awful"]
_NEG_ASPECTS = (["charging", "service", "price", "range", "comfort"], [0.35, 0.25, 0.2, 0.12, 0.08])
_POS_ASPECTS = (["performance", "comfort", "design", "range", "price", "charging"], [0.3, 0.2, 0.2, 0.15, 0.1, 0.05])


def _sentence(rng: np.random.Generator, aspect: str, positive: bool) -> str:
    term = rng.choice(_ASPECT_TERMS[aspect])
    if positive:
        form = rng.integers(0, 3)
        adj = rng.choice(_POS)
        if form == 0:
            return f"The {term} is {adj}."
        if form == 1:
            return f"I love the {term}, it is {adj}."
        return f"Never had a problem with the {term}."
    form = rng.integers(0, 3)
    if form == 0:
        return f"The {term} is {rng.choice(_NEG)}."
    if form == 1:
        return f"The {term} is not {rng.choice(_POS)}."
    return f"Honestly the {term} is not {rng.choice(_POS)} at all."


def _review_text(rng: np.random.Generator, label: str) -> str:
    if label == "positive":
        aspects = rng.choice(_POS_ASPECTS[0], size=rng.integers(1, 4), p=_POS_ASPECTS[1])
        return " ".join(_sentence(rng, a, True) for a in aspects)
    if label == "negative":
        aspects = rng.choice(_NEG_ASPECTS[0], size=rng.integers(1, 4), p=_NEG_ASPECTS[1])
        return " ".join(_sentence(rng, a, False) for a in aspects)
    good = rng.choice(_POS_ASPECTS[0], p=_POS_ASPECTS[1])
    bad = rng.choice(_NEG_ASPECTS[0], p=_NEG_ASPECTS[1])
    return f"{_sentence(rng, good, True)} But {_sentence(rng, bad, False).lower()}"


def review_frame(n: int = 420, seed: int = 42, n_duplicates: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    kinds = rng.choice(["positive", "neutral", "negative", "unrated"], size=n, p=[0.70, 0.09, 0.14, 0.07])
    rows = []
    for kind in kinds:
        if kind == "unrated":
            text = _review_text(rng, rng.choice(["positive", "negative"]))
            rating = 0
        else:
            text = _review_text(rng, kind)
            rating = {"positive": rng.choice([4, 5]), "neutral": 3, "negative": rng.choice([1, 2])}[kind]
        rows.append({"text": text, "rating": int(rating), "model_name": str(rng.choice(MODELS)), "market": "US"})
    for idx in rng.choice(len(rows), size=n_duplicates, replace=False):
        copy = dict(rows[idx])
        copy["text"] = "  " + copy["text"].upper() + " "
        rows.append(copy)
    return pd.DataFrame(rows)


def write_synthetic(out_dir, seed: int = 42) -> dict[str, Path]:
    """Write the three synthetic files and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "registrations": out / "registrations.csv",
        "covariates": out / "covariates.csv",
        "reviews": out / "reviews.csv",
    }
    registration_frame(seed=seed).to_csv(paths["registrations"], index=False)
    covariate_frame().to_csv(paths["covariates"], index=False)
    review_frame(seed=seed).to_csv(paths["reviews"], index=False)
    return paths
