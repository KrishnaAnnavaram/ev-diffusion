"""Owner reviews: loading, sentiment labels from star ratings, and duplicate removal."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import pandas as pd

from ..schema import Column, conform, read_table

REVIEW_COLUMNS = (
    Column("text", "str", aliases=("review", "review_text", "body", "comment")),
    Column("rating", "float", aliases=("stars", "overall_rating", "score"), allow_missing=True),
    Column("model_name", "str", required=False, aliases=("model", "vehicle"), allow_missing=True),
    Column("market", "str", required=False, aliases=("country", "site_country"), allow_missing=True),
)

LABELS = ("negative", "neutral", "positive")


def rating_to_label(rating) -> str | None:
    """Map a 1-5 star rating to a sentiment label.

    ``0``, a negative value and a missing value mean "not rated" and give ``None``.
    The review is then left out of the labelled set. It is not a negative review.
    """
    if rating is None:
        return None
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value <= 0 or value > 5:
        return None
    if value < 2.5:
        return "negative"
    if value < 3.5:
        return "neutral"
    return "positive"


def dedupe_key(text: str) -> str:
    """Lower case, letters and digits only, single spaces: equal keys are duplicates."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


@dataclass
class LabelledReviews:
    """The labelled review set and the counts of what was removed and why."""

    frame: pd.DataFrame
    n_input: int
    n_unrated: int
    n_duplicates: int
    n_empty: int
    market: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return self.frame["text"].tolist()

    @property
    def labels(self) -> list[str]:
        return self.frame["label"].tolist()

    def class_counts(self) -> dict[str, int]:
        counts = self.frame["label"].value_counts()
        return {label: int(counts.get(label, 0)) for label in LABELS}


def load_reviews(path) -> pd.DataFrame:
    return conform(read_table(path), REVIEW_COLUMNS, source=str(path))


def prepare_reviews(frame: pd.DataFrame, market: str | None = None) -> LabelledReviews:
    """Drop empty and duplicate reviews, map ratings to labels and drop unrated reviews."""
    n_input = len(frame)
    frame = frame.copy()
    frame["text"] = frame["text"].astype("string").fillna("").str.strip()
    empty = frame["text"].str.len() == 0
    frame = frame[~empty]

    keys = frame["text"].map(dedupe_key)
    duplicated = keys.duplicated(keep="first")
    frame = frame[~duplicated]

    frame["label"] = frame["rating"].map(rating_to_label)
    unrated = frame["label"].isna()
    frame = frame[~unrated].reset_index(drop=True)

    notes = []
    if market is None and "market" in frame.columns:
        markets = sorted(frame["market"].dropna().unique().tolist())
        market = markets[0] if len(markets) == 1 else (",".join(markets) or None)
    if int(unrated.sum()):
        notes.append(f"{int(unrated.sum())} review(s) without a valid 1-5 rating left out (rating 0 = not rated)")
    if int(duplicated.sum()):
        notes.append(f"{int(duplicated.sum())} duplicate review(s) removed")
    return LabelledReviews(
        frame=frame,
        n_input=n_input,
        n_unrated=int(unrated.sum()),
        n_duplicates=int(duplicated.sum()),
        n_empty=int(empty.sum()),
        market=market,
        notes=notes,
    )
