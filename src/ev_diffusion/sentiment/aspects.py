"""Aspect sentiment: which topics the reviews praise or criticise, with counts.

Each review is cut into sentences. A sentence mentions an aspect if it holds one
of the aspect keywords. The sentence polarity comes from a small opinion lexicon,
with the polarity of a word flipped inside a negation scope. Every claim in the
report about an aspect cites the numbers of this table.

A concern is ``supported`` only when the aspect has at least ``min_mentions``
mentions and the lower bound of the Wilson 95 % interval of the negative share
is above ``min_negative_share``.
"""

from __future__ import annotations

import math
import re

import pandas as pd

from .preprocess import NEGATION_CUES, SCOPE_BREAKS, tokenize

ASPECTS: dict[str, tuple[str, ...]] = {
    "charging": ("charge", "charging", "charger", "chargers", "station", "stations", "plug", "socket"),
    "range": ("range", "km", "kms", "mileage", "distance", "battery"),
    "price": ("price", "priced", "cost", "costly", "expensive", "cheap", "money", "value", "afford", "affordable"),
    "service": ("service", "dealer", "dealers", "dealership", "support", "maintenance", "warranty"),
    "performance": ("speed", "pickup", "acceleration", "power", "torque", "performance", "drive", "handling"),
    "comfort": ("comfort", "comfortable", "seat", "seats", "space", "ride", "suspension", "cabin", "noise"),
    "design": ("design", "look", "looks", "style", "interior", "exterior", "build", "quality"),
}

POSITIVE_WORDS = frozenset(
    """good great excellent amazing awesome superb fantastic smooth fast quick comfortable happy love loved best
    impressive reliable spacious quiet premium perfect nice worth satisfied pleasant easy efficient strong
    affordable convenient""".split()
)
NEGATIVE_WORDS = frozenset(
    """bad poor worst terrible awful slow expensive costly overpriced weak noisy uncomfortable disappointed
    disappointing problem problems issue issues broken difficult hard rare scarce useless pathetic horrible
    unreliable cramped short delay delayed fails failed failure lacking""".split()
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def sentence_polarity(tokens: list[str]) -> int:
    """+1, -1 or 0 from the opinion words of one sentence, with negation flips."""
    score = 0
    in_scope = False
    for tok in tokens:
        if tok in SCOPE_BREAKS:
            in_scope = False
            continue
        if tok in NEGATION_CUES:
            in_scope = True
            continue
        value = 1 if tok in POSITIVE_WORDS else -1 if tok in NEGATIVE_WORDS else 0
        score += -value if in_scope else value
    return (score > 0) - (score < 0)


def aspect_mentions(text: str) -> list[tuple[str, int]]:
    """(aspect, polarity) for every aspect mention in every sentence of ``text``."""
    found = []
    for sentence in _SENTENCE.split(str(text)):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        polarity = sentence_polarity(tokens)
        words = set(tokens)
        for aspect, keys in ASPECTS.items():
            if words.intersection(keys):
                found.append((aspect, polarity))
    return found


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def aspect_table(texts, labels=None, min_mentions: int = 5, min_negative_share: float = 0.2) -> pd.DataFrame:
    """One row per aspect: mentions, polarity counts, negative share with a Wilson interval."""
    labels = list(labels) if labels is not None else [None] * len(texts)
    stats_by_aspect = {a: {"mentions": 0, "reviews": 0, "negative": 0, "positive": 0, "neutral": 0, "in_negative_reviews": 0} for a in ASPECTS}
    for text, label in zip(texts, labels):
        seen = set()
        for aspect, polarity in aspect_mentions(text):
            row = stats_by_aspect[aspect]
            row["mentions"] += 1
            row[{1: "positive", -1: "negative", 0: "neutral"}[polarity]] += 1
            seen.add(aspect)
        for aspect in seen:
            stats_by_aspect[aspect]["reviews"] += 1
            if label == "negative":
                stats_by_aspect[aspect]["in_negative_reviews"] += 1

    rows = []
    for aspect, row in stats_by_aspect.items():
        polar = row["negative"] + row["positive"]
        share = row["negative"] / polar if polar else 0.0
        low, high = wilson_interval(row["negative"], polar)
        supported = row["mentions"] >= min_mentions and polar > 0 and low > min_negative_share
        rows.append(
            {
                "aspect": aspect,
                **row,
                "negative_share": share,
                "negative_share_ci_low": low,
                "negative_share_ci_high": high,
                "concern": "supported" if supported else "not supported",
            }
        )
    return pd.DataFrame(rows).sort_values(["concern", "negative_share"], ascending=[False, False]).reset_index(drop=True)
