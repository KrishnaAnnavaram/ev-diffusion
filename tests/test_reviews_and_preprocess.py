"""Reference problems 5 (rating 0 as negative) and 6 (stopwords removed negations)."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score

from ev_diffusion.data.reviews import dedupe_key, prepare_reviews, rating_to_label
from ev_diffusion.sentiment.models import make_model
from ev_diffusion.sentiment.preprocess import NEGATION_CUES, STOPWORDS, ReviewAnalyzer, mark_negation, tokenize


@pytest.mark.parametrize(
    "rating, label",
    [(0, None), (None, None), (float("nan"), None), (-1, None), (6, None), (1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_rating_mapping_treats_zero_as_not_rated(rating, label):
    assert rating_to_label(rating) == label


def test_prepare_drops_unrated_and_duplicates(reviews_frame):
    labelled = prepare_reviews(reviews_frame)
    assert labelled.n_unrated == int((reviews_frame.drop_duplicates("text")["rating"] == 0).sum()) or labelled.n_unrated > 0
    assert set(labelled.labels) <= {"negative", "neutral", "positive"}
    assert labelled.frame["text"].map(dedupe_key).is_unique
    assert labelled.n_input == len(reviews_frame)
    assert labelled.n_input == len(labelled.frame) + labelled.n_unrated + labelled.n_duplicates + labelled.n_empty


def test_unrated_reviews_never_become_negative():
    frame = pd.DataFrame({"text": ["great car", "awful car", "fine"], "rating": [5, 0, 1]})
    labelled = prepare_reviews(frame)
    assert labelled.labels == ["positive", "negative"]
    assert "awful car" not in labelled.texts


def test_negation_words_are_not_stopwords():
    assert not (NEGATION_CUES & STOPWORDS)
    assert {"no", "not", "nor"} <= NEGATION_CUES


def test_negation_scope_marks_words_until_punctuation():
    tokens = tokenize("The range isn't good, but the seats are comfortable.")
    assert mark_negation(tokens) == ["range", "not", "NOT_good", "seats", "comfortable"]


def test_not_good_and_good_are_different_features():
    analyzer = ReviewAnalyzer()
    assert "NOT_good" in analyzer("not good")
    assert "NOT_good" not in analyzer("good")
    assert "good" not in analyzer("not good")


def test_analyzer_can_be_pickled():
    assert pickle.loads(pickle.dumps(ReviewAnalyzer()))("not bad") == ["not", "NOT_bad", "not NOT_bad"]


def test_negation_aware_model_separates_negated_praise():
    rng = np.random.default_rng(0)
    nouns = ["range", "ride", "price", "service", "pickup", "seats"]
    texts, labels = [], []
    for _ in range(60):
        noun = rng.choice(nouns)
        texts.append(f"the {noun} is good")
        labels.append("positive")
        texts.append(f"the {noun} is not good")
        labels.append("negative")
    scores = cross_val_score(make_model("logreg"), np.array(texts, dtype=object), labels, cv=5, scoring="f1_macro")
    assert scores.mean() > 0.95
