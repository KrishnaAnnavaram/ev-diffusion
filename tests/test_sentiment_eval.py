"""Reference problems 7 (accuracy on a tiny imbalanced test split) and 8 (claims without evidence)."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from ev_diffusion.data.reviews import prepare_reviews
from ev_diffusion.report import market_note
from ev_diffusion.sentiment.aspects import aspect_mentions, aspect_table, sentence_polarity, wilson_interval
from ev_diffusion.sentiment.evaluate import corrected_interval, cross_validate
from ev_diffusion.sentiment.models import MODEL_NAMES, make_model
from ev_diffusion.sentiment.preprocess import tokenize


@pytest.fixture(scope="module")
def cv_result(reviews_frame):
    labelled = prepare_reviews(reviews_frame)
    return cross_validate(labelled.texts, labelled.labels, n_repeats=2, seed=1)


def test_macro_f1_with_majority_baseline(cv_result):
    summary = cv_result.summary.set_index("model")
    assert set(summary.index) == set(MODEL_NAMES)
    assert summary.loc["majority", "balanced_accuracy"] == pytest.approx(1 / 3)
    assert summary.loc["majority", "accuracy"] > summary.loc["majority", "macro_f1"]
    assert summary.loc["logreg", "macro_f1"] > summary.loc["majority", "macro_f1"] + 0.2
    assert summary.loc["logreg", "gain_p_value"] < 0.05
    assert {"recall_negative", "recall_neutral", "recall_positive"} <= set(summary.columns)


def test_folds_are_repeated_and_stratified(cv_result):
    assert cv_result.n_repeats == 2
    assert len(cv_result.folds) == cv_result.n_splits * 2 * len(MODEL_NAMES)


def test_split_count_adapts_to_the_smallest_class():
    texts = [f"good car {i}" for i in range(20)] + ["bad car", "bad car again", "bad bad"]
    labels = ["positive"] * 20 + ["negative"] * 3
    result = cross_validate(texts, labels, models=("majority", "logreg"), n_repeats=1)
    assert result.n_splits == 3


def test_a_class_with_one_review_is_refused():
    with pytest.raises(ValueError, match="smallest class"):
        cross_validate(["a", "b", "c", "d"], ["x", "x", "x", "y"], models=("majority",), n_repeats=1)


def test_corrected_interval_is_wider_than_the_naive_one():
    scores = np.random.default_rng(0).normal(0.7, 0.05, 50)
    mean, low, high = corrected_interval(scores, n_train=240, n_test=60)
    naive_half = 2.01 * scores.std(ddof=1) / np.sqrt(scores.size)
    assert (high - low) / 2 > 2 * naive_half
    assert low < mean < high


def test_every_model_is_one_pipeline_with_tfidf_inside():
    for name in MODEL_NAMES:
        model = make_model(name)
        assert isinstance(model, Pipeline)
        assert model.steps[0][0] == "tfidf"
    with pytest.raises(ValueError):
        make_model("lstm")


def test_sentence_polarity_flips_inside_negation():
    assert sentence_polarity(tokenize("the range is good")) == 1
    assert sentence_polarity(tokenize("the range is not good")) == -1
    assert sentence_polarity(tokenize("never had a problem")) == 1


def test_aspect_mentions_find_charging_complaints():
    found = aspect_mentions("The charging station is not good. I love the seats.")
    assert ("charging", -1) in found and ("comfort", 1) in found


def test_concern_needs_enough_mentions_and_a_clear_negative_share():
    texts = ["The charging is bad."] * 2 + ["The service is terrible."] * 12 + ["The service is great."] * 2
    table = aspect_table(texts, min_mentions=5).set_index("aspect")
    assert table.loc["service", "concern"] == "supported"
    assert table.loc["charging", "concern"] == "not supported"


def test_wilson_interval_bounds():
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0 < high < 0.35
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_market_mismatch_blocks_a_joint_claim():
    assert "does not link" in market_note("US", "IN")
    assert "descriptive, not causal" in market_note("US", "us")
    assert "unknown" in market_note("US", None)
