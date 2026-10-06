"""Sentiment classifiers as scikit-learn pipelines.

Each model is one ``Pipeline``. Cross-validation fits the whole pipeline (the
TF-IDF vocabulary too) on the training folds only.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import ReviewAnalyzer

MODEL_NAMES = ("majority", "logreg", "complement_nb", "linear_svm")
OPTIONAL_MODELS = ("transformer",)


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=ReviewAnalyzer(negation=True), sublinear_tf=True, min_df=1)


def make_model(name: str, seed: int = 42, transformer_model: str | None = None) -> Pipeline:
    if name == "majority":
        return Pipeline([("tfidf", _tfidf()), ("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0, random_state=seed)
        return Pipeline([("tfidf", _tfidf()), ("clf", clf)])
    if name == "complement_nb":
        return Pipeline([("tfidf", _tfidf()), ("clf", ComplementNB(alpha=0.5))])
    if name == "linear_svm":
        return Pipeline([("tfidf", _tfidf()), ("clf", LinearSVC(class_weight="balanced", C=0.5, random_state=seed))])
    if name == "transformer":
        from .transformer import make_transformer_model

        return make_transformer_model(transformer_model or "distilbert-base-uncased", seed=seed)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES + OPTIONAL_MODELS}")
