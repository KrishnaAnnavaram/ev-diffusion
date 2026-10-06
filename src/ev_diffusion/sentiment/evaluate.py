"""Repeated stratified cross-validation with macro-F1 as the main metric.

Accuracy hides the minority classes when most reviews are positive. The main
metric is macro-F1, with balanced accuracy and the recall of each class. Every
model is compared with the majority-class baseline on the same folds.

The 95 % intervals use the Nadeau-Bengio correction for repeated
cross-validation, because the folds share training data and are not independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .models import MODEL_NAMES, make_model

METRICS = ("macro_f1", "balanced_accuracy", "accuracy")


@dataclass
class CVResult:
    folds: pd.DataFrame
    summary: pd.DataFrame
    n_splits: int
    n_repeats: int
    class_counts: dict[str, int]


def corrected_interval(scores: np.ndarray, n_train: int, n_test: int, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and the Nadeau-Bengio corrected t interval of fold scores."""
    scores = np.asarray(scores, float)
    k = scores.size
    mean = float(scores.mean())
    if k < 2:
        return mean, mean, mean
    var = float(scores.var(ddof=1)) * (1.0 / k + n_test / n_train)
    half = float(stats.t.ppf(0.5 + level / 2, k - 1)) * math.sqrt(var)
    return mean, mean - half, mean + half


def corrected_paired_test(diff: np.ndarray, n_train: int, n_test: int) -> float:
    """Two-sided p-value of the corrected resampled t test on paired fold differences."""
    diff = np.asarray(diff, float)
    k = diff.size
    var = float(diff.var(ddof=1)) * (1.0 / k + n_test / n_train)
    if var <= 0:
        return 0.0 if diff.mean() != 0 else 1.0
    t_val = float(diff.mean()) / math.sqrt(var)
    return float(2 * stats.t.sf(abs(t_val), k - 1))


def cross_validate(
    texts,
    labels,
    models=MODEL_NAMES,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    transformer_model: str | None = None,
) -> CVResult:
    texts = np.asarray(list(texts), dtype=object)
    labels = np.asarray(list(labels), dtype=object)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        raise ValueError("cross-validation needs at least two classes")
    splits = int(min(n_splits, counts.min()))
    if splits < 2:
        raise ValueError(f"the smallest class has {counts.min()} review(s): at least 2 are needed")
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=n_repeats, random_state=seed)
    prototypes = {name: make_model(name, seed=seed, transformer_model=transformer_model) for name in models}

    rows = []
    for fold_id, (train, test) in enumerate(cv.split(texts, labels)):
        for name, proto in prototypes.items():
            model = clone(proto)
            model.fit(texts[train], labels[train])
            pred = model.predict(texts[test])
            truth = labels[test]
            row = {
                "model": name,
                "repeat": fold_id // splits,
                "fold": fold_id % splits,
                "n_train": len(train),
                "n_test": len(test),
                "macro_f1": f1_score(truth, pred, average="macro", labels=classes, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(truth, pred),
                "accuracy": accuracy_score(truth, pred),
            }
            recalls = recall_score(truth, pred, labels=classes, average=None, zero_division=0)
            for cls, rec in zip(classes, recalls):
                row[f"recall_{cls}"] = rec
            rows.append(row)
    folds = pd.DataFrame(rows)
    return CVResult(
        folds=folds,
        summary=summarize(folds, baseline="majority" if "majority" in models else None),
        n_splits=splits,
        n_repeats=n_repeats,
        class_counts={str(c): int(n) for c, n in zip(classes, counts)},
    )


def summarize(folds: pd.DataFrame, baseline: str | None = "majority") -> pd.DataFrame:
    n_train = int(folds["n_train"].mean())
    n_test = int(folds["n_test"].mean())
    recall_cols = [c for c in folds.columns if c.startswith("recall_")]
    rows = []
    for name, group in folds.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": name}
        for metric in METRICS:
            mean, low, high = corrected_interval(group[metric].to_numpy(), n_train, n_test)
            row[metric] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        for col in recall_cols:
            row[col] = float(group[col].mean())
        if baseline and name != baseline:
            base = folds[folds["model"] == baseline]["macro_f1"].to_numpy()
            diff = group["macro_f1"].to_numpy() - base
            row["macro_f1_gain_vs_majority"] = float(diff.mean())
            row["gain_p_value"] = corrected_paired_test(diff, n_train, n_test)
        rows.append(row)
    return pd.DataFrame(rows)
