"""Text preparation that keeps negations.

A generic English stopword list removes "no", "not" and "nor". Then "not good"
becomes "good". Here the negation words are never stopwords. The words after a
negation, up to the next punctuation mark or "but", get the prefix ``NOT_``.
So "not good" gives the feature ``NOT_good``, and it is a different feature from ``good``.
"""

from __future__ import annotations

import re

NEGATION_CUES = frozenset(
    {"no", "not", "nor", "never", "none", "nothing", "neither", "nobody", "nowhere", "without", "hardly", "barely", "cannot"}
)

_BASE_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below between
    both by could did do does doing down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself me more most my myself of off on once only or other
    our ours ourselves out over own same she so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves also just car vehicle
    """.split()
)

STOPWORDS = frozenset(_BASE_STOPWORDS - NEGATION_CUES)
SCOPE_BREAKS = frozenset({".", "!", "?", ",", ";", ":", "but", "however", "although"})

_CONTRACTIONS = (
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"n['’]t\b"), " not"),
)
_TOKEN = re.compile(r"[a-z]+|[.!?,;:]")


def tokenize(text: str) -> list[str]:
    """Lower-case word and punctuation tokens, with "n't" changed to "not"."""
    text = str(text).lower()
    for pattern, repl in _CONTRACTIONS:
        text = pattern.sub(repl, text)
    return _TOKEN.findall(text)


def mark_negation(tokens: list[str], remove_stopwords: bool = True) -> list[str]:
    """Prefix ``NOT_`` to the words in a negation scope. Drop punctuation and stopwords."""
    out: list[str] = []
    in_scope = False
    for tok in tokens:
        if tok in SCOPE_BREAKS:
            in_scope = False
            continue
        if tok in NEGATION_CUES:
            in_scope = True
            out.append(tok)
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(f"NOT_{tok}" if in_scope else tok)
    return out


class ReviewAnalyzer:
    """A picklable analyzer for ``TfidfVectorizer(analyzer=...)``: unigrams and bigrams."""

    def __init__(self, negation: bool = True, ngram_max: int = 2):
        self.negation = negation
        self.ngram_max = ngram_max

    def __call__(self, text: str) -> list[str]:
        tokens = tokenize(text)
        words = mark_negation(tokens) if self.negation else [t for t in tokens if t.isalpha() and t not in STOPWORDS]
        features = list(words)
        for size in range(2, self.ngram_max + 1):
            features += [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
        return features
