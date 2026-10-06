"""Optional transformer model: frozen encoder features plus a logistic regression.

This module imports ``torch`` and ``transformers`` only when you build the model.
Install them with ``pip install 'ev-diffusion[transformer]'``. The encoder is not
fine-tuned: on a few hundred reviews a frozen encoder with a linear head is the
safer choice. The first run downloads the model weights from the Hugging Face hub.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_CACHE: dict[str, tuple[object, object]] = {}


def _require():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError("the transformer model needs: pip install 'ev-diffusion[transformer]'") from exc
    return torch, transformers


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    """Mean-pooled last hidden states of a Hugging Face encoder."""

    def __init__(self, model_name: str = "distilbert-base-uncased", batch_size: int = 16, max_length: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

    def _load(self):
        torch, transformers = _require()
        if self.model_name not in _CACHE:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            model = transformers.AutoModel.from_pretrained(self.model_name)
            model.eval()
            _CACHE[self.model_name] = (tokenizer, model)
        return torch, _CACHE[self.model_name]

    def fit(self, X, y=None):
        self._load()
        return self

    def transform(self, X):
        torch, (tokenizer, model) = self._load()
        texts = [str(t) for t in X]
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                chunks.append(pooled.cpu().numpy())
        return np.vstack(chunks)


def make_transformer_model(model_name: str, seed: int = 42) -> Pipeline:
    _require()
    clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
    return Pipeline([("embed", TransformerEmbedder(model_name=model_name)), ("clf", clf)])
