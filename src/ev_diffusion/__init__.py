"""ev-diffusion: Bass diffusion forecasts of EV adoption plus evidence-backed review sentiment.

The core package needs only numpy, pandas, scipy and scikit-learn. The optional
transformer model lives in :mod:`ev_diffusion.sentiment.transformer` and imports
torch and transformers only when you use it.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
