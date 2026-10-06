import numpy as np
import pytest

from ev_diffusion.data import synthetic
from ev_diffusion.data.registrations import normalize_registrations
from ev_diffusion.diffusion.bass import bass_period


@pytest.fixture(scope="session")
def registrations():
    return normalize_registrations(synthetic.registration_frame(seed=7))


@pytest.fixture(scope="session")
def reviews_frame():
    return synthetic.review_frame(n=300, seed=7)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    synthetic.write_synthetic(out, seed=7)
    return out


@pytest.fixture
def clean_bass():
    """A noiseless Bass series that passes its peak: p, q and m are identified."""
    return bass_period(0.01, 0.4, 100_000.0, 30), (0.01, 0.4, 100_000.0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
