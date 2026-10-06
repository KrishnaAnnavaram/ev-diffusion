"""Settings read from environment variables, with safe defaults for the offline demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name, "").strip()
    return raw or default


def _env_int(name: str, default: int | None, minimum: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """All run-time settings in one place. CLI flags override these values."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("reports")
    seed: int = 42
    n_boot: int = 200
    cv_repeats: int = 10
    last_complete_year: int | None = None
    market: str = "US"
    transformer_model: str = "distilbert-base-uncased"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(_env_str("EV_DIFFUSION_DATA_DIR", "data")),
            output_dir=Path(_env_str("EV_DIFFUSION_OUTPUT_DIR", "reports")),
            seed=_env_int("EV_DIFFUSION_SEED", 42),
            n_boot=_env_int("EV_DIFFUSION_N_BOOT", 200, minimum=20),
            cv_repeats=_env_int("EV_DIFFUSION_CV_REPEATS", 10, minimum=1),
            last_complete_year=_env_int("EV_DIFFUSION_LAST_COMPLETE_YEAR", None),
            market=_env_str("EV_DIFFUSION_MARKET", "US"),
            transformer_model=_env_str("EV_DIFFUSION_TRANSFORMER_MODEL", "distilbert-base-uncased"),
        )
