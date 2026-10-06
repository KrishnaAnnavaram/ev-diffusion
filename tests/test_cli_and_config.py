import json
import subprocess
import sys

import pytest

from ev_diffusion.cli import main
from ev_diffusion.config import Settings

FAST = ["--seed", "3"]


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("EV_DIFFUSION_N_BOOT", "50")
    monkeypatch.setenv("EV_DIFFUSION_LAST_COMPLETE_YEAR", "2022")
    settings = Settings.from_env()
    assert settings.n_boot == 50 and settings.last_complete_year == 2022


def test_invalid_env_value_is_refused(monkeypatch):
    monkeypatch.setenv("EV_DIFFUSION_SEED", "abc")
    with pytest.raises(ValueError, match="EV_DIFFUSION_SEED"):
        Settings.from_env()


def test_synth_coverage_fit_backtest(data_dir, capsys, tmp_path):
    reg, cov = str(data_dir / "registrations.csv"), str(data_dir / "covariates.csv")
    assert main(["coverage", "--registrations", reg]) == 0
    assert "looks partial" in capsys.readouterr().out
    assert main(["fit", "--registrations", reg, "--covariates", cov]) == 0
    out = capsys.readouterr().out
    assert "beta_charging_ports" in out and "f_p_value" in out
    assert main(["backtest", "--registrations", reg, "--horizon", "2", "--out", str(tmp_path / "bt.csv")]) == 0
    assert (tmp_path / "bt.csv").exists()


def test_sentiment_and_aspects(data_dir, capsys):
    reviews = str(data_dir / "reviews.csv")
    assert main(["sentiment", "--reviews", reviews, "--cv-repeats", "1", "--models", "majority", "logreg"]) == 0
    assert "macro_f1" in capsys.readouterr().out
    assert main(["aspects", "--reviews", reviews]) == 0
    assert "charging" in capsys.readouterr().out


def test_report_writes_all_outputs(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DIFFUSION_N_BOOT", "20")
    monkeypatch.setenv("EV_DIFFUSION_CV_REPEATS", "1")
    out = tmp_path / "report"
    assert main(["report", "--data-dir", str(data_dir), "--out", str(out), *FAST]) == 0
    summary = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert summary["series_source"] == "national"
    assert "gbm_comparison" in summary
    for name in ("forecast.csv", "backtest.csv", "cv_folds.csv", "aspects.csv", "report.md"):
        assert (out / name).exists()


def test_report_without_data_returns_an_error(tmp_path):
    assert main(["report", "--data-dir", str(tmp_path), "--out", str(tmp_path / "o")]) == 2


def test_demo_runs_offline(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo"), "--n-boot", "20", "--cv-repeats", "1"]) == 0
    assert "SYNTHETIC DATA" in capsys.readouterr().out


def test_core_package_does_not_import_heavy_frameworks():
    code = "import ev_diffusion.cli, ev_diffusion.report, sys; print('torch' in sys.modules, 'transformers' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False False"


def test_transformer_model_gives_a_clear_error_without_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    from ev_diffusion.sentiment.models import make_model

    with pytest.raises(ImportError, match="ev-diffusion\\[transformer\\]"):
        make_model("transformer")
