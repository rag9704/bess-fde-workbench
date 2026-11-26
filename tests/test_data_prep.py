"""The data-prep script: both TEPCO encodings, complete days only, and committed files rebuild exactly."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep", ROOT / "data" / "prepare_japan_grid_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("month", ["202604", "202607"])  # UTF-8 and Shift-JIS files
def test_reads_both_encodings(prep, month):
    df = prep.read_tepco_csv(RAW / f"eria_jukyu_{month}_03.csv")
    assert "エリア需要" in df.columns and len(df) >= 1440


def test_april_rebuilds_byte_identical(prep, tmp_path):
    out = tmp_path / "april.csv"
    prep.build_dataset(RAW / "eria_jukyu_202604_03.csv", RAW / "spot_summary_2026.csv", out)
    assert out.read_bytes() == (ROOT / "data" / "april_2026_hourly.csv").read_bytes()


def test_full_year_dataset_is_complete_days_and_matches_committed(prep, tmp_path):
    out = tmp_path / "all.csv"
    prep.build_dataset(sorted(RAW.glob("eria_jukyu_*.csv")), RAW / "spot_summary_2026.csv", out)
    df = pd.read_csv(out, parse_dates=["timestamp"])
    assert len(df) % 24 == 0 and df["timestamp"].iloc[-1].hour == 23
    assert (np.diff(df["timestamp"].to_numpy()) == np.timedelta64(1, "h")).all()
    assert df["load_kw"].max() == pytest.approx(1000.0) and df["solar_kw"].max() == pytest.approx(600.0)
    assert out.read_bytes() == (ROOT / "data" / "japan_2026_hourly.csv").read_bytes()
