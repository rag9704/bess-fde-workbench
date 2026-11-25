import numpy as np
import pytest

from core_model.data_validation import DataValidationError, validate_timeseries


def test_bundled_data_is_valid(dataset):
    assert validate_timeseries(dataset) == []


def test_missing_column(day):
    with pytest.raises(DataValidationError, match="Missing required column"):
        validate_timeseries(day.drop(columns=["solar_kw"]))


def test_reports_all_issues_at_once(day):
    bad = day.copy()
    bad.loc[3, "load_kw"] = -10
    bad.loc[5, "import_price_yen_kwh"] = np.nan
    bad = bad.drop(index=10)
    with pytest.raises(DataValidationError) as exc:
        validate_timeseries(bad)
    text = " ".join(exc.value.issues)
    assert "negative" in text and "missing" in text and "gap" in text


def test_duplicate_timestamps(day):
    bad = day.copy()
    bad.loc[1, "timestamp"] = bad.loc[0, "timestamp"]
    with pytest.raises(DataValidationError, match="duplicated"):
        validate_timeseries(bad)


def test_negative_price_is_only_a_warning(day):
    d = day.copy()
    d.loc[2, "import_price_yen_kwh"] = -1.0
    warnings = validate_timeseries(d)
    assert any("negative import price" in w for w in warnings)
