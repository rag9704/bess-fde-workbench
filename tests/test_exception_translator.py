import numpy as np
import pytest

from ai_harness.exception_translator import translate_exception_text


@pytest.mark.parametrize(
    "text, soc, cap",
    [
        ("reserve 80% from 14 to 20", [(14, 20, 1600.0)], []),
        ("reserve 1200 kWh", [(0, 23, 1200.0)], []),
        ("Typhoon warning, cap grid import at 300 kW", [(0, 23, 1500.0)], [(0, 23, 300.0)]),
        ("台風 警報", [(0, 23, 1500.0)], []),
        ("transformer limit 500kw", [], [(0, 23, 500.0)]),
        ("受電 600kW", [], [(0, 23, 600.0)]),
        ("hours14-20 reserve 50%", [(14, 20, 1000.0)], []),
        ("all normal today", [], []),
    ],
)
def test_supported_phrasings(text, soc, cap):
    patch = translate_exception_text(text, capacity_kwh=2000.0)
    assert patch.soc_min_overrides == soc
    assert patch.grid_cap_overrides == cap


@pytest.mark.xfail(strict=True, reason="Known gap: regex parser needs the number after the keyword")
def test_known_gap_number_before_keyword():
    assert translate_exception_text("hold 90% reserve").soc_min_overrides


@pytest.mark.xfail(strict=True, reason="Known gap: 'hours 14-20' with a space is not parsed")
def test_known_gap_hours_with_space():
    assert translate_exception_text("reserve 50% hours 14-20").soc_min_overrides[0][:2] == (14, 20)


def test_patch_apply_tightens_only():
    patch = translate_exception_text("reserve 80% from 2 to 3, cap grid import at 300 kW", capacity_kwh=2000.0)
    soc_min, grid_cap = patch.apply(horizon_len=6, default_soc_min_kwh=200.0, default_grid_cap_kw=1200.0)
    assert len(soc_min) == 7 and len(grid_cap) == 6
    assert list(soc_min[2:4]) == [1600.0, 1600.0] and soc_min[0] == 200.0
    assert np.all(grid_cap <= 1200.0)
