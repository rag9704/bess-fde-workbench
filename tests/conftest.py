import numpy as np
import pandas as pd
import pytest

from core_model.formulation import BessParams
from service.pipeline import DEFAULT_DATA, load_data


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return load_data(DEFAULT_DATA)


@pytest.fixture(scope="session")
def day(dataset) -> pd.DataFrame:
    return dataset.iloc[:24].reset_index(drop=True)


@pytest.fixture(scope="session")
def week(dataset) -> pd.DataFrame:
    return dataset.iloc[:168].reset_index(drop=True)


@pytest.fixture(scope="session")
def params() -> BessParams:
    return BessParams()


@pytest.fixture(scope="session")
def typhoon_overrides():
    """75% reserve + 300 kW import cap all day: infeasible against ~700 kW site load."""
    return {"soc_min_kwh": np.full(24, 1500.0), "grid_import_cap_kw": np.full(24, 300.0)}
