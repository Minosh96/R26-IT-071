import os
import pathlib
import sys

import pytest

# The component uses paths relative to its own folder (models/saved/..., data/...)
COMPONENT_DIR = pathlib.Path(__file__).parent.parent
os.chdir(COMPONENT_DIR)
sys.path.insert(0, str(COMPONENT_DIR))


@pytest.fixture(scope="session")
def api():
    import api.app as app_module
    return app_module


@pytest.fixture(scope="session")
def client(api):
    return api.app.test_client()


@pytest.fixture(scope="session")
def auth(api):
    return {"Authorization": f"Bearer {api.API_TOKEN}"}


@pytest.fixture(scope="session")
def assets(api):
    return api.PRELOADED_ASSETS


@pytest.fixture
def alto():
    """A typical 2015 Alto, well inside the training data."""
    return {"maf_year": 2015, "reg_year": 2015, "mileage_km": 100000, "previous_owners": 2,
            "is_reconditioned": 0, "power_shutters": 1, "power_mirrors": 0}
