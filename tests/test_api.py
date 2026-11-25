from fastapi.testclient import TestClient
import pytest

from service.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "highs" in r.json()["solvers"]


def test_optimize_default(client):
    r = client.post("/optimize", json={"hours": 24})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "OPTIMAL"
    assert len(body["schedule"]) == 24
    assert body["comparison"][-1]["savings_yen"] > 0
    assert "x-request-id" in r.headers


def test_optimize_alert_recovered(client):
    r = client.post("/optimize", json={"alert": "Typhoon, cap grid import at 300 kW", "include_schedule": False})
    body = r.json()
    assert body["status"] == "RECOVERED"
    assert body["violations"] and body["comparison"] is None and body["schedule"] is None


def test_optimize_custom_series_and_params(client):
    series = [
        {
            "timestamp": f"2026-01-01 {h:02d}:00",
            "load_kw": 300.0,
            "solar_kw": 0.0,
            "import_price_yen_kwh": 20.0 + (10.0 if 17 <= h <= 20 else 0.0),
            "export_price_yen_kwh": 5.0,
        }
        for h in range(24)
    ]
    r = client.post(
        "/optimize",
        json={"series": series, "battery": {"capacity_kwh": 500, "init_soc_kwh": 250, "end_soc_min_kwh": 100}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "OPTIMAL"


def test_invalid_series_returns_all_issues(client):
    series = [
        {
            "timestamp": "2026-01-01 00:00",
            "load_kw": -5,
            "solar_kw": 0,
            "import_price_yen_kwh": 20,
            "export_price_yen_kwh": 5,
        },
        {
            "timestamp": "2026-01-01 03:00",
            "load_kw": 5,
            "solar_kw": 0,
            "import_price_yen_kwh": 20,
            "export_price_yen_kwh": 5,
        },
    ]
    r = client.post("/optimize", json={"series": series})
    assert r.status_code == 422
    issues = r.json()["detail"]["issues"]
    assert len(issues) == 2


def test_request_validation(client):
    assert client.post("/optimize", json={"hours": 0}).status_code == 422
    assert client.post("/optimize", json={"solver": "gurobi"}).status_code == 422
    assert client.post("/optimize", json={"battery": {"capacity_kwh": 100}}).status_code == 422  # init SoC > capacity


def test_unknown_start_returns_404(client):
    assert client.post("/optimize", json={"start": "2030-01-01"}).status_code == 404
