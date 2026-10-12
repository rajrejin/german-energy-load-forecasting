"""API contract and error paths (empty store: no forecast payloads are needed)."""

import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.settings import ApiSettings
from energy.store.base import ArtifactStore
from energy.store.local import LocalStore


@pytest.fixture()
def client(tmp_path):
    app = create_app(LocalStore(tmp_path / "s"), ApiSettings(json_logs=False))
    return TestClient(app, raise_server_exceptions=False)


def test_health_is_always_ok(client):
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_ready_is_503_without_artifacts(client):
    r = client.get("/v1/ready")
    assert r.status_code == 503 and r.json()["ready"] is False


def test_zone_whitelist_and_error_schema(client):
    r = client.get("/v1/forecast", params={"zone": "Bavaria"})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_zone" and err["request_id"] and "Traceback" not in r.text
    assert client.get("/v1/forecast").status_code == 422  # zone is required


def test_hour_bounds(client):
    for h in (-1, 24, "x"):
        r = client.get("/v1/explain", params={"zone": "DE", "hour": h})
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_parameter"


@pytest.mark.parametrize("bad", ["not-a-date", "2999-01-01", "1999-01-01"])
def test_issue_time_bounds(client, bad):
    r = client.get("/v1/forecast", params={"zone": "DE", "issue_time": bad})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_issue_time"


def test_not_found_without_forecast(client):
    for url, p in (
        ("/v1/forecast", {"zone": "DE"}),
        ("/v1/explain", {"zone": "DE", "hour": 3}),
        ("/v1/history", {"zone": "DE"}),
        ("/v1/briefing/latest", {}),
        ("/v1/results/ladder.csv", {}),
    ):
        r = client.get(url, params=p)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", url


def test_results_name_is_validated(client):
    assert client.get("/v1/results/..%2Fsecrets.json").status_code in (404, 422)
    assert client.get("/v1/results/bad name.json").status_code in (404, 422)
    assert client.get("/v1/results/UPPER.JSON").status_code == 422


def test_results_served_when_published(tmp_path):
    store = LocalStore(tmp_path / "s")
    store.put_bytes("results/ladder.csv", b"model,mae\nlstm,1\n")
    store.put_bytes("results/data_quality.json", b'{"a": 1}')
    c = TestClient(create_app(store, ApiSettings(json_logs=False)))
    r = c.get("/v1/results/ladder.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert c.get("/v1/results/data_quality.json").json() == {"a": 1}


def test_api_key_enforced_when_configured(tmp_path):
    app = create_app(LocalStore(tmp_path / "s"), ApiSettings(json_logs=False, api_key="s3cret"))
    c = TestClient(app)
    assert c.get("/v1/zones").status_code == 401
    assert c.get("/v1/zones", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/v1/zones", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/v1/health").status_code == 200  # liveness stays open
    assert c.get("/metrics").status_code == 401


def test_cors_is_off_by_default_and_opt_in(tmp_path):
    off = TestClient(create_app(LocalStore(tmp_path / "a"), ApiSettings(json_logs=False)))
    r = off.get("/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    on = TestClient(
        create_app(
            LocalStore(tmp_path / "b"),
            ApiSettings(json_logs=False, cors_origins="http://ok.example"),
        )
    )
    r = on.get("/v1/health", headers={"Origin": "http://ok.example"})
    assert r.headers.get("access-control-allow-origin") == "http://ok.example"


class BrokenStore(LocalStore):
    def get_bytes(self, key: str) -> bytes:
        raise RuntimeError("secret internal path D:/private/thing")


class SlowStore(LocalStore):
    def exists(self, key: str) -> bool:
        time.sleep(0.5)
        return False


def test_unexpected_errors_do_not_leak_details(tmp_path):
    c = TestClient(
        create_app(BrokenStore(tmp_path / "s"), ApiSettings(json_logs=False)),
        raise_server_exceptions=False,
    )
    r = c.get("/v1/forecast", params={"zone": "DE"})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text and "private" not in r.text


def test_request_timeout_returns_504(tmp_path):
    store: ArtifactStore = SlowStore(tmp_path / "s")
    c = TestClient(create_app(store, ApiSettings(json_logs=False, request_timeout_s=0.05)))
    r = c.get("/v1/zones")
    assert r.status_code == 504 and r.json()["error"]["code"] == "timeout"


def test_request_id_is_propagated(client):
    r = client.get("/v1/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
