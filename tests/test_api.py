"""API tests, including that the operator surface is actually closed."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.runtime import reset_runtime

TOKEN = {"X-Operator-Token": "test-token"}


@pytest.fixture(scope="module")
def client():
    reset_runtime()
    with TestClient(app) as test_client:
        time.sleep(1.5)  # let the control loop produce a few frames
        yield test_client


def test_health_reports_control_mode_and_interlock_count(client):
    body = client.get("/api/health").json()
    assert body["status"] in ("ok", "faulted")
    assert body["control_mode"] in ("adaptive", "fixed_time")
    assert body["interlock_events"] == 0


def test_state_includes_the_command_and_the_reasoning_behind_it(client):
    body = client.get("/api/state").json()
    assert body["status"] == "running"
    assert set(body["command"]["vehicle"]) == {"north", "south", "east", "west"}
    assert set(body["pressure"]) == {"NS", "EW"}
    for terms in body["pressure_terms"].values():
        assert {"queue", "occupancy", "forecast", "wait", "pedestrian"} == set(terms)


def test_layout_exposes_the_conflict_matrix(client):
    body = client.get("/api/layout").json()
    assert "east" in body["conflicts"]["north"]
    assert "south" not in body["conflicts"]["north"]
    assert body["phases"]["PED"]["vehicles"] == []


def test_series_returns_samples_bounded_by_the_limit(client):
    body = client.get("/api/series?limit=5").json()
    assert len(body["samples"]) <= 5
    if body["samples"]:
        assert "total_queue" in body["samples"][0]


def test_safety_endpoint_is_empty_in_normal_operation(client):
    body = client.get("/api/safety").json()
    assert body["statistics"]["total"] == 0
    assert body["faulted"] is False


def test_operator_routes_reject_a_missing_token(client):
    for path, payload in (
        ("/api/signals/ped-call", {"crosswalk": "cw_north"}),
        ("/api/signals/emergency", {"approach": "north"}),
        ("/api/signals/mode", {"mode": "fixed_time"}),
        ("/api/ml/train", None),
    ):
        response = client.post(path, json=payload)
        assert response.status_code in (401, 503), (path, response.status_code)


def test_operator_routes_reject_a_wrong_token(client):
    response = client.post(
        "/api/signals/mode",
        json={"mode": "fixed_time"},
        headers={"X-Operator-Token": "nope"},
    )
    assert response.status_code == 401


def test_pedestrian_call_is_accepted_with_a_token(client):
    response = client.post(
        "/api/signals/ped-call", json={"crosswalk": "cw_north"}, headers=TOKEN
    )
    assert response.status_code == 200
    assert response.json()["queued"] is True


def test_unknown_crosswalk_is_a_validation_error(client):
    response = client.post(
        "/api/signals/ped-call", json={"crosswalk": "cw_up"}, headers=TOKEN
    )
    assert response.status_code == 422


def test_emergency_reports_the_remaining_preemption_budget(client):
    response = client.post(
        "/api/signals/emergency",
        json={"approach": "north", "kind": "ambulance", "duration": 20},
        headers=TOKEN,
    )
    assert response.status_code == 200
    assert response.json()["budget_remaining"] >= 0


def test_weather_change_lengthens_clearance(client):
    clear = client.post(
        "/api/signals/weather", json={"weather": "clear"}, headers=TOKEN
    ).json()
    snow = client.post(
        "/api/signals/weather", json={"weather": "snow"}, headers=TOKEN
    ).json()
    assert snow["all_red"] > clear["all_red"]
    assert snow["discharge_rate"] < clear["discharge_rate"]
    client.post("/api/signals/weather", json={"weather": "clear"}, headers=TOKEN)


def test_unknown_mode_is_rejected(client):
    response = client.post(
        "/api/signals/mode", json={"mode": "vibes"}, headers=TOKEN
    )
    assert response.status_code == 422


def test_forecast_endpoint_names_what_it_is_serving(client):
    body = client.get("/api/ml/forecast").json()
    assert body["unit"] == "vehicles_per_minute"
    assert set(body["serving"]) == {"north", "south", "east", "west"}
    assert all(v in ("model", "baseline") for v in body["serving"].values())


def test_websocket_pushes_state(client):
    with client.websocket_connect("/ws") as ws:
        message = json.loads(ws.receive_text())
    assert message["type"] == "state"
    assert "command" in message["data"]


def test_dashboard_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
