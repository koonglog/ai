from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from ai import dashboard_api


def _seed_db(db_url: str) -> None:
    engine = create_engine(db_url, future=True)
    now = datetime.now().replace(microsecond=0)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE households (
                    id INTEGER PRIMARY KEY,
                    alias TEXT,
                    building_name TEXT,
                    unit_number TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE noise_logs (
                    id INTEGER PRIMARY KEY,
                    household_id INTEGER,
                    sound_level FLOAT,
                    vibration_value FLOAT,
                    duration_ms INTEGER,
                    event_type TEXT,
                    severity TEXT,
                    status TEXT,
                    timestamp TEXT
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE mediations (
                    id INTEGER PRIMARY KEY,
                    household_id INTEGER,
                    target_unit TEXT,
                    recommended_action TEXT,
                    status TEXT,
                    created_at TEXT
                )
                """
            )
        )

        conn.execute(
            text(
                """
                INSERT INTO households (id, alias, building_name, unit_number)
                VALUES
                    (1, 'A-101', 'A', '101'),
                    (2, 'B-202', 'B', '202')
                """
            )
        )

        rows = [
            (1, 1, 62.0, 710, 4000, "impact_noise", "high", "new", now.isoformat(sep=" ")),
            (
                2,
                1,
                55.0,
                380,
                3000,
                "repeated_vibration",
                "high",
                "new",
                (now - timedelta(minutes=4)).isoformat(sep=" "),
            ),
            (
                3,
                1,
                47.0,
                220,
                2500,
                "daily_noise",
                "medium",
                "new",
                (now - timedelta(hours=1)).isoformat(sep=" "),
            ),
            (
                4,
                1,
                48.0,
                260,
                3500,
                "daily_noise",
                "medium",
                "new",
                (now - timedelta(days=1)).isoformat(sep=" "),
            ),
            (
                5,
                2,
                44.0,
                140,
                2000,
                "background_noise",
                "low",
                "new",
                (now - timedelta(hours=2)).isoformat(sep=" "),
            ),
        ]

        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO noise_logs
                    (id, household_id, sound_level, vibration_value, duration_ms, event_type, severity, status, timestamp)
                    VALUES
                    (:id, :household_id, :sound_level, :vibration_value, :duration_ms, :event_type, :severity, :status, :timestamp)
                    """
                ),
                {
                    "id": row[0],
                    "household_id": row[1],
                    "sound_level": row[2],
                    "vibration_value": row[3],
                    "duration_ms": row[4],
                    "event_type": row[5],
                    "severity": row[6],
                    "status": row[7],
                    "timestamp": row[8],
                },
            )

        conn.execute(
            text(
                """
                INSERT INTO mediations (id, household_id, target_unit, recommended_action, status, created_at)
                VALUES
                    (1, 1, 'A-101', 'admin_review', 'completed', :created1),
                    (2, 2, 'B-202', 'notice', 'pending', :created2)
                """
            ),
            {
                "created1": (now - timedelta(hours=1)).isoformat(sep=" "),
                "created2": (now - timedelta(minutes=30)).isoformat(sep=" "),
            },
        )


def _build_client(db_url: str) -> TestClient:
    dashboard_api.get_backend_settings.cache_clear()
    dashboard_api.get_engine.cache_clear()
    dashboard_api.get_schema.cache_clear()
    dashboard_api.get_settings.cache_clear()
    app = dashboard_api.create_dashboard_app()
    return TestClient(app)


def _new_db_url() -> str:
    tmp_dir = Path(__file__).resolve().parent / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    db_path = tmp_dir / f"dashboard_{uuid4().hex}.db"
    return f"sqlite:///{db_path}"


def test_noise_distribution_and_export(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    _seed_db(db_url)
    client = _build_client(db_url)

    res = client.get("/api/v1/noise/distribution", params={"days": 7})
    assert res.status_code == 200
    payload = res.json()
    assert payload["total_events"] == 5
    event_types = {row["event_type"]: row["count"] for row in payload["by_event_type"]}
    assert event_types["daily_noise"] == 2
    assert event_types["impact_noise"] == 1

    export_res = client.get("/api/v1/noise/distribution/export", params={"days": 7})
    assert export_res.status_code == 200
    assert "text/csv" in export_res.headers["content-type"]
    assert "event_type,severity,count" in export_res.text


def test_households_urgent_and_hourly(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    _seed_db(db_url)
    client = _build_client(db_url)

    households = client.get("/api/v1/dashboard/households", params={"days": 7})
    assert households.status_code == 200
    items = households.json()["items"]
    assert len(items) == 2
    assert any(item["unit_label"] == "A-101" for item in items)

    urgent = client.get("/api/v1/dashboard/urgent", params={"days": 7})
    assert urgent.status_code == 200
    assert isinstance(urgent.json()["items"], list)

    hourly = client.get("/api/v1/dashboard/hourly", params={"days": 7})
    assert hourly.status_code == 200
    series = hourly.json()["series"]
    assert len(series) == 24
    assert sum(item["count"] for item in series) == 5


def test_today_events_and_completed(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    _seed_db(db_url)
    client = _build_client(db_url)

    today = client.get("/api/v1/dashboard/today-events")
    assert today.status_code == 200
    assert today.json()["total_events"] >= 1

    completed = client.get("/api/v1/dashboard/completed")
    assert completed.status_code == 200
    items = completed.json()["items"]
    assert len(items) == 1
    assert items[0]["status"] == "completed"


def test_analyze_route(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    _seed_db(db_url)
    client = _build_client(db_url)

    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 58.2,
            "vibration_value": 640,
            "duration_ms": 4200,
            "accel_delta": 0.16,
            "timestamp": datetime.now().isoformat(),
            "recent_count_10min": 2,
        },
        "household_id": 1,
        "analysis_period_days": 7,
        "recent_meaningful_events_10min": [
            {
                "detected_at": (datetime.now() - timedelta(minutes=3)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
                "is_meaningful": True,
            },
            {
                "detected_at": (datetime.now() - timedelta(minutes=8)).isoformat(),
                "event_type": "repeated_vibration",
                "severity": "medium",
                "is_meaningful": True,
            },
        ],
        "noise_events": [
            {
                "detected_at": (datetime.now() - timedelta(days=1)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
            {
                "detected_at": (datetime.now() - timedelta(hours=3)).isoformat(),
                "event_type": "daily_noise",
                "severity": "medium",
            },
        ],
        "mediation_messages": [
            {"created_at": (datetime.now() - timedelta(hours=10)).isoformat()}
        ],
        "noise_type": "충격성 소리",
        "noise_time_slot": "주로 야간",
        "noise_frequency": "거의 매일",
        "situation_description": "늦은 밤에 반복되는 큰 소리",
        "generate_message": True,
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()

    assert body["status"] == "success"
    assert body["noise_log"]["event_type"] in {
        "impact_noise",
        "repeated_vibration",
        "daily_noise",
        "background_noise",
        "unknown",
    }
    assert body["noise_log"]["severity"] in {"low", "medium", "high", "critical"}
    assert isinstance(body["noise_log"]["is_meaningful"], bool)
    assert body["pattern_result"]["period_days"] == 7
    assert "needs_mediation" in body["pattern_result"]
    assert isinstance(body["message_created"], bool)
    if body["ai_result"] is not None:
        assert "신고 소음 유형: 충격성 소리" in body["ai_result"]["admin_summary"]

    alias_res = client.post("/api/v1/sensor-readings", json=payload)
    assert alias_res.status_code == 200
    alias_body = alias_res.json()
    assert alias_body["status"] == "success"
    assert alias_body["noise_log"]["sensor_id"] == payload["sensor_id"]


def test_analyze_route_blocks_repeated_vibration_when_only_raw_like_recent_logs(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    _seed_db(db_url)
    client = _build_client(db_url)

    now = datetime.now()
    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 48.0,
            "vibration_value": 420,
            "duration_ms": 3000,
            "accel_delta": 0.02,
            "timestamp": now.isoformat(),
        },
        # Raw-like rows: no event_type/severity => excluded from meaningful count.
        "recent_noise_logs": [
            {"detected_at": (now - timedelta(minutes=2)).isoformat()},
            {"detected_at": (now - timedelta(minutes=4)).isoformat()},
            {"detected_at": (now - timedelta(minutes=6)).isoformat()},
            {"detected_at": (now - timedelta(minutes=8)).isoformat()},
            {"detected_at": (now - timedelta(minutes=9)).isoformat()},
        ],
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["noise_log"]["event_type"] != "repeated_vibration"
    assert body["pattern_result"]["recent_count_10min"] == 0


def test_analyze_route_repeated_vibration_uses_meaningful_event_count(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    _seed_db(db_url)
    client = _build_client(db_url)

    now = datetime.now()
    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 48.0,
            "vibration_value": 420,
            "duration_ms": 3000,
            "accel_delta": 0.02,
            "timestamp": now.isoformat(),
        },
        "recent_meaningful_events_10min": [
            {
                "detected_at": (now - timedelta(minutes=2)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
            {
                "detected_at": (now - timedelta(minutes=4)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
            {
                "detected_at": (now - timedelta(minutes=8)).isoformat(),
                "event_type": "repeated_vibration",
                "severity": "medium",
            },
        ],
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["noise_log"]["event_type"] == "repeated_vibration"
    assert body["pattern_result"]["recent_count_10min"] == 3


def test_analyze_route_uses_noise_events_only_for_7day_analysis(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    _seed_db(db_url)
    client = _build_client(db_url)

    now = datetime.now()
    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 52.0,
            "vibration_value": 260,
            "duration_ms": 4200,
            "accel_delta": 0.01,
            "timestamp": now.isoformat(),
            "recent_count_10min": 0,
        },
        "household_id": 1,
        "analysis_period_days": 7,
        "noise_events": [
            {
                "detected_at": (now - timedelta(days=1)).isoformat(),
                "event_type": "daily_noise",
                "severity": "medium",
            },
            {
                "detected_at": (now - timedelta(days=10)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
        ],
        # Should be ignored because noise_events exists (fallback stage-out path).
        "recent_noise_logs": [
            {
                "detected_at": (now - timedelta(days=1)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
            {
                "detected_at": (now - timedelta(days=2)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
            },
        ],
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["pattern_result"]["period_days"] == 7
    assert body["pattern_result"]["total_count"] == 1


def test_classify_event_route(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    _seed_db(db_url)
    client = _build_client(db_url)

    now = datetime.now()
    payload = {
        "sensor_id": "SENSOR-A101-01",
        "household_id": 1,
        "source": "backend",
        "event_feature": {
            "sound_level": 58.2,
            "vibration_value": 640,
            "duration_ms": 4200,
            "accel_delta": 0.16,
            "timestamp": now.isoformat(),
            "recent_count_10min": 2,
        },
        "recent_meaningful_events_10min": [
            {
                "detected_at": (now - timedelta(minutes=3)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
                "is_meaningful": True,
            }
        ],
    }

    res = client.post("/api/v1/ai/classify-event", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["classification"]["sensor_id"] == payload["sensor_id"]
    assert body["classification"]["event_type"] in {
        "impact_noise",
        "repeated_vibration",
        "daily_noise",
        "background_noise",
        "unknown",
    }
    assert body["classification"]["severity"] in {"low", "medium", "high", "critical"}
    assert isinstance(body["classification"]["is_meaningful"], bool)


def test_analyze_patterns_route(monkeypatch) -> None:
    db_url = _new_db_url()
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    _seed_db(db_url)
    client = _build_client(db_url)

    now = datetime.now()
    payload = {
        "household_id": 1,
        "analysis_period_days": 7,
        "reference_time": now.isoformat(),
        "noise_events": [
            {
                "detected_at": (now - timedelta(minutes=5)).isoformat(),
                "event_type": "impact_noise",
                "severity": "high",
                "is_meaningful": True,
            },
            {
                "detected_at": (now - timedelta(days=9)).isoformat(),
                "event_type": "daily_noise",
                "severity": "medium",
                "is_meaningful": True,
            },
        ],
        "mediation_messages": [{"created_at": (now - timedelta(hours=3)).isoformat()}],
    }

    res = client.post("/api/v1/ai/analyze-patterns", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["pattern_result"]["household_id"] == 1
    assert body["pattern_result"]["period_days"] == 7
    assert body["pattern_result"]["total_count"] == 1
    assert body["pattern_result"]["recent_count_10min"] == 1


def test_shadow_mode_writes_classification_log(monkeypatch) -> None:
    db_url = _new_db_url()
    shadow_log = Path(__file__).resolve().parent / ".tmp" / f"shadow_{uuid4().hex}.jsonl"
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    monkeypatch.setenv("AI_CLASSIFIER_BACKEND", "rule")
    monkeypatch.setenv("AI_LGBM_SHADOW_MODE", "true")
    monkeypatch.setenv("AI_LGBM_SHADOW_LOG_PATH", str(shadow_log))
    _seed_db(db_url)
    client = _build_client(db_url)

    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 58.2,
            "vibration_value": 640,
            "duration_ms": 4200,
            "accel_delta": 0.16,
            "timestamp": datetime.now().isoformat(),
            "recent_count_10min": 2,
        },
        "household_id": 1,
        "analysis_period_days": 7,
        "generate_message": False,
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["noise_log"]["event_type"] in {
        "impact_noise",
        "repeated_vibration",
        "daily_noise",
        "background_noise",
        "unknown",
    }

    assert shadow_log.exists()
    lines = [line for line in shadow_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 1
    latest = json.loads(lines[-1])
    assert latest["route"] == "/api/v1/ai/analyze"
    assert "primary" in latest
    assert "rule" in latest
    assert "lightgbm" in latest
    assert "diff" in latest


def test_analyze_response_schema_unchanged_when_lightgbm_falls_back(monkeypatch) -> None:
    db_url = _new_db_url()
    model_dir = Path(__file__).resolve().parent / ".tmp" / f"missing_models_{uuid4().hex}"
    monkeypatch.setenv("BACKEND_DB_URL", db_url)
    monkeypatch.setenv("ENABLE_OPENAI", "false")
    monkeypatch.setenv("AI_CLASSIFIER_BACKEND", "lightgbm")
    monkeypatch.setenv("AI_LGBM_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("AI_LGBM_SHADOW_MODE", "false")
    _seed_db(db_url)
    client = _build_client(db_url)

    payload = {
        "sensor_id": "SENSOR-A101-01",
        "source": "arduino",
        "event_feature": {
            "sound_level": 58.2,
            "vibration_value": 640,
            "duration_ms": 4200,
            "accel_delta": 0.16,
            "timestamp": datetime.now().isoformat(),
            "recent_count_10min": 2,
        },
        "household_id": 1,
        "analysis_period_days": 7,
        "generate_message": False,
    }

    res = client.post("/api/v1/ai/analyze", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"

    noise_log_keys = set(body["noise_log"].keys())
    assert noise_log_keys == {
        "id",
        "sensor_id",
        "household_id",
        "event_type",
        "severity",
        "severity_score",
        "confidence",
        "is_night",
        "is_meaningful",
        "timestamp",
    }
    assert "classifier_backend" not in body["noise_log"]
    assert "rule_hits" not in body["noise_log"]
