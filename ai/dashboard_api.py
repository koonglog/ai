from __future__ import annotations

import csv
import io
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from .config import AISettings, ClassifierSettings, get_settings
from .event_classifier import classify_event, is_meaningful_event
from .lgbm_runtime import LightGBMRuntime
from .message_generator import generate_mediation_message
from .pattern_analyzer import analyze_patterns
from .schemas import EventClassificationResult, EventFeatures


@dataclass(frozen=True)
class BackendDBSettings:
    database_url: str
    timezone_name: str = "Asia/Seoul"
    default_period_days: int = 7

    @classmethod
    def from_env(cls) -> "BackendDBSettings":
        database_url = os.getenv("BACKEND_DB_URL")
        if not database_url:
            default_path = os.getenv("BACKEND_DB_PATH", "./_backend_db_expand/kunglog.db")
            database_url = f"sqlite:///{default_path}"
        return cls(
            database_url=database_url,
            timezone_name=os.getenv("AI_TIMEZONE", "Asia/Seoul"),
            default_period_days=int(os.getenv("AI_DASHBOARD_DEFAULT_DAYS", "7")),
        )


@dataclass(frozen=True)
class BackendSchema:
    noise_table: str
    noise_timestamp_col: str
    noise_columns: set[str]
    household_table: str | None
    household_columns: set[str]
    mediation_table: str | None
    mediation_columns: set[str]


COMPLETED_STATUSES = {
    "completed",
    "resolved",
    "done",
    "sent",
    "closed",
    "조치완료",
    "완료",
    "해결",
}


class AnalyzeEventFeatureIn(BaseModel):
    sound_level: float
    vibration_value: float
    duration_ms: int = 0
    accel_delta: float = 0.0
    timestamp: datetime | None = None
    # Contract: meaningful-event count in last 10 minutes only.
    recent_count_10min: int | None = Field(default=None, ge=0)


class AnalyzeNoiseEventIn(BaseModel):
    detected_at: datetime | None = None
    timestamp: datetime | None = None
    event_type: str | None = None
    severity: str | None = None
    is_meaningful: bool | None = None


class AnalyzeMediationMessageIn(BaseModel):
    created_at: datetime


class AnalyzeRequestIn(BaseModel):
    sensor_id: str
    source: str = "backend"

    # Preferred contract (event-level aggregated feature).
    event_feature: AnalyzeEventFeatureIn | None = None

    # Legacy raw fields (stage-out path; only used when event_feature is omitted).
    sound_level: float | None = None
    vibration_value: float | None = None
    duration_ms: int | None = None
    timestamp: datetime | None = None
    acceleration: dict[str, float] | None = None

    # Optional context for richer AI analysis.
    household_id: int | None = None
    target_unit: str | None = None
    recent_meaningful_events_10min: list[AnalyzeNoiseEventIn] = Field(default_factory=list)
    noise_events: list[AnalyzeNoiseEventIn] = Field(default_factory=list)
    # Legacy context path (stage-out): accepted, but raw fallback classification is disabled.
    recent_noise_logs: list[AnalyzeNoiseEventIn] = Field(default_factory=list)
    mediation_messages: list[AnalyzeMediationMessageIn] = Field(default_factory=list)
    analysis_period_days: int = Field(default=7, ge=1, le=90)
    generate_message: bool = True
    # Optional manual context from mediation request UI.
    noise_type: str | None = None
    noise_time_slot: str | None = None
    noise_frequency: str | None = None
    situation_description: str | None = None


class ClassifyEventRequestIn(BaseModel):
    sensor_id: str
    source: str = "backend"
    household_id: int | None = None
    event_feature: AnalyzeEventFeatureIn
    recent_meaningful_events_10min: list[AnalyzeNoiseEventIn] = Field(default_factory=list)


class AnalyzePatternsRequestIn(BaseModel):
    household_id: int
    analysis_period_days: int = Field(default=7, ge=1, le=90)
    reference_time: datetime | None = None
    noise_events: list[AnalyzeNoiseEventIn] = Field(default_factory=list)
    mediation_messages: list[AnalyzeMediationMessageIn] = Field(default_factory=list)


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        normalized = normalized.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _as_local(dt: datetime, tz: timezone) -> datetime:
    if dt.tzinfo is None:
        # Backend logs are usually written in local server time.
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _choose(candidates: list[str], available: set[str], required: bool = False) -> str | None:
    for name in candidates:
        if name in available:
            return name
    if required:
        raise RuntimeError(f"Required column not found. candidates={candidates}")
    return None


@lru_cache(maxsize=1)
def get_backend_settings() -> BackendDBSettings:
    return BackendDBSettings.from_env()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = get_backend_settings()
    return create_engine(cfg.database_url, future=True, pool_pre_ping=True)


@lru_cache(maxsize=1)
def get_schema() -> BackendSchema:
    engine = get_engine()
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    noise_table = _choose(["noise_logs", "noise_log", "noise_events"], table_names, required=True)
    assert noise_table is not None
    noise_columns = {col["name"] for col in inspector.get_columns(noise_table)}
    noise_timestamp_col = _choose(
        ["timestamp", "detected_at", "created_at"],
        noise_columns,
        required=True,
    )
    assert noise_timestamp_col is not None

    household_table = _choose(["households", "household"], table_names)
    household_columns = (
        {col["name"] for col in inspector.get_columns(household_table)}
        if household_table
        else set()
    )

    mediation_table = _choose(["mediations", "mediation"], table_names)
    mediation_columns = (
        {col["name"] for col in inspector.get_columns(mediation_table)}
        if mediation_table
        else set()
    )

    return BackendSchema(
        noise_table=noise_table,
        noise_timestamp_col=noise_timestamp_col,
        noise_columns=noise_columns,
        household_table=household_table,
        household_columns=household_columns,
        mediation_table=mediation_table,
        mediation_columns=mediation_columns,
    )


def _get_local_tz() -> timezone:
    # KST fixed offset (UTC+09:00) for dashboard consistency.
    return timezone(timedelta(hours=9))


def _query_noise_logs(
    *,
    days: int | None = None,
    household_id: int | None = None,
    since: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    schema = get_schema()
    tz = _get_local_tz()
    now_local = datetime.now(tz)
    start_local = None
    if since is not None:
        start_local = since if since.tzinfo else since.replace(tzinfo=tz)
    elif days is not None:
        start_local = now_local - timedelta(days=days)

    id_col = _choose(["id", "noise_log_id"], schema.noise_columns)
    household_col = _choose(["household_id"], schema.noise_columns)
    event_col = _choose(["event_type", "noise_type"], schema.noise_columns)
    severity_col = _choose(["severity"], schema.noise_columns)
    sound_col = _choose(["sound_level", "decibel"], schema.noise_columns)
    vibration_col = _choose(["vibration_value"], schema.noise_columns)
    duration_col = _choose(["duration_ms"], schema.noise_columns)
    status_col = _choose(["status"], schema.noise_columns)

    select_parts = [
        f"{id_col} AS id" if id_col else "NULL AS id",
        f"{household_col} AS household_id" if household_col else "NULL AS household_id",
        f"{schema.noise_timestamp_col} AS detected_at",
        f"{event_col} AS event_type" if event_col else "NULL AS event_type",
        f"{severity_col} AS severity" if severity_col else "NULL AS severity",
        f"{sound_col} AS sound_level" if sound_col else "NULL AS sound_level",
        f"{vibration_col} AS vibration_value" if vibration_col else "NULL AS vibration_value",
        f"{duration_col} AS duration_ms" if duration_col else "NULL AS duration_ms",
        f"{status_col} AS status" if status_col else "NULL AS status",
    ]

    sql = f"SELECT {', '.join(select_parts)} FROM {schema.noise_table} WHERE 1=1"
    params: dict[str, Any] = {}

    if household_id is not None and household_col:
        sql += f" AND {household_col} = :household_id"
        params["household_id"] = household_id

    if start_local is not None:
        sql += f" AND {schema.noise_timestamp_col} >= :start_dt"
        params["start_dt"] = start_local.isoformat(sep=" ")

    sql += f" ORDER BY {schema.noise_timestamp_col} DESC"
    if limit is not None and limit > 0:
        sql += " LIMIT :limit"
        params["limit"] = int(limit)

    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()

    ai_settings = get_settings()
    normalized: list[dict[str, Any]] = []

    for row in rows:
        record = dict(row)
        detected_at = _parse_dt(record.get("detected_at"))
        if not detected_at:
            continue
        detected_at = _as_local(detected_at, tz)

        event_type = str(record.get("event_type") or "").strip() or None
        severity = str(record.get("severity") or "").strip() or None
        sound = float(record.get("sound_level") or 0.0)
        vibration = int(float(record.get("vibration_value") or 0))
        duration_ms = int(record.get("duration_ms") or 0)

        if event_type is None or severity is None:
            # If backend has not classified yet, classify on read path.
            event_features = EventFeatures(
                device_id=str(record.get("id") or "unknown"),
                source="backend_db",
                sound_level=sound,
                vibration_value=vibration,
                duration_ms=duration_ms,
                accel_delta=0.0,
                timestamp=detected_at,
                recent_count_10min=0,
            )
            result = classify_event(event=event_features, settings=ai_settings)
            if event_type is None:
                event_type = result.event_type
            if severity is None:
                severity = result.severity

        normalized.append(
            {
                "id": record.get("id"),
                "household_id": int(record["household_id"]) if record.get("household_id") is not None else None,
                "detected_at": detected_at,
                "event_type": event_type or "unknown",
                "severity": severity or "low",
                "sound_level": sound,
                "vibration_value": vibration,
                "duration_ms": duration_ms,
                "status": record.get("status"),
            }
        )

    return normalized


def _query_household_labels() -> dict[int, str]:
    schema = get_schema()
    if not schema.household_table:
        return {}

    id_col = _choose(["id", "household_id"], schema.household_columns, required=True)
    alias_col = _choose(["alias", "unit_label", "unit_number"], schema.household_columns)
    building_col = _choose(["building_name", "building"], schema.household_columns)
    unit_col = _choose(["unit_number", "unit"], schema.household_columns)
    assert id_col is not None

    select_parts = [f"{id_col} AS household_id"]
    if alias_col:
        select_parts.append(f"{alias_col} AS alias")
    else:
        select_parts.append("NULL AS alias")
    if building_col:
        select_parts.append(f"{building_col} AS building_name")
    else:
        select_parts.append("NULL AS building_name")
    if unit_col:
        select_parts.append(f"{unit_col} AS unit_number")
    else:
        select_parts.append("NULL AS unit_number")

    sql = f"SELECT {', '.join(select_parts)} FROM {schema.household_table}"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql)).mappings().all()

    labels: dict[int, str] = {}
    for row in rows:
        household_id = int(row["household_id"])
        alias = str(row.get("alias") or "").strip()
        if alias:
            labels[household_id] = alias
            continue
        building = str(row.get("building_name") or "").strip()
        unit = str(row.get("unit_number") or "").strip()
        if building and unit:
            labels[household_id] = f"{building}-{unit}"
        elif unit:
            labels[household_id] = unit
        else:
            labels[household_id] = f"household-{household_id}"
    return labels


def _query_mediations(limit: int = 200) -> list[dict[str, Any]]:
    schema = get_schema()
    if not schema.mediation_table:
        return []

    med_id_col = _choose(["id"], schema.mediation_columns, required=True)
    household_col = _choose(["household_id"], schema.mediation_columns)
    target_col = _choose(["target_unit"], schema.mediation_columns)
    action_col = _choose(["recommended_action"], schema.mediation_columns)
    status_col = _choose(["status"], schema.mediation_columns)
    created_col = _choose(["created_at", "timestamp"], schema.mediation_columns)
    assert med_id_col is not None

    select_parts = [f"{med_id_col} AS id"]
    select_parts.append(f"{household_col} AS household_id" if household_col else "NULL AS household_id")
    select_parts.append(f"{target_col} AS target_unit" if target_col else "NULL AS target_unit")
    select_parts.append(f"{action_col} AS recommended_action" if action_col else "NULL AS recommended_action")
    select_parts.append(f"{status_col} AS status" if status_col else "NULL AS status")
    select_parts.append(f"{created_col} AS created_at" if created_col else "NULL AS created_at")

    order_col = created_col or med_id_col
    sql = f"SELECT {', '.join(select_parts)} FROM {schema.mediation_table} ORDER BY {order_col} DESC LIMIT :limit"
    with get_engine().connect() as conn:
        rows = conn.execute(text(sql), {"limit": int(limit)}).mappings().all()
    return [dict(row) for row in rows]


def _pattern_for_household(
    household_id: int,
    logs: list[dict[str, Any]],
    mediation_messages: list[dict[str, Any]],
    days: int,
) -> dict[str, Any]:
    def _naive(dt_value: datetime) -> datetime:
        return dt_value.astimezone(timezone.utc).replace(tzinfo=None) if dt_value.tzinfo else dt_value

    pattern_logs = [
        {
            "detected_at": _naive(row["detected_at"]),
            "event_type": row["event_type"],
            "severity": row["severity"],
        }
        for row in logs
    ]
    mediation_payload = []
    for row in mediation_messages:
        created_at = _parse_dt(row.get("created_at"))
        mediation_payload.append({"created_at": _naive(created_at) if created_at else None})
    pattern = analyze_patterns(
        household_id=household_id,
        noise_events=pattern_logs,
        mediation_messages=mediation_payload,
        analysis_period_days=days,
        reference_time=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    return {
        "pattern_label": pattern.pattern_label,
        "repeated_days": pattern.repeated_days,
        "night_events": pattern.night_events,
        "max_events_in_10min": pattern.max_events_in_10min,
        "needs_mediation": pattern.needs_mediation,
        "needs_escalation": pattern.needs_escalation,
        "post_mediation_recurrence": pattern.post_mediation_recurrence,
        "summary": pattern.summary,
    }


def _build_household_status(
    rows: list[dict[str, Any]],
    labels: dict[int, str],
    mediation_rows: list[dict[str, Any]],
    days: int,
) -> list[dict[str, Any]]:
    by_household: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        household_id = row.get("household_id")
        if household_id is None:
            continue
        by_household[int(household_id)].append(row)

    mediation_by_household: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for med in mediation_rows:
        household_id = med.get("household_id")
        if household_id is None:
            continue
        mediation_by_household[int(household_id)].append(med)

    results: list[dict[str, Any]] = []
    for household_id, logs in by_household.items():
        latest = max(logs, key=lambda x: x["detected_at"])
        severity_counter = Counter(row["severity"] for row in logs)
        high_critical = severity_counter.get("high", 0) + severity_counter.get("critical", 0)
        pattern = _pattern_for_household(
            household_id=household_id,
            logs=logs,
            mediation_messages=mediation_by_household.get(household_id, []),
            days=days,
        )

        if pattern["needs_escalation"]:
            status = "urgent"
        elif pattern["needs_mediation"]:
            status = "monitoring"
        else:
            status = "stable"

        results.append(
            {
                "household_id": household_id,
                "unit_label": labels.get(household_id, f"household-{household_id}"),
                "total_events": len(logs),
                "high_or_critical_events": high_critical,
                "last_event_at": latest["detected_at"].isoformat(),
                "last_event_type": latest["event_type"],
                "status": status,
                **pattern,
            }
        )
    return sorted(results, key=lambda x: (x["status"] != "urgent", -x["total_events"]))


def _calc_accel_delta_from_vector(acceleration: dict[str, float] | None) -> float:
    if not acceleration:
        return 0.0
    x = float(acceleration.get("x", 0.0))
    y = float(acceleration.get("y", 0.0))
    z = float(acceleration.get("z", 1.0))
    magnitude = math.sqrt(x * x + y * y + z * z)
    return abs(magnitude - 1.0)


def _normalize_noise_events(events: list[AnalyzeNoiseEventIn]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        dt = event.detected_at or event.timestamp
        if dt is None:
            continue
        event_type = str(event.event_type or "").strip()
        severity = str(event.severity or "").strip()
        if not event_type or not severity:
            continue
        normalized.append(
            {
                "detected_at": _to_naive_utc(dt),
                "event_type": event_type,
                "severity": severity,
                "is_meaningful": event.is_meaningful,
            }
        )
    return normalized


def _count_recent_meaningful_events(
    recent_events: list[dict[str, Any]],
    reference_time: datetime,
) -> int:
    count = 0
    for row in recent_events:
        dt = row["detected_at"]
        seconds = (reference_time - dt).total_seconds()
        if seconds < 0 or seconds > 600:
            continue
        is_meaningful = row.get("is_meaningful")
        if is_meaningful is None:
            is_meaningful = is_meaningful_event(row["event_type"], row["severity"])
        if is_meaningful:
            count += 1
    return count


def _build_event_features(
    *,
    sensor_id: str,
    source: str,
    feature_in: AnalyzeEventFeatureIn,
    fallback_timestamp: datetime | None = None,
    recent_context_events: list[dict[str, Any]] | None = None,
) -> tuple[datetime, EventFeatures]:
    sensor_ts = _to_naive_utc(feature_in.timestamp or fallback_timestamp or datetime.now(timezone.utc))
    recent_count_10min = (
        int(feature_in.recent_count_10min)
        if feature_in.recent_count_10min is not None
        else _count_recent_meaningful_events(recent_context_events or [], sensor_ts)
    )
    event_features = EventFeatures(
        device_id=sensor_id,
        source=source,
        sound_level=float(feature_in.sound_level),
        vibration_value=int(feature_in.vibration_value),
        duration_ms=int(feature_in.duration_ms or 0),
        accel_delta=float(feature_in.accel_delta),
        timestamp=sensor_ts,
        recent_count_10min=recent_count_10min,
    )
    return sensor_ts, event_features


def _count_high_or_critical_in_window(
    noise_events: list[dict[str, Any]],
    reference_time: datetime,
    period_days: int,
) -> int:
    start = reference_time - timedelta(days=period_days)
    return sum(
        1
        for row in noise_events
        if start <= row["detected_at"] <= reference_time and row["severity"] in {"high", "critical"}
    )


def _classifier_settings_with_backend(
    settings: AISettings,
    backend: str,
) -> ClassifierSettings:
    return replace(settings.classifier, backend=backend)


def _to_serializable_classification(result: EventClassificationResult) -> dict[str, Any]:
    return {
        "event_type": result.event_type,
        "severity": result.severity,
        "severity_score": result.severity_score,
        "confidence": float(result.confidence),
        "is_night": result.is_night,
        "is_meaningful": result.is_meaningful,
        "rule_hits": list(result.rule_hits),
        "classifier_backend": str(result.features.get("classifier_backend", "")),
    }


def _append_shadow_log(
    *,
    settings: AISettings,
    route: str,
    event: EventFeatures,
    primary: EventClassificationResult,
    rule_result: EventClassificationResult,
    lgbm_result: EventClassificationResult,
) -> None:
    payload = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "input": {
            "sensor_id": event.device_id,
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "sound_level": float(event.sound_level),
            "vibration_value": int(event.vibration_value),
            "duration_ms": int(event.duration_ms),
            "accel_delta": float(event.accel_delta),
            "recent_count_10min": int(event.recent_count_10min),
        },
        "settings": {
            "primary_backend": settings.classifier.backend,
            "shadow_mode": settings.classifier.lgbm_shadow_mode,
            "min_confidence": float(settings.classifier.lgbm_min_confidence),
            "model_dir": settings.classifier.lgbm_model_dir,
        },
        "primary": _to_serializable_classification(primary),
        "rule": _to_serializable_classification(rule_result),
        "lightgbm": _to_serializable_classification(lgbm_result),
        "diff": {
            "primary_vs_rule_event_type": primary.event_type != rule_result.event_type,
            "primary_vs_lgbm_event_type": primary.event_type != lgbm_result.event_type,
            "rule_vs_lgbm_event_type": rule_result.event_type != lgbm_result.event_type,
            "rule_vs_lgbm_severity": rule_result.severity != lgbm_result.severity,
            "rule_vs_lgbm_is_meaningful": rule_result.is_meaningful != lgbm_result.is_meaningful,
            "rule_vs_lgbm_confidence_delta": round(
                float(lgbm_result.confidence) - float(rule_result.confidence),
                6,
            ),
        },
    }

    try:
        log_path = Path(settings.classifier.lgbm_shadow_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Shadow logging must never break request processing.
        return


def _classify_event_with_shadow(
    *,
    event: EventFeatures,
    settings: AISettings,
    route: str,
) -> EventClassificationResult:
    runtime: LightGBMRuntime | None = None
    if settings.classifier.backend != "rule" or settings.classifier.lgbm_shadow_mode:
        runtime = LightGBMRuntime(settings=settings)

    primary = classify_event(event=event, settings=settings, runtime=runtime)
    if not settings.classifier.lgbm_shadow_mode:
        return primary

    rule_settings = replace(
        settings,
        classifier=_classifier_settings_with_backend(settings, "rule"),
    )
    lgbm_settings = replace(
        settings,
        classifier=_classifier_settings_with_backend(settings, "lightgbm"),
    )
    rule_result = classify_event(event=event, settings=rule_settings, runtime=runtime)
    lgbm_result = classify_event(event=event, settings=lgbm_settings, runtime=runtime)
    _append_shadow_log(
        settings=settings,
        route=route,
        event=event,
        primary=primary,
        rule_result=rule_result,
        lgbm_result=lgbm_result,
    )
    return primary


def create_dashboard_app() -> FastAPI:
    app = FastAPI(title="KoongLog AI Dashboard API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            schema = get_schema()
        except Exception as exc:  # pragma: no cover - runtime safety
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "status": "ok",
            "noise_table": schema.noise_table,
            "household_table": schema.household_table,
            "mediation_table": schema.mediation_table,
        }

    @app.post("/api/v1/ai/classify-event")
    def classify_event_api(payload: ClassifyEventRequestIn) -> dict[str, Any]:
        settings = get_settings()
        recent_context_events = _normalize_noise_events(payload.recent_meaningful_events_10min)
        sensor_ts, event_features = _build_event_features(
            sensor_id=payload.sensor_id,
            source=payload.source,
            feature_in=payload.event_feature,
            recent_context_events=recent_context_events,
        )
        classification = _classify_event_with_shadow(
            event=event_features,
            settings=settings,
            route="/api/v1/ai/classify-event",
        )
        return {
            "status": "success",
            "classification": {
                "sensor_id": payload.sensor_id,
                "household_id": payload.household_id,
                "event_type": classification.event_type,
                "severity": classification.severity,
                "severity_score": classification.severity_score,
                "confidence": classification.confidence,
                "is_night": classification.is_night,
                "is_meaningful": classification.is_meaningful,
                "timestamp": sensor_ts.isoformat(),
            },
        }

    @app.post("/api/v1/ai/analyze-patterns")
    def analyze_patterns_api(payload: AnalyzePatternsRequestIn) -> dict[str, Any]:
        reference_time = _to_naive_utc(payload.reference_time or datetime.now(timezone.utc))
        noise_events = _normalize_noise_events(payload.noise_events)
        mediation_messages = [
            {"created_at": _to_naive_utc(row.created_at)}
            for row in payload.mediation_messages
        ]
        pattern_obj = analyze_patterns(
            household_id=payload.household_id,
            noise_events=noise_events,
            mediation_messages=mediation_messages,
            analysis_period_days=payload.analysis_period_days,
            reference_time=reference_time,
        )
        pattern_result = {
            "household_id": pattern_obj.household_id,
            "period_days": pattern_obj.analysis_period_days,
            "total_count": pattern_obj.total_events,
            "night_count": pattern_obj.night_events,
            "high_count": _count_high_or_critical_in_window(
                noise_events=noise_events,
                reference_time=reference_time,
                period_days=payload.analysis_period_days,
            ),
            "recent_count_10min": _count_recent_meaningful_events(noise_events, reference_time),
            "repeated_days": pattern_obj.repeated_days,
            "max_events_in_10min": pattern_obj.max_events_in_10min,
            "pattern_label": pattern_obj.pattern_label,
            "needs_mediation": pattern_obj.needs_mediation,
            "needs_escalation": pattern_obj.needs_escalation,
            "post_mediation_recurrence": pattern_obj.post_mediation_recurrence,
            "summary": pattern_obj.summary,
        }
        return {"status": "success", "pattern_result": pattern_result}

    @app.post("/api/v1/ai/analyze")
    @app.post("/api/v1/sensor-readings")
    def analyze_ai(payload: AnalyzeRequestIn) -> dict[str, Any]:
        settings = get_settings()

        # Preferred context: event-only records. Legacy recent_noise_logs is accepted
        # only for already-classified event rows (no raw fallback classification).
        recent_context_events = _normalize_noise_events(payload.recent_meaningful_events_10min)
        if payload.recent_noise_logs:
            recent_context_events.extend(_normalize_noise_events(payload.recent_noise_logs))

        if payload.event_feature is not None:
            sensor_ts, event_features = _build_event_features(
                sensor_id=payload.sensor_id,
                source=payload.source,
                feature_in=payload.event_feature,
                fallback_timestamp=payload.timestamp,
                recent_context_events=recent_context_events,
            )
        else:
            if payload.sound_level is None or payload.vibration_value is None:
                raise HTTPException(
                    status_code=422,
                    detail="event_feature is required (legacy raw fields require sound_level/vibration_value)",
                )
            sensor_ts = _to_naive_utc(payload.timestamp or datetime.now(timezone.utc))
            event_features = EventFeatures(
                device_id=payload.sensor_id,
                source=payload.source,
                sound_level=float(payload.sound_level),
                vibration_value=int(payload.vibration_value),
                duration_ms=int(payload.duration_ms or 0),
                accel_delta=_calc_accel_delta_from_vector(payload.acceleration),
                timestamp=sensor_ts,
                recent_count_10min=_count_recent_meaningful_events(recent_context_events, sensor_ts),
            )

        classification = _classify_event_with_shadow(
            event=event_features,
            settings=settings,
            route="/api/v1/ai/analyze",
        )

        noise_events = _normalize_noise_events(payload.noise_events)
        if not noise_events and payload.recent_noise_logs:
            # Stage-out path: fallback source reduced to event rows only.
            noise_events = _normalize_noise_events(payload.recent_noise_logs)

        high_count = sum(1 for row in noise_events if row["severity"] in {"high", "critical"})
        night_count = sum(
            1
            for row in noise_events
            if row["detected_at"].hour >= 22 or row["detected_at"].hour < 7
        )

        pattern_obj = None
        period_days = payload.analysis_period_days
        if payload.household_id is not None:
            mediation_messages = [
                {"created_at": _to_naive_utc(row.created_at)}
                for row in payload.mediation_messages
            ]
            pattern_obj = analyze_patterns(
                household_id=payload.household_id,
                noise_events=noise_events,
                mediation_messages=mediation_messages,
                analysis_period_days=period_days,
                reference_time=sensor_ts,
            )
            total_count = pattern_obj.total_events
            night_count = pattern_obj.night_events
        else:
            total_count = len(noise_events)

        needs_mediation = (
            (pattern_obj.needs_mediation if pattern_obj is not None else False)
            or classification.severity in {"medium", "high", "critical"}
            or total_count >= 3
            or high_count >= 2
            or night_count >= 2
        )

        pattern_result = {
            "total_count": total_count,
            "night_count": night_count,
            "high_count": high_count,
            "needs_mediation": needs_mediation,
            "period_days": period_days,
            "recent_count_10min": event_features.recent_count_10min,
            "needs_escalation": pattern_obj.needs_escalation if pattern_obj is not None else False,
            "pattern_label": (
                pattern_obj.pattern_label
                if pattern_obj is not None
                else ("recent_noise" if total_count > 0 else "single_event")
            ),
            "summary": (
                pattern_obj.summary
                if pattern_obj is not None
                else f"single_event ({classification.event_type}, {classification.severity})"
            ),
        }

        message_created = False
        ai_result = None
        if payload.generate_message and (
            classification.severity in {"medium", "high", "critical"} or needs_mediation
        ):
            event_count = max(1, total_count)
            manual_report = {
                "noise_type": (payload.noise_type or "").strip(),
                "noise_time_slot": (payload.noise_time_slot or "").strip(),
                "noise_frequency": (payload.noise_frequency or "").strip(),
                "situation_description": (payload.situation_description or "").strip(),
            }
            manual_report = {k: v for k, v in manual_report.items() if v}
            event_context = {
                "event_type": classification.event_type,
                "severity": classification.severity,
                "time_range": sensor_ts.strftime("%H:%M"),
                "event_count": event_count,
                "pattern_summary": pattern_result["summary"],
                "target_unit": payload.target_unit,
                "manual_report": manual_report,
            }
            message = generate_mediation_message(
                event_context=event_context,
                pattern_result=pattern_obj,
                settings=settings,
            )
            message_created = True
            ai_result = {
                "ai_message": message.resident_message,
                "event_summary": message.event_summary,
                "resident_message": message.resident_message,
                "admin_summary": message.admin_summary,
                "recommended_action": message.recommended_action,
                "tone_check": message.tone_check,
                "generation_method": message.generation_method,
            }

        return {
            "status": "success",
            "noise_log": {
                "id": None,
                "sensor_id": payload.sensor_id,
                "household_id": payload.household_id,
                "event_type": classification.event_type,
                "severity": classification.severity,
                "severity_score": classification.severity_score,
                "confidence": classification.confidence,
                "is_night": classification.is_night,
                "is_meaningful": classification.is_meaningful,
                "timestamp": sensor_ts.isoformat(),
            },
            "pattern_result": pattern_result,
            "message_created": message_created,
            "ai_result": ai_result,
        }

    @app.get("/api/v1/noise/distribution")
    def noise_distribution(
        days: int = Query(default=get_backend_settings().default_period_days, ge=1, le=90),
        household_id: int | None = Query(default=None),
    ) -> dict[str, Any]:
        rows = _query_noise_logs(days=days, household_id=household_id)
        event_counter = Counter(row["event_type"] for row in rows)
        severity_counter = Counter(row["severity"] for row in rows)
        return {
            "period_days": days,
            "household_id": household_id,
            "total_events": len(rows),
            "by_event_type": [
                {"event_type": key, "count": value}
                for key, value in sorted(event_counter.items(), key=lambda x: (-x[1], x[0]))
            ],
            "by_severity": [
                {"severity": key, "count": value}
                for key, value in sorted(severity_counter.items(), key=lambda x: (-x[1], x[0]))
            ],
        }

    @app.get("/api/v1/noise/distribution/export")
    def noise_distribution_export(
        days: int = Query(default=get_backend_settings().default_period_days, ge=1, le=90),
        household_id: int | None = Query(default=None),
    ) -> StreamingResponse:
        rows = _query_noise_logs(days=days, household_id=household_id)
        combo_counter = Counter((row["event_type"], row["severity"]) for row in rows)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["event_type", "severity", "count"])
        for (event_type, severity), count in sorted(
            combo_counter.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1]),
        ):
            writer.writerow([event_type, severity, count])
        output.seek(0)

        filename = f"noise_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(output, media_type="text/csv; charset=utf-8", headers=headers)

    @app.get("/api/v1/dashboard/households")
    def dashboard_households(
        days: int = Query(default=get_backend_settings().default_period_days, ge=1, le=90),
    ) -> dict[str, Any]:
        rows = _query_noise_logs(days=days)
        labels = _query_household_labels()
        mediations = _query_mediations(limit=1000)
        items = _build_household_status(rows, labels, mediations, days=days)
        return {"period_days": days, "count": len(items), "items": items}

    @app.get("/api/v1/dashboard/urgent")
    def dashboard_urgent(
        days: int = Query(default=get_backend_settings().default_period_days, ge=1, le=90),
    ) -> dict[str, Any]:
        rows = _query_noise_logs(days=days)
        labels = _query_household_labels()
        mediations = _query_mediations(limit=1000)
        items = _build_household_status(rows, labels, mediations, days=days)
        urgent_items = [row for row in items if row["status"] == "urgent"]
        return {"period_days": days, "count": len(urgent_items), "items": urgent_items}

    @app.get("/api/v1/dashboard/today-events")
    def dashboard_today_events() -> dict[str, Any]:
        tz = _get_local_tz()
        start_of_day = datetime.combine(date.today(), time.min).replace(tzinfo=tz)
        rows = _query_noise_logs(since=start_of_day)
        combo_counter = Counter((row["event_type"], row["severity"]) for row in rows)
        return {
            "date": date.today().isoformat(),
            "total_events": len(rows),
            "combinations": [
                {"event_type": event_type, "severity": severity, "count": count}
                for (event_type, severity), count in sorted(
                    combo_counter.items(),
                    key=lambda x: (-x[1], x[0][0], x[0][1]),
                )
            ],
        }

    @app.get("/api/v1/dashboard/completed")
    def dashboard_completed(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = _query_mediations(limit=limit)
        completed = []
        for row in rows:
            status = str(row.get("status") or "").strip().lower()
            if status in COMPLETED_STATUSES:
                completed.append(
                    {
                        "id": row.get("id"),
                        "household_id": row.get("household_id"),
                        "target_unit": row.get("target_unit"),
                        "recommended_action": row.get("recommended_action"),
                        "status": row.get("status"),
                        "created_at": row.get("created_at"),
                    }
                )
        return {"count": len(completed), "items": completed}

    @app.get("/api/v1/dashboard/hourly")
    def dashboard_hourly(
        days: int = Query(default=get_backend_settings().default_period_days, ge=1, le=90),
    ) -> dict[str, Any]:
        rows = _query_noise_logs(days=days)
        hourly = Counter()
        for row in rows:
            dt = row["detected_at"]
            hourly[dt.hour] += 1
        series = [
            {"hour": f"{hour:02d}:00", "count": hourly.get(hour, 0)}
            for hour in range(24)
        ]
        return {"period_days": days, "series": series}

    return app


app = create_dashboard_app()
