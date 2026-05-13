from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .config import AISettings, get_settings
from .lgbm_runtime import LGBMInferenceResult, LightGBMRuntime, get_lgbm_runtime
from .schemas import EventClassificationResult, EventFeatures, EventType, Severity


def is_night(dt: datetime, settings: AISettings) -> bool:
    """Return True if timestamp is inside configured night-time hours."""
    start = settings.nighttime.start_hour
    end = settings.nighttime.end_hour
    hour = dt.hour
    return hour >= start or hour < end


def _normalize_recent_count(raw_count: int) -> int:
    count = int(raw_count)
    if count < 0:
        raise ValueError("recent_count_10min must be >= 0")
    return count


def is_meaningful_event(event_type: str, severity: str) -> bool:
    """Return True only for persistence-target event categories."""
    _ = severity  # kept for backward-compatible function signature
    return event_type in {
        EventType.IMPACT_NOISE.value,
        EventType.REPEATED_VIBRATION.value,
    }


def _score_severity(
    event: EventFeatures,
    night: bool,
    event_type: str,
    settings: AISettings,
) -> tuple[str, int]:
    thresholds = settings.thresholds
    impact_lmax_threshold = thresholds.impact_lmax(night)
    airborne_leq_threshold = thresholds.airborne_leq(night)
    recent_count_10min = _normalize_recent_count(event.recent_count_10min)
    score = 0

    # dB score (relative to legal day/night thresholds)
    if event.sound_level >= impact_lmax_threshold + 8:
        score += 4
    elif event.sound_level >= impact_lmax_threshold + 2:
        score += 3
    elif event.sound_level >= impact_lmax_threshold:
        score += 2
    elif event.sound_level >= airborne_leq_threshold:
        score += 1

    # vibration score
    if event.vibration_value >= 800:
        score += 3
    elif event.vibration_value >= thresholds.vibration_high:
        score += 2
    elif event.vibration_value >= thresholds.vibration_mid:
        score += 1

    # duration score
    if event.duration_ms >= thresholds.duration_long_ms:
        score += 2
    elif event.duration_ms >= thresholds.duration_medium_ms:
        score += 1

    # Event-context bonuses (never raw sample count).
    if night and event_type != EventType.BACKGROUND_NOISE.value:
        score += 1
    if recent_count_10min >= 5:
        score += 2
    elif recent_count_10min >= 3:
        score += 1
    if event_type == EventType.REPEATED_VIBRATION.value:
        score += 1

    if score >= 9:
        return Severity.CRITICAL.value, score
    if score >= 6:
        return Severity.HIGH.value, score
    if score >= 3:
        return Severity.MEDIUM.value, score
    return Severity.LOW.value, score


class RuntimePredictor(Protocol):
    def predict_event(
        self,
        event: EventFeatures,
        *,
        min_confidence: float | None = None,
    ) -> LGBMInferenceResult: ...


def _append_runtime_metadata(
    result: EventClassificationResult,
    *,
    backend: str,
    runtime_result: LGBMInferenceResult | None = None,
    extra_rule_hits: list[str] | None = None,
) -> EventClassificationResult:
    rule_hits = list(result.rule_hits)
    if extra_rule_hits:
        rule_hits.extend(extra_rule_hits)

    features = dict(result.features)
    features["classifier_backend"] = backend
    if runtime_result is not None:
        if runtime_result.reason is not None:
            features["lgbm_reason"] = runtime_result.reason
        if runtime_result.meaningful_probability is not None:
            features["lgbm_meaningful_probability"] = round(
                float(runtime_result.meaningful_probability),
                6,
            )
        if runtime_result.event_type_probability is not None:
            features["lgbm_event_type_probability"] = round(
                float(runtime_result.event_type_probability),
                6,
            )
        if runtime_result.event_type_probabilities is not None:
            features["lgbm_event_type_probabilities"] = {
                key: round(float(value), 6)
                for key, value in runtime_result.event_type_probabilities.items()
            }

    return EventClassificationResult(
        event_type=result.event_type,
        severity=result.severity,
        severity_score=result.severity_score,
        confidence=result.confidence,
        is_night=result.is_night,
        is_meaningful=result.is_meaningful,
        rule_hits=rule_hits,
        features=features,
    )


def _classify_event_rule(
    event: EventFeatures,
    cfg: AISettings,
) -> EventClassificationResult:
    thresholds = cfg.thresholds
    night = is_night(event.timestamp, cfg)
    impact_leq_threshold = thresholds.impact_leq(night)
    impact_lmax_threshold = thresholds.impact_lmax(night)
    airborne_leq_threshold = thresholds.airborne_leq(night)
    accel_delta = float(event.accel_delta)
    recent_count_10min = _normalize_recent_count(event.recent_count_10min)

    event_type = EventType.UNKNOWN.value
    severity = Severity.LOW.value
    score = 0
    confidence = 0.5
    rule_hits: list[str] = []

    # 1) background noise
    if event.sound_level < impact_leq_threshold or (
        event.sound_level < airborne_leq_threshold
        and event.vibration_value < thresholds.vibration_low
        and event.duration_ms < thresholds.duration_short_ms
    ):
        event_type = EventType.BACKGROUND_NOISE.value
        confidence = 0.95
        rule_hits.append("background_rule")

    # 2) repeated vibration (recent_count_10min: meaningful events only)
    elif event.vibration_value >= thresholds.vibration_mid and recent_count_10min >= 3:
        event_type = EventType.REPEATED_VIBRATION.value
        confidence = 0.82
        rule_hits.append("repeated_vibration_rule")

    # 3) impact noise
    elif (
        event.sound_level >= impact_lmax_threshold
        and event.vibration_value >= thresholds.vibration_high
        and accel_delta >= 0.12
    ):
        event_type = EventType.IMPACT_NOISE.value
        confidence = 0.87
        rule_hits.extend(["impact_db_rule", "impact_vibration_rule", "impact_accel_rule"])

    # 4) daily noise
    elif event.sound_level >= airborne_leq_threshold:
        event_type = EventType.DAILY_NOISE.value
        confidence = 0.74
        rule_hits.append("daily_noise_rule")

    else:
        event_type = EventType.UNKNOWN.value
        confidence = 0.55
        rule_hits.append("unknown_rule")

    if event_type == EventType.BACKGROUND_NOISE.value:
        severity = Severity.LOW.value
        score = 0
    else:
        severity, score = _score_severity(
            event=event,
            night=night,
            event_type=event_type,
            settings=cfg,
        )
    is_meaningful = is_meaningful_event(event_type=event_type, severity=severity)

    return EventClassificationResult(
        event_type=event_type,
        severity=severity,
        severity_score=score,
        confidence=confidence,
        is_night=night,
        is_meaningful=is_meaningful,
        rule_hits=rule_hits,
        features={
            "sound_level": event.sound_level,
            "vibration_value": event.vibration_value,
            "duration_ms": event.duration_ms,
            "accel_delta": round(accel_delta, 4),
            "recent_10min_meaningful_count": recent_count_10min,
            "impact_leq_threshold": impact_leq_threshold,
            "impact_lmax_threshold": impact_lmax_threshold,
            "airborne_leq_threshold": airborne_leq_threshold,
        },
    )


def _classify_event_lightgbm_or_fallback(
    *,
    event: EventFeatures,
    cfg: AISettings,
    rule_result: EventClassificationResult,
    runtime: RuntimePredictor,
) -> EventClassificationResult:
    # Keep rule-based impact detection as a hard backup while impact class
    # data remains extremely sparse in exports.
    if rule_result.event_type == EventType.IMPACT_NOISE.value:
        return _append_runtime_metadata(
            rule_result,
            backend="rule_impact_backup",
            extra_rule_hits=["impact_rule_backup"],
        )

    runtime_result = runtime.predict_event(
        event,
        min_confidence=cfg.classifier.lgbm_min_confidence,
    )
    if runtime_result.should_fallback:
        return _append_runtime_metadata(
            rule_result,
            backend="rule_fallback",
            runtime_result=runtime_result,
            extra_rule_hits=["lgbm_fallback"],
        )

    if not runtime_result.event_type:
        return _append_runtime_metadata(
            rule_result,
            backend="rule_fallback",
            runtime_result=runtime_result,
            extra_rule_hits=["lgbm_fallback_no_event_type"],
        )

    event_type = runtime_result.event_type
    night = is_night(event.timestamp, cfg)
    if event_type == EventType.BACKGROUND_NOISE.value:
        severity = Severity.LOW.value
        score = 0
    else:
        severity, score = _score_severity(
            event=event,
            night=night,
            event_type=event_type,
            settings=cfg,
        )
    is_meaningful = is_meaningful_event(event_type=event_type, severity=severity)
    confidence = float(
        runtime_result.event_type_probability
        if runtime_result.event_type_probability is not None
        else (runtime_result.meaningful_probability or 0.5)
    )

    model_result = EventClassificationResult(
        event_type=event_type,
        severity=severity,
        severity_score=score,
        confidence=confidence,
        is_night=night,
        is_meaningful=is_meaningful,
        rule_hits=["lgbm_model"],
        features=dict(rule_result.features),
    )
    return _append_runtime_metadata(
        model_result,
        backend="lgbm",
        runtime_result=runtime_result,
    )


def classify_event(
    event: EventFeatures,
    settings: AISettings | None = None,
    runtime: RuntimePredictor | None = None,
) -> EventClassificationResult:
    """
    Classify one event feature into event type and severity.

    Contract:
    - `event.recent_count_10min` must be "count of meaningful events in the
      last 10 minutes".
    - Raw log/sample count must not be passed.

    Backend policy:
    - rule: always use rule-based classifier
    - hybrid/lightgbm: try LightGBM first, fallback to rule on runtime/model issues
      or low confidence
    """
    cfg = settings or get_settings()
    rule_result = _classify_event_rule(event=event, cfg=cfg)
    backend = cfg.classifier.backend

    if backend == "rule":
        return _append_runtime_metadata(rule_result, backend="rule")

    runtime_obj: RuntimePredictor
    if runtime is not None:
        runtime_obj = runtime
    elif settings is None:
        runtime_obj = get_lgbm_runtime()
    else:
        runtime_obj = LightGBMRuntime(settings=cfg)

    model_or_fallback = _classify_event_lightgbm_or_fallback(
        event=event,
        cfg=cfg,
        rule_result=rule_result,
        runtime=runtime_obj,
    )
    if backend == "hybrid":
        return _append_runtime_metadata(
            model_or_fallback,
            backend=f"hybrid/{model_or_fallback.features.get('classifier_backend', 'unknown')}",
        )
    return model_or_fallback
