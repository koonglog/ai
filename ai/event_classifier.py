from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .config import AISettings, get_settings
from .lgbm_runtime import LGBMInferenceResult, LightGBMRuntime, get_lgbm_runtime
from .ml_features import FEATURE_SPEC_VERSION
from .schemas import EventClassificationResult, EventFeatures, EventType, Severity
from .time_utils import as_kst
from .vibration import (
    REPEATED_IMPACT_CAUTION_COUNT,
    REPEATED_IMPACT_WARNING_COUNT,
    VIBRATION_LEVEL_CATEGORY_CODES,
    build_vibration_risk_profile,
)


def is_night(dt: datetime, settings: AISettings) -> bool:
    """Return True if timestamp is inside configured night-time hours."""
    start = settings.nighttime.start_hour
    end = settings.nighttime.end_hour
    hour = as_kst(dt).hour
    return hour >= start or hour < end


def _normalize_recent_count(raw_count: int) -> int:
    count = int(raw_count)
    if count < 0:
        raise ValueError("recent_count_10min must be >= 0")
    return count


def _normalize_impact_count(raw_count: int) -> int:
    count = int(raw_count)
    if count < 0:
        raise ValueError("impact_count_in_window must be >= 0")
    return count


def is_meaningful_event(event_type: str, severity: str) -> bool:
    """Return True only for persistence-target event categories."""
    _ = severity  # kept for backward-compatible function signature
    return event_type in {
        EventType.IMPACT_NOISE.value,
        EventType.REPEATED_VIBRATION.value,
    }


def _is_meaningful_noise_risk(
    *,
    event_type: str,
    severity: str,
    noise_risk_level: str,
    repeated_impact_flag: bool,
) -> bool:
    return (
        is_meaningful_event(event_type=event_type, severity=severity)
        or repeated_impact_flag
        or noise_risk_level == "high"
    )


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
    impact_count_in_window = _normalize_impact_count(event.impact_count_in_window)
    profile = build_vibration_risk_profile(
        vibration_value=event.vibration_value,
        timestamp=event.timestamp,
        settings=settings,
        impact_count_in_window=impact_count_in_window,
    )
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

    # Vibration score uses estimated physical units, not raw ADC thresholds.
    vibration_category_score = VIBRATION_LEVEL_CATEGORY_CODES.get(
        profile.vibration_level_category,
        0,
    )
    score += vibration_category_score

    # duration score
    if event.duration_ms >= thresholds.duration_long_ms:
        score += 2
    elif event.duration_ms >= thresholds.duration_medium_ms:
        score += 1

    # Event-context bonuses (never raw sample count).
    if night and event_type != EventType.BACKGROUND_NOISE.value:
        score += 1
    if impact_count_in_window >= REPEATED_IMPACT_WARNING_COUNT:
        score += 2
    elif impact_count_in_window >= REPEATED_IMPACT_CAUTION_COUNT:
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
    features["prediction_path"] = (
        runtime_result.prediction_path
        if runtime_result is not None and runtime_result.prediction_path is not None
        else ("rule_fallback" if "fallback" in backend else backend)
    )
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
        if runtime_result.noise_risk_probability is not None:
            features["lgbm_noise_risk_probability"] = round(
                float(runtime_result.noise_risk_probability),
                6,
            )
        if runtime_result.noise_risk_probabilities is not None:
            features["lgbm_noise_risk_probabilities"] = {
                key: round(float(value), 6)
                for key, value in runtime_result.noise_risk_probabilities.items()
            }
        if runtime_result.model_version is not None:
            features["model_version"] = runtime_result.model_version
        if runtime_result.feature_spec_version is not None:
            features["feature_spec_version"] = runtime_result.feature_spec_version
        if runtime_result.model_path is not None:
            features["model_artifact_path"] = runtime_result.model_path
        if runtime_result.metadata_path is not None:
            features["metadata_path"] = runtime_result.metadata_path
        if runtime_result.label_map_path is not None:
            features["label_map_path"] = runtime_result.label_map_path

    return EventClassificationResult(
        event_type=result.event_type,
        severity=result.severity,
        severity_score=result.severity_score,
        confidence=result.confidence,
        is_night=result.is_night,
        is_meaningful=result.is_meaningful,
        noise_risk_level=result.noise_risk_level,
        vibration_raw=result.vibration_raw,
        vibration_acc_mps2=result.vibration_acc_mps2,
        vibration_dbv=result.vibration_dbv,
        reason=result.reason,
        time_period=result.time_period,
        impact_count=result.impact_count,
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
    impact_count_in_window = _normalize_impact_count(event.impact_count_in_window)
    profile = build_vibration_risk_profile(
        vibration_value=event.vibration_value,
        timestamp=event.timestamp,
        settings=cfg,
        impact_count_in_window=impact_count_in_window,
    )
    vibration_category = profile.vibration_level_category

    event_type = EventType.UNKNOWN.value
    severity = Severity.LOW.value
    score = 0
    confidence = 0.5
    rule_hits: list[str] = []

    # 1) background noise
    if (
        event.sound_level < airborne_leq_threshold
        and vibration_category == "normal"
        and profile.impact_count < REPEATED_IMPACT_CAUTION_COUNT
        and event.duration_ms < thresholds.duration_short_ms
    ):
        event_type = EventType.BACKGROUND_NOISE.value
        confidence = 0.95
        rule_hits.append("background_rule")

    # 2) repeated vibration in a short window.
    elif (
        profile.impact_count >= REPEATED_IMPACT_CAUTION_COUNT
        and vibration_category != "normal"
    ):
        event_type = EventType.REPEATED_VIBRATION.value
        confidence = 0.82
        rule_hits.append("repeated_vibration_window_rule")

    # 3) impact noise
    elif (
        event.sound_level >= impact_lmax_threshold
        and vibration_category == "impact_risk"
        and accel_delta >= 0.12
    ):
        event_type = EventType.IMPACT_NOISE.value
        confidence = 0.88
        rule_hits.extend(["impact_db_rule", "impact_acc_mps2_rule", "impact_accel_rule"])

    # 4) daily noise
    elif event.sound_level >= airborne_leq_threshold:
        event_type = EventType.DAILY_NOISE.value
        confidence = 0.74
        rule_hits.append("daily_noise_rule")

    # 5) single vibration risk without enough repetition.
    elif vibration_category in {"warning", "impact_risk"}:
        event_type = EventType.UNKNOWN.value
        confidence = 0.62
        rule_hits.append("single_vibration_risk_observed")

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
    is_meaningful = _is_meaningful_noise_risk(
        event_type=event_type,
        severity=severity,
        noise_risk_level=profile.noise_risk_level,
        repeated_impact_flag=profile.repeated_impact_flag,
    )

    return EventClassificationResult(
        event_type=event_type,
        severity=severity,
        severity_score=score,
        confidence=confidence,
        is_night=night,
        is_meaningful=is_meaningful,
        noise_risk_level=profile.noise_risk_level,
        vibration_raw=profile.vibration_raw,
        vibration_acc_mps2=round(profile.vibration_acc_mps2, 6),
        vibration_dbv=round(profile.vibration_dbv, 2),
        reason=profile.reason,
        time_period=profile.time_period,
        impact_count=profile.impact_count,
        rule_hits=rule_hits,
        features={
            "sound_level": event.sound_level,
            "resolved_timestamp": as_kst(event.timestamp).isoformat(),
            "timestamp_source": event.timestamp_source,
            "timestamp_conflict": event.timestamp_conflict,
            "prediction_path": "rule",
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "vibration_raw": profile.vibration_raw,
            "vibration_acc_mps2": round(profile.vibration_acc_mps2, 6),
            "vibration_dbv": round(profile.vibration_dbv, 2),
            "is_daytime": profile.is_daytime,
            "is_nighttime": profile.is_nighttime,
            "vibration_level_category": profile.vibration_level_category,
            "impact_count_in_window": profile.impact_count,
            "repeated_impact_flag": profile.repeated_impact_flag,
            "noise_risk_level": profile.noise_risk_level,
            "reason": profile.reason,
            "time_period": profile.time_period,
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

    if runtime_result.noise_risk_level is not None:
        night = is_night(event.timestamp, cfg)
        impact_count_in_window = _normalize_impact_count(event.impact_count_in_window)
        profile = build_vibration_risk_profile(
            vibration_value=event.vibration_value,
            timestamp=event.timestamp,
            settings=cfg,
            impact_count_in_window=impact_count_in_window,
        )
        model_level = runtime_result.noise_risk_level
        reason = (
            "LightGBM internal alert model predicted "
            f"{model_level} using feature_spec "
            f"{runtime_result.feature_spec_version or FEATURE_SPEC_VERSION}; "
            f"{profile.reason}"
        )
        is_meaningful = _is_meaningful_noise_risk(
            event_type=rule_result.event_type,
            severity=rule_result.severity,
            noise_risk_level=model_level,
            repeated_impact_flag=profile.repeated_impact_flag,
        )
        confidence = float(
            runtime_result.noise_risk_probability
            if runtime_result.noise_risk_probability is not None
            else 0.5
        )
        features = dict(rule_result.features)
        features.update(
            {
                "noise_risk_level": model_level,
                "reason": reason,
                "time_period": profile.time_period,
                "impact_count_in_window": profile.impact_count,
                "repeated_impact_flag": profile.repeated_impact_flag,
                "vibration_level_category": profile.vibration_level_category,
                "is_daytime": profile.is_daytime,
                "is_nighttime": profile.is_nighttime,
                "vibration_raw": profile.vibration_raw,
                "vibration_acc_mps2": round(profile.vibration_acc_mps2, 6),
                "vibration_dbv": round(profile.vibration_dbv, 2),
            }
        )
        model_result = EventClassificationResult(
            event_type=rule_result.event_type,
            severity=rule_result.severity,
            severity_score=rule_result.severity_score,
            confidence=confidence,
            is_night=night,
            is_meaningful=is_meaningful,
            noise_risk_level=model_level,
            vibration_raw=profile.vibration_raw,
            vibration_acc_mps2=round(profile.vibration_acc_mps2, 6),
            vibration_dbv=round(profile.vibration_dbv, 2),
            reason=reason,
            time_period=profile.time_period,
            impact_count=profile.impact_count,
            rule_hits=["lgbm_noise_risk_model"],
            features=features,
        )
        return _append_runtime_metadata(
            model_result,
            backend="lgbm_noise_risk",
            runtime_result=runtime_result,
        )

    # Legacy event-type LightGBM remains conservative for direct-impact cases.
    if rule_result.event_type == EventType.IMPACT_NOISE.value:
        return _append_runtime_metadata(
            rule_result,
            backend="rule_impact_backup",
            runtime_result=runtime_result,
            extra_rule_hits=["impact_rule_backup"],
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
    profile = build_vibration_risk_profile(
        vibration_value=event.vibration_value,
        timestamp=event.timestamp,
        settings=cfg,
        impact_count_in_window=event.impact_count_in_window,
    )
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
    is_meaningful = _is_meaningful_noise_risk(
        event_type=event_type,
        severity=severity,
        noise_risk_level=profile.noise_risk_level,
        repeated_impact_flag=profile.repeated_impact_flag,
    )
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
        noise_risk_level=profile.noise_risk_level,
        vibration_raw=profile.vibration_raw,
        vibration_acc_mps2=round(profile.vibration_acc_mps2, 6),
        vibration_dbv=round(profile.vibration_dbv, 2),
        reason=profile.reason,
        time_period=profile.time_period,
        impact_count=profile.impact_count,
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
