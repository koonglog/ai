from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import AISettings, get_settings
from .time_utils import as_kst

BASELINE_RAW = 8.0
REFERENCE_RAW = 1007.0
MIN_ACCELERATION_MPS2 = 0.005
MAX_REFERENCE_ACCELERATION_MPS2 = 0.45
DBV_REFERENCE_ACCELERATION = 1e-5
IMPACT_WINDOW_SECONDS = 10
REPEATED_IMPACT_CAUTION_COUNT = 3
REPEATED_IMPACT_WARNING_COUNT = 6

VIBRATION_LEVEL_CATEGORY_CODES: dict[str, int] = {
    "normal": 0,
    "caution": 1,
    "warning": 2,
    "impact_risk": 3,
}

NOISE_RISK_LEVEL_CODES: dict[str, int] = {
    "normal": 0,
    "caution": 1,
    "warning": 2,
    "high": 3,
}


@dataclass(frozen=True, slots=True)
class VibrationRiskProfile:
    vibration_raw: float
    vibration_acc_mps2: float
    vibration_dbv: float
    is_daytime: bool
    is_nighttime: bool
    time_period: str
    vibration_level_category: str
    vibration_level_category_code: int
    impact_count: int
    repeated_impact_flag: bool
    noise_risk_level: str
    reason: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def vibration_raw_to_acc_mps2(vibration_value: Any) -> float:
    """Convert SEN0209 ADC count to estimated calibrated RMS acceleration."""
    raw = _safe_float(vibration_value, 0.0)
    scale = (
        (MAX_REFERENCE_ACCELERATION_MPS2 - MIN_ACCELERATION_MPS2)
        / (REFERENCE_RAW - BASELINE_RAW)
    )
    return max(
        MIN_ACCELERATION_MPS2,
        MIN_ACCELERATION_MPS2 + (raw - BASELINE_RAW) * scale,
    )


def acceleration_to_dbv(acc_mps2: float) -> float:
    return 20.0 * math.log10(max(acc_mps2, MIN_ACCELERATION_MPS2) / DBV_REFERENCE_ACCELERATION)


def is_nighttime(timestamp: datetime, settings: AISettings | None = None) -> bool:
    cfg = settings or get_settings()
    start = cfg.nighttime.start_hour
    end = cfg.nighttime.end_hour
    hour = as_kst(timestamp).hour
    return hour >= start or hour < end


def classify_vibration_level(acc_mps2: float, *, nighttime: bool) -> str:
    if nighttime:
        if acc_mps2 < 0.007:
            return "normal"
        if acc_mps2 < 0.02:
            return "caution"
        if acc_mps2 < 0.15:
            return "warning"
        return "impact_risk"

    if acc_mps2 < 0.01:
        return "normal"
    if acc_mps2 < 0.02:
        return "caution"
    if acc_mps2 < 0.15:
        return "warning"
    return "impact_risk"


def category_is_caution_or_higher(category: str) -> bool:
    return VIBRATION_LEVEL_CATEGORY_CODES.get(category, 0) >= VIBRATION_LEVEL_CATEGORY_CODES["caution"]


def risk_level_from_category_and_count(category: str, impact_count: int) -> str:
    base = {
        "normal": "normal",
        "caution": "caution",
        "warning": "warning",
        "impact_risk": "high",
    }.get(category, "normal")

    if impact_count >= REPEATED_IMPACT_WARNING_COUNT:
        count_level = "warning"
    elif impact_count >= REPEATED_IMPACT_CAUTION_COUNT:
        count_level = "caution"
    else:
        count_level = "normal"

    return max(
        (base, count_level),
        key=lambda level: NOISE_RISK_LEVEL_CODES.get(level, 0),
    )


def build_vibration_risk_profile(
    *,
    vibration_value: Any,
    timestamp: datetime,
    settings: AISettings | None = None,
    impact_count_in_window: int = 0,
) -> VibrationRiskProfile:
    cfg = settings or get_settings()
    raw = max(0.0, _safe_float(vibration_value, 0.0))
    acc = vibration_raw_to_acc_mps2(raw)
    dbv = acceleration_to_dbv(acc)
    timestamp = as_kst(timestamp)
    night = is_nighttime(timestamp, cfg)
    category = classify_vibration_level(acc, nighttime=night)
    impact_count = max(0, int(impact_count_in_window))
    if category_is_caution_or_higher(category):
        impact_count = max(1, impact_count)
    repeated = impact_count >= REPEATED_IMPACT_CAUTION_COUNT
    risk_level = risk_level_from_category_and_count(category, impact_count)
    time_period = "nighttime" if night else "daytime"

    reason = (
        f"SEN0209 raw ADC {raw} -> estimated {acc:.6f} m/s^2, "
        f"{dbv:.2f} dB(V); {time_period} category={category}; "
        f"caution_or_higher_count_{IMPACT_WINDOW_SECONDS}s={impact_count}"
    )
    if impact_count >= REPEATED_IMPACT_WARNING_COUNT:
        reason += " (repeated impact warning threshold met)"
    elif impact_count >= REPEATED_IMPACT_CAUTION_COUNT:
        reason += " (repeated impact caution threshold met)"

    return VibrationRiskProfile(
        vibration_raw=raw,
        vibration_acc_mps2=acc,
        vibration_dbv=dbv,
        is_daytime=not night,
        is_nighttime=night,
        time_period=time_period,
        vibration_level_category=category,
        vibration_level_category_code=VIBRATION_LEVEL_CATEGORY_CODES[category],
        impact_count=impact_count,
        repeated_impact_flag=repeated,
        noise_risk_level=risk_level,
        reason=reason,
    )
