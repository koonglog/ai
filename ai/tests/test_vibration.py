from __future__ import annotations

import math
from datetime import datetime

from ai.vibration import (
    acceleration_to_dbv,
    build_vibration_risk_profile,
    vibration_raw_to_acc_mps2,
)


def test_sen0209_raw_conversion_formula() -> None:
    assert vibration_raw_to_acc_mps2(8) == 0.005
    assert vibration_raw_to_acc_mps2(1007) == 0.45
    assert vibration_raw_to_acc_mps2(10.8) == (
        0.005 + (10.8 - 8.0) * ((0.45 - 0.005) / (1007.0 - 8.0))
    )
    assert acceleration_to_dbv(0.005) == 20 * math.log10(0.005 / 1e-5)


def test_fractional_raw_is_not_int_truncated_at_category_boundary() -> None:
    day = datetime.fromisoformat("2026-05-13T14:00:00+09:00")

    float_profile = build_vibration_risk_profile(
        vibration_value=19.3,
        timestamp=day,
        impact_count_in_window=0,
    )
    truncated_profile = build_vibration_risk_profile(
        vibration_value=19.0,
        timestamp=day,
        impact_count_in_window=0,
    )

    assert float_profile.vibration_raw == 19.3
    assert float_profile.vibration_level_category == "caution"
    assert truncated_profile.vibration_level_category == "normal"


def test_day_and_night_vibration_categories() -> None:
    day = datetime.fromisoformat("2026-05-13T14:00:00+09:00")
    night = datetime.fromisoformat("2026-05-13T23:00:00+09:00")

    day_profile = build_vibration_risk_profile(
        vibration_value=13,
        timestamp=day,
        impact_count_in_window=0,
    )
    night_profile = build_vibration_risk_profile(
        vibration_value=13,
        timestamp=night,
        impact_count_in_window=0,
    )

    assert day_profile.time_period == "daytime"
    assert night_profile.time_period == "nighttime"
    assert day_profile.vibration_level_category == "normal"
    assert night_profile.vibration_level_category == "caution"
