from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class ThresholdSettings:
    """Rule-based threshold settings for event classification."""

    db_background: float = 40.0
    db_daily: float = 45.0
    db_impact: float = 50.0
    vibration_low: int = 150
    vibration_mid: int = 350
    vibration_high: int = 600
    duration_short_ms: int = 3000
    duration_medium_ms: int = 8000
    duration_long_ms: int = 15000


@dataclass(frozen=True)
class NightTimeSettings:
    """Night-time settings (local timezone-based hour window)."""

    start_hour: int = 22
    end_hour: int = 7

    def validate(self) -> None:
        if not (0 <= self.start_hour <= 23 and 0 <= self.end_hour <= 23):
            raise ValueError("NightTimeSettings hours must be in 0..23")


@dataclass(frozen=True)
class PatternSettings:
    """Pattern analysis configuration."""

    analysis_days_default: int = 7
    analysis_days_extended: int = 30
    cluster_window_minutes: int = 10
    repeated_days_for_warning: int = 3
    repeated_days_for_escalation: int = 4


@dataclass(frozen=True)
class OpenAISettings:
    """OpenAI call policy and runtime settings."""

    enabled: bool = True
    api_key: str | None = None
    model: str = "gpt-5-mini"
    timeout_seconds: float = 8.0
    max_retries: int = 1
    max_output_tokens: int = 500
    llm_min_severity: str = "medium"
    reasoning_effort: str = "minimal"

    @property
    def can_call(self) -> bool:
        return self.enabled and bool(self.api_key)


@dataclass(frozen=True)
class AISettings:
    """Top-level AI settings container."""

    thresholds: ThresholdSettings
    nighttime: NightTimeSettings
    pattern: PatternSettings
    openai: OpenAISettings

    @classmethod
    def from_env(cls) -> "AISettings":
        thresholds = ThresholdSettings(
            db_background=_to_float(os.getenv("AI_DB_BACKGROUND"), 40.0),
            db_daily=_to_float(os.getenv("AI_DB_DAILY"), 45.0),
            db_impact=_to_float(os.getenv("AI_DB_IMPACT"), 50.0),
            vibration_low=_to_int(os.getenv("AI_VIBRATION_LOW"), 150),
            vibration_mid=_to_int(os.getenv("AI_VIBRATION_MID"), 350),
            vibration_high=_to_int(os.getenv("AI_VIBRATION_HIGH"), 600),
            duration_short_ms=_to_int(os.getenv("AI_DURATION_SHORT_MS"), 3000),
            duration_medium_ms=_to_int(os.getenv("AI_DURATION_MEDIUM_MS"), 8000),
            duration_long_ms=_to_int(os.getenv("AI_DURATION_LONG_MS"), 15000),
        )

        nighttime = NightTimeSettings(
            start_hour=_to_int(os.getenv("AI_NIGHT_START_HOUR"), 22),
            end_hour=_to_int(os.getenv("AI_NIGHT_END_HOUR"), 7),
        )
        nighttime.validate()

        pattern = PatternSettings(
            analysis_days_default=_to_int(os.getenv("AI_ANALYSIS_DAYS_DEFAULT"), 7),
            analysis_days_extended=_to_int(os.getenv("AI_ANALYSIS_DAYS_EXTENDED"), 30),
            cluster_window_minutes=_to_int(os.getenv("AI_CLUSTER_WINDOW_MINUTES"), 10),
            repeated_days_for_warning=_to_int(os.getenv("AI_REPEATED_DAYS_WARNING"), 3),
            repeated_days_for_escalation=_to_int(os.getenv("AI_REPEATED_DAYS_ESCALATION"), 4),
        )

        openai_settings = OpenAISettings(
            enabled=_to_bool(os.getenv("ENABLE_OPENAI"), True),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            timeout_seconds=_to_float(os.getenv("OPENAI_TIMEOUT_SEC"), 8.0),
            max_retries=_to_int(os.getenv("OPENAI_MAX_RETRIES"), 1),
            max_output_tokens=_to_int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS"), 500),
            llm_min_severity=os.getenv("LLM_MIN_SEVERITY", "medium"),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "minimal"),
        )

        return cls(
            thresholds=thresholds,
            nighttime=nighttime,
            pattern=pattern,
            openai=openai_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> AISettings:
    """Return cached settings loaded from environment variables."""
    return AISettings.from_env()


def severity_rank(severity: str) -> int:
    """Convert severity string to rank for policy checks."""
    table = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    return table.get(severity, 1)
