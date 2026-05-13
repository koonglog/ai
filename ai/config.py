from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return None


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

    # Official baseline (Rules on Inter-floor Noise in Multi-family Housing):
    # - Direct impact noise: Leq(1min) 39/34 (day/night), Lmax 57/52 (day/night)
    # - Airborne noise: Leq(5min) 45/40 (day/night)
    direct_impact_leq_day: float = 39.0
    direct_impact_leq_night: float = 34.0
    direct_impact_lmax_day: float = 57.0
    direct_impact_lmax_night: float = 52.0
    airborne_leq_day: float = 45.0
    airborne_leq_night: float = 40.0
    # Optional offset for legacy buildings (e.g., +2dB after 2025-01-01).
    direct_impact_extra_db: float = 0.0
    vibration_low: int = 150
    vibration_mid: int = 350
    vibration_high: int = 600
    duration_short_ms: int = 3000
    duration_medium_ms: int = 8000
    duration_long_ms: int = 15000

    def impact_leq(self, night: bool) -> float:
        base = self.direct_impact_leq_night if night else self.direct_impact_leq_day
        return base + self.direct_impact_extra_db

    def impact_lmax(self, night: bool) -> float:
        base = self.direct_impact_lmax_night if night else self.direct_impact_lmax_day
        return base + self.direct_impact_extra_db

    def airborne_leq(self, night: bool) -> float:
        return self.airborne_leq_night if night else self.airborne_leq_day

    # Backward-compatible aliases used by older call sites/docs.
    @property
    def db_background(self) -> float:
        return self.direct_impact_leq_day

    @property
    def db_daily(self) -> float:
        return self.airborne_leq_day

    @property
    def db_impact(self) -> float:
        return self.direct_impact_lmax_day


@dataclass(frozen=True)
class NightTimeSettings:
    """Night-time settings (local timezone-based hour window)."""

    start_hour: int = 22
    end_hour: int = 6

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
class ClassifierSettings:
    """Classifier backend settings."""

    backend: str = "rule"  # rule | hybrid | lightgbm
    lgbm_model_dir: str = "./ai/artifacts/models"
    lgbm_min_confidence: float = 0.7
    lgbm_shadow_mode: bool = False
    lgbm_shadow_log_path: str = "./ai/artifacts/logs/lgbm_shadow.jsonl"

    def validate(self) -> None:
        allowed = {"rule", "hybrid", "lightgbm"}
        if self.backend not in allowed:
            raise ValueError(
                f"AI_CLASSIFIER_BACKEND must be one of {sorted(allowed)} (got: {self.backend})"
            )
        if not (0.0 <= self.lgbm_min_confidence <= 1.0):
            raise ValueError("AI_LGBM_MIN_CONFIDENCE must be in range 0.0..1.0")


@dataclass(frozen=True)
class AISettings:
    """Top-level AI settings container."""

    thresholds: ThresholdSettings
    nighttime: NightTimeSettings
    pattern: PatternSettings
    openai: OpenAISettings
    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)

    @classmethod
    def from_env(cls) -> "AISettings":
        thresholds = ThresholdSettings(
            direct_impact_leq_day=_to_float(
                _first_env("AI_DIRECT_IMPACT_LEQ_DAY", "AI_DB_BACKGROUND"),
                39.0,
            ),
            direct_impact_leq_night=_to_float(
                os.getenv("AI_DIRECT_IMPACT_LEQ_NIGHT"),
                34.0,
            ),
            direct_impact_lmax_day=_to_float(
                _first_env("AI_DIRECT_IMPACT_LMAX_DAY", "AI_DB_IMPACT"),
                57.0,
            ),
            direct_impact_lmax_night=_to_float(
                os.getenv("AI_DIRECT_IMPACT_LMAX_NIGHT"),
                52.0,
            ),
            airborne_leq_day=_to_float(
                _first_env("AI_AIRBORNE_LEQ_DAY", "AI_DB_DAILY"),
                45.0,
            ),
            airborne_leq_night=_to_float(
                os.getenv("AI_AIRBORNE_LEQ_NIGHT"),
                40.0,
            ),
            direct_impact_extra_db=_to_float(
                os.getenv("AI_DIRECT_IMPACT_EXTRA_DB"),
                0.0,
            ),
            vibration_low=_to_int(os.getenv("AI_VIBRATION_LOW"), 150),
            vibration_mid=_to_int(os.getenv("AI_VIBRATION_MID"), 350),
            vibration_high=_to_int(os.getenv("AI_VIBRATION_HIGH"), 600),
            duration_short_ms=_to_int(os.getenv("AI_DURATION_SHORT_MS"), 3000),
            duration_medium_ms=_to_int(os.getenv("AI_DURATION_MEDIUM_MS"), 8000),
            duration_long_ms=_to_int(os.getenv("AI_DURATION_LONG_MS"), 15000),
        )

        nighttime = NightTimeSettings(
            start_hour=_to_int(os.getenv("AI_NIGHT_START_HOUR"), 22),
            end_hour=_to_int(os.getenv("AI_NIGHT_END_HOUR"), 6),
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

        classifier = ClassifierSettings(
            backend=os.getenv("AI_CLASSIFIER_BACKEND", "rule").strip().lower(),
            lgbm_model_dir=os.getenv("AI_LGBM_MODEL_DIR", "./ai/artifacts/models"),
            lgbm_min_confidence=_to_float(os.getenv("AI_LGBM_MIN_CONFIDENCE"), 0.7),
            lgbm_shadow_mode=_to_bool(os.getenv("AI_LGBM_SHADOW_MODE"), False),
            lgbm_shadow_log_path=os.getenv(
                "AI_LGBM_SHADOW_LOG_PATH",
                "./ai/artifacts/logs/lgbm_shadow.jsonl",
            ),
        )
        classifier.validate()

        return cls(
            thresholds=thresholds,
            nighttime=nighttime,
            pattern=pattern,
            openai=openai_settings,
            classifier=classifier,
        )


@lru_cache(maxsize=1)
def get_settings() -> AISettings:
    """Return cached settings loaded from environment variables."""
    return AISettings.from_env()


def severity_rank(severity: str) -> int:
    """Convert severity string to rank for policy checks."""
    table = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    return table.get(severity, 1)
