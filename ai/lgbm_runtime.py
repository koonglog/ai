from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from .config import AISettings, get_settings
from .ml_features import FEATURE_COLUMNS, feature_vector_from_event
from .schemas import EventFeatures


@dataclass(slots=True)
class LGBMInferenceResult:
    should_fallback: bool
    reason: str | None
    is_meaningful: bool | None = None
    meaningful_probability: float | None = None
    event_type: str | None = None
    event_type_probability: float | None = None
    event_type_probabilities: dict[str, float] | None = None
    model_dir: str | None = None


class LightGBMRuntime:
    """Lazy-loading inference runtime for LightGBM classifiers."""

    def __init__(
        self,
        *,
        settings: AISettings | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_dir = Path(model_dir or self.settings.classifier.lgbm_model_dir)
        self.min_confidence = float(self.settings.classifier.lgbm_min_confidence)

        self._load_attempted = False
        self._ready = False
        self._load_error: str | None = None

        self._meaningful_booster: lgb.Booster | None = None
        self._event_type_booster: lgb.Booster | None = None
        self._id_to_label: list[str] = []

    def _paths(self) -> dict[str, Path]:
        return {
            "meaningful_model": self.model_dir / "is_meaningful_model.txt",
            "event_type_model": self.model_dir / "event_type_model.txt",
            "label_map": self.model_dir / "event_type_label_map.json",
            "feature_spec": self.model_dir / "feature_spec.json",
        }

    def _set_error(self, message: str) -> None:
        self._ready = False
        self._load_error = message

    def _validate_feature_spec(self, spec_path: Path) -> None:
        if not spec_path.exists():
            return
        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive runtime check
            raise RuntimeError(f"failed to read feature spec: {exc}") from exc
        columns = payload.get("feature_columns")
        if columns is None:
            return
        if list(columns) != FEATURE_COLUMNS:
            raise RuntimeError(
                "feature_spec mismatch: runtime FEATURE_COLUMNS and model feature spec differ"
            )

    def load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        paths = self._paths()
        missing = [name for name, path in paths.items() if name != "feature_spec" and not path.exists()]
        if missing:
            self._set_error(
                "missing LightGBM artifacts: "
                + ", ".join(f"{name}={paths[name]}" for name in missing)
            )
            return

        try:
            self._validate_feature_spec(paths["feature_spec"])
            label_map = json.loads(paths["label_map"].read_text(encoding="utf-8"))
            id_to_label = label_map.get("id_to_label")
            if not isinstance(id_to_label, list) or not id_to_label:
                raise RuntimeError("event_type_label_map.json has invalid id_to_label")
            self._id_to_label = [str(v) for v in id_to_label]

            self._meaningful_booster = lgb.Booster(model_file=str(paths["meaningful_model"]))
            self._event_type_booster = lgb.Booster(model_file=str(paths["event_type_model"]))
            self._ready = True
            self._load_error = None
        except Exception as exc:  # pragma: no cover - defensive runtime check
            self._set_error(f"failed to load LightGBM artifacts: {exc}")

    def status(self) -> dict[str, Any]:
        self.load()
        return {
            "ready": self._ready,
            "load_attempted": self._load_attempted,
            "error": self._load_error,
            "model_dir": str(self.model_dir),
            "classes": list(self._id_to_label),
        }

    def _normalize_event_probs(self, raw: np.ndarray) -> np.ndarray:
        if raw.ndim == 2:
            if raw.shape[0] < 1:
                raise RuntimeError("event_type prediction returned empty 2D tensor")
            probs = raw[0]
        elif raw.ndim == 1:
            if raw.size == len(self._id_to_label):
                probs = raw
            elif raw.size == 1 and len(self._id_to_label) == 2:
                p1 = float(raw[0])
                probs = np.asarray([1.0 - p1, p1], dtype=float)
            else:
                raise RuntimeError(
                    "unexpected event_type prediction shape: "
                    f"size={raw.size}, expected={len(self._id_to_label)}"
                )
        else:
            raise RuntimeError(f"unexpected event_type prediction ndim={raw.ndim}")

        if probs.size != len(self._id_to_label):
            raise RuntimeError(
                "event_type probabilities size mismatch: "
                f"probs={probs.size}, labels={len(self._id_to_label)}"
            )
        return probs.astype(float)

    def predict_event(
        self,
        event: EventFeatures,
        *,
        min_confidence: float | None = None,
    ) -> LGBMInferenceResult:
        self.load()
        threshold = self.min_confidence if min_confidence is None else float(min_confidence)
        if not (0.0 <= threshold <= 1.0):
            return LGBMInferenceResult(
                should_fallback=True,
                reason=f"invalid min_confidence: {threshold}",
                model_dir=str(self.model_dir),
            )

        if not self._ready or self._meaningful_booster is None or self._event_type_booster is None:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=self._load_error or "LightGBM runtime is not ready",
                model_dir=str(self.model_dir),
            )

        x = np.asarray([feature_vector_from_event(event, settings=self.settings)], dtype=float)

        meaningful_prob = float(np.asarray(self._meaningful_booster.predict(x), dtype=float)[0])
        if meaningful_prob < threshold:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=(
                    "meaningful probability below threshold: "
                    f"{meaningful_prob:.4f} < {threshold:.4f}"
                ),
                is_meaningful=False,
                meaningful_probability=meaningful_prob,
                model_dir=str(self.model_dir),
            )

        raw_event_probs = np.asarray(self._event_type_booster.predict(x), dtype=float)
        try:
            probs = self._normalize_event_probs(raw_event_probs)
        except RuntimeError as exc:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=str(exc),
                is_meaningful=True,
                meaningful_probability=meaningful_prob,
                model_dir=str(self.model_dir),
            )

        top_idx = int(np.argmax(probs))
        event_type = self._id_to_label[top_idx]
        event_prob = float(probs[top_idx])
        prob_map = {label: float(probs[idx]) for idx, label in enumerate(self._id_to_label)}

        if event_prob < threshold:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=(
                    "event_type probability below threshold: "
                    f"{event_prob:.4f} < {threshold:.4f}"
                ),
                is_meaningful=True,
                meaningful_probability=meaningful_prob,
                event_type=event_type,
                event_type_probability=event_prob,
                event_type_probabilities=prob_map,
                model_dir=str(self.model_dir),
            )

        return LGBMInferenceResult(
            should_fallback=False,
            reason=None,
            is_meaningful=True,
            meaningful_probability=meaningful_prob,
            event_type=event_type,
            event_type_probability=event_prob,
            event_type_probabilities=prob_map,
            model_dir=str(self.model_dir),
        )


@lru_cache(maxsize=1)
def get_lgbm_runtime() -> LightGBMRuntime:
    return LightGBMRuntime()
