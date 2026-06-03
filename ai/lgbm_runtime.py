from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np

from .config import AISettings, get_settings
from .ml_features import FEATURE_COLUMNS, feature_vector_from_event
from .schemas import EventFeatures

NOISE_RISK_MODEL_NAME = "lgbm_noise_risk_feature_spec_2_0_0_base"
NOISE_RISK_MODEL_FILENAME = f"{NOISE_RISK_MODEL_NAME}.pkl"
NOISE_RISK_METADATA_FILENAME = f"{NOISE_RISK_MODEL_NAME}.metadata.json"
NOISE_RISK_LABEL_MAP_FILENAME = f"{NOISE_RISK_MODEL_NAME}.label_map.json"
NOISE_RISK_FEATURE_SPEC_VERSION = "2.0.0"
NOISE_RISK_LABELS = ["normal", "caution", "warning", "high"]


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
    noise_risk_level: str | None = None
    noise_risk_probability: float | None = None
    noise_risk_probabilities: dict[str, float] | None = None
    prediction_path: str | None = None
    model_version: str | None = None
    feature_spec_version: str | None = None
    model_path: str | None = None
    metadata_path: str | None = None
    label_map_path: str | None = None


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

        self._runtime_mode: str | None = None
        self._noise_risk_model: Any | None = None
        self._noise_risk_id_to_label: list[str] = []
        self._noise_risk_metadata: dict[str, Any] = {}
        self._noise_risk_model_path: Path | None = None
        self._noise_risk_metadata_path: Path | None = None
        self._noise_risk_label_map_path: Path | None = None

    def _paths(self) -> dict[str, Path]:
        return {
            "meaningful_model": self.model_dir / "is_meaningful_model.txt",
            "event_type_model": self.model_dir / "event_type_model.txt",
            "label_map": self.model_dir / "event_type_label_map.json",
            "feature_spec": self.model_dir / "feature_spec.json",
        }

    def _noise_risk_paths(self) -> dict[str, Path] | None:
        candidates: list[Path]
        if self.model_dir.suffix.lower() == ".pkl":
            candidates = [self.model_dir]
        else:
            candidates = [self.model_dir / NOISE_RISK_MODEL_FILENAME]
            if self.model_dir.name == "models":
                candidates.append(self.model_dir.parent / NOISE_RISK_MODEL_FILENAME)

        model_path = next((path for path in candidates if path.exists()), None)
        if model_path is None:
            return None

        return {
            "noise_risk_model": model_path,
            "metadata": model_path.with_name(NOISE_RISK_METADATA_FILENAME),
            "label_map": model_path.with_name(NOISE_RISK_LABEL_MAP_FILENAME),
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

    def _validate_feature_names(self, payload: Mapping[str, Any], *, source: str) -> None:
        columns = payload.get("feature_names") or payload.get("feature_columns")
        if columns is None:
            raise RuntimeError(f"{source} has no feature_names")
        if list(columns) != FEATURE_COLUMNS:
            missing = [name for name in FEATURE_COLUMNS if name not in list(columns)]
            extra = [name for name in list(columns) if name not in FEATURE_COLUMNS]
            raise RuntimeError(
                "feature_names mismatch: "
                f"source={source}, missing={missing}, extra={extra}"
            )

    def _load_noise_risk_artifact(self, paths: dict[str, Path]) -> bool:
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            self._set_error(
                "missing LightGBM noise-risk artifacts: "
                + ", ".join(f"{name}={paths[name]}" for name in missing)
            )
            return True

        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("feature_spec_version") != NOISE_RISK_FEATURE_SPEC_VERSION:
                raise RuntimeError(
                    "feature_spec_version mismatch: "
                    f"{metadata.get('feature_spec_version')} != {NOISE_RISK_FEATURE_SPEC_VERSION}"
                )
            self._validate_feature_names(metadata, source=str(paths["metadata"]))

            label_map = json.loads(paths["label_map"].read_text(encoding="utf-8"))
            id_to_label = label_map.get("id_to_label")
            if not isinstance(id_to_label, list) or [str(v) for v in id_to_label] != NOISE_RISK_LABELS:
                raise RuntimeError("noise-risk label_map has invalid id_to_label")

            with paths["noise_risk_model"].open("rb") as f:
                artifact = pickle.load(f)
            if not isinstance(artifact, Mapping):
                raise RuntimeError("noise-risk pickle payload must be a mapping")
            if artifact.get("feature_spec_version") != NOISE_RISK_FEATURE_SPEC_VERSION:
                raise RuntimeError(
                    "pickle feature_spec_version mismatch: "
                    f"{artifact.get('feature_spec_version')} != {NOISE_RISK_FEATURE_SPEC_VERSION}"
                )
            self._validate_feature_names(artifact, source=str(paths["noise_risk_model"]))
            model = artifact.get("model")
            if model is None or not hasattr(model, "predict_proba"):
                raise RuntimeError("noise-risk pickle payload has no predict_proba model")

            self._noise_risk_model = model
            self._noise_risk_id_to_label = [str(v) for v in id_to_label]
            self._noise_risk_metadata = dict(metadata)
            self._noise_risk_model_path = paths["noise_risk_model"]
            self._noise_risk_metadata_path = paths["metadata"]
            self._noise_risk_label_map_path = paths["label_map"]
            self._runtime_mode = "noise_risk"
            self._ready = True
            self._load_error = None
            return True
        except Exception as exc:  # pragma: no cover - defensive runtime check
            self._set_error(f"failed to load LightGBM noise-risk artifact: {exc}")
            return True

    def load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        noise_risk_paths = self._noise_risk_paths()
        if noise_risk_paths is not None and self._load_noise_risk_artifact(noise_risk_paths):
            return

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
            self._runtime_mode = "event_type"
            self._ready = True
            self._load_error = None
        except Exception as exc:  # pragma: no cover - defensive runtime check
            self._set_error(f"failed to load LightGBM artifacts: {exc}")

    def status(self) -> dict[str, Any]:
        self.load()
        classes = (
            list(self._noise_risk_id_to_label)
            if self._runtime_mode == "noise_risk"
            else list(self._id_to_label)
        )
        return {
            "ready": self._ready,
            "load_attempted": self._load_attempted,
            "error": self._load_error,
            "model_dir": str(self.model_dir),
            "runtime_mode": self._runtime_mode,
            "classes": classes,
            "model_path": str(self._noise_risk_model_path) if self._noise_risk_model_path else None,
            "metadata_path": (
                str(self._noise_risk_metadata_path) if self._noise_risk_metadata_path else None
            ),
            "label_map_path": (
                str(self._noise_risk_label_map_path) if self._noise_risk_label_map_path else None
            ),
            "model_version": self._noise_risk_metadata.get("model_name"),
            "feature_spec_version": self._noise_risk_metadata.get("feature_spec_version"),
            "feature_names_match": (
                list(self._noise_risk_metadata.get("feature_names") or []) == FEATURE_COLUMNS
                if self._runtime_mode == "noise_risk"
                else None
            ),
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

    def _predict_noise_risk(self, event: EventFeatures) -> LGBMInferenceResult:
        if self._noise_risk_model is None:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=self._load_error or "LightGBM noise-risk model is not ready",
                model_dir=str(self.model_dir),
                prediction_path="rule_fallback",
                model_path=str(self._noise_risk_model_path) if self._noise_risk_model_path else None,
                metadata_path=(
                    str(self._noise_risk_metadata_path) if self._noise_risk_metadata_path else None
                ),
                label_map_path=(
                    str(self._noise_risk_label_map_path) if self._noise_risk_label_map_path else None
                ),
            )

        x = np.asarray([feature_vector_from_event(event, settings=self.settings)], dtype=float)
        try:
            raw_prob = np.asarray(self._noise_risk_model.predict_proba(x), dtype=float)
        except Exception as exc:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=f"LightGBM noise-risk inference failed: {exc}",
                model_dir=str(self.model_dir),
                prediction_path="rule_fallback",
                model_path=str(self._noise_risk_model_path) if self._noise_risk_model_path else None,
                metadata_path=(
                    str(self._noise_risk_metadata_path) if self._noise_risk_metadata_path else None
                ),
                label_map_path=(
                    str(self._noise_risk_label_map_path) if self._noise_risk_label_map_path else None
                ),
            )

        probs = raw_prob[0] if raw_prob.ndim == 2 else raw_prob
        if probs.size != len(self._noise_risk_id_to_label):
            return LGBMInferenceResult(
                should_fallback=True,
                reason=(
                    "noise-risk probabilities size mismatch: "
                    f"probs={probs.size}, labels={len(self._noise_risk_id_to_label)}"
                ),
                model_dir=str(self.model_dir),
                prediction_path="rule_fallback",
                model_path=str(self._noise_risk_model_path) if self._noise_risk_model_path else None,
                metadata_path=(
                    str(self._noise_risk_metadata_path) if self._noise_risk_metadata_path else None
                ),
                label_map_path=(
                    str(self._noise_risk_label_map_path) if self._noise_risk_label_map_path else None
                ),
            )

        top_idx = int(np.argmax(probs))
        label = self._noise_risk_id_to_label[top_idx]
        prob_map = {
            label_name: float(probs[idx])
            for idx, label_name in enumerate(self._noise_risk_id_to_label)
        }
        return LGBMInferenceResult(
            should_fallback=False,
            reason=None,
            is_meaningful=label in {"warning", "high"},
            meaningful_probability=float(probs[top_idx]),
            noise_risk_level=label,
            noise_risk_probability=float(probs[top_idx]),
            noise_risk_probabilities=prob_map,
            prediction_path="lightgbm",
            model_dir=str(self.model_dir),
            model_version=str(self._noise_risk_metadata.get("model_name") or NOISE_RISK_MODEL_NAME),
            feature_spec_version=str(
                self._noise_risk_metadata.get("feature_spec_version")
                or NOISE_RISK_FEATURE_SPEC_VERSION
            ),
            model_path=str(self._noise_risk_model_path) if self._noise_risk_model_path else None,
            metadata_path=str(self._noise_risk_metadata_path) if self._noise_risk_metadata_path else None,
            label_map_path=str(self._noise_risk_label_map_path) if self._noise_risk_label_map_path else None,
        )

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
                prediction_path="rule_fallback",
            )

        if not self._ready:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=self._load_error or "LightGBM runtime is not ready",
                model_dir=str(self.model_dir),
                prediction_path="rule_fallback",
            )

        if self._runtime_mode == "noise_risk":
            return self._predict_noise_risk(event)

        if self._meaningful_booster is None or self._event_type_booster is None:
            return LGBMInferenceResult(
                should_fallback=True,
                reason=self._load_error or "LightGBM event-type model is not ready",
                model_dir=str(self.model_dir),
                prediction_path="rule_fallback",
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
                prediction_path="rule_fallback",
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
                prediction_path="rule_fallback",
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
                prediction_path="rule_fallback",
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
            prediction_path="lightgbm",
        )


@lru_cache(maxsize=1)
def get_lgbm_runtime() -> LightGBMRuntime:
    return LightGBMRuntime()
