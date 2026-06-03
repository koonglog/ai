from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ai.ml_features import FEATURE_COLUMNS, FEATURE_SPEC_VERSION
from ai.training.train_utils import build_feature_matrix, load_csv_rows, sort_rows_by_timestamp

MODEL_NAME = "lgbm_noise_risk_feature_spec_2_0_0_base"
TARGET_COLUMN = "noise_risk_level"
CLASS_NAMES = ["normal", "caution", "warning", "high"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
PATTERN_EVENT_COLUMNS = {
    "event_type",
    "severity",
    "severity_score",
    "confidence",
    "is_meaningful",
    "pattern_label",
    "recent_count_10min",
}
FORBIDDEN_FEATURE_COLUMNS = {
    TARGET_COLUMN,
    "feature_spec_version",
    "id",
    "received_at",
    "sensor_timestamp",
    *PATTERN_EVENT_COLUMNS,
}


@dataclass(frozen=True, slots=True)
class SplitRows:
    train: list[dict[str, Any]]
    valid: list[dict[str, Any]]
    test: list[dict[str, Any]]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _required_dataset_columns() -> list[str]:
    return [
        "sensor_id",
        "household_id",
        "sound_level",
        "vibration_value",
        "duration_ms",
        "sensor_timestamp",
        "vibration_raw",
        "vibration_acc_mps2",
        "vibration_dbv",
        "is_daytime",
        "is_nighttime",
        "time_period",
        "vibration_level_category",
        "impact_count_in_window",
        "repeated_impact_flag",
        TARGET_COLUMN,
        "feature_spec_version",
    ]


def validate_training_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("training dataset is empty")
    columns = set(rows[0].keys())
    missing = [col for col in _required_dataset_columns() if col not in columns]
    if missing:
        raise ValueError(f"missing required training columns: {', '.join(missing)}")

    wrong_spec_count = sum(
        1 for row in rows if str(row.get("feature_spec_version")) != FEATURE_SPEC_VERSION
    )
    invalid_labels = sorted(
        {
            str(row.get(TARGET_COLUMN))
            for row in rows
            if str(row.get(TARGET_COLUMN)) not in LABEL_TO_ID
        }
    )
    if invalid_labels:
        raise ValueError(f"invalid noise_risk_level values: {invalid_labels}")

    pattern_columns_present = sorted(PATTERN_EVENT_COLUMNS & columns)
    non_finite_counts = _non_finite_feature_counts(rows)
    return {
        "row_count": len(rows),
        "columns": sorted(columns),
        "missing_required_columns": missing,
        "wrong_feature_spec_version_count": wrong_spec_count,
        "pattern_event_columns_present": pattern_columns_present,
        "vibration_raw_float_preserved": _vibration_raw_float_preserved(rows),
        "non_finite_feature_counts": non_finite_counts,
        "noise_risk_level_distribution": _count_values(rows, TARGET_COLUMN),
        "vibration_level_category_distribution": _count_values(rows, "vibration_level_category"),
        "sensor_id_distribution": _count_values(rows, "sensor_id"),
        "time_period_distribution": _count_values(rows, "time_period"),
    }


def _vibration_raw_float_preserved(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fractional_count = 0
    mismatch_count = 0
    for row in rows:
        value = float(row.get("vibration_value") or 0.0)
        raw = float(row.get("vibration_raw") or 0.0)
        if abs(value - int(value)) > 1e-9:
            fractional_count += 1
        if abs(value - raw) > 1e-9:
            mismatch_count += 1
    return {
        "fractional_vibration_value_count": fractional_count,
        "vibration_value_raw_mismatch_count": mismatch_count,
    }


def _non_finite_feature_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    matrix = build_feature_matrix(rows)
    result: dict[str, int] = {}
    for index, column in enumerate(FEATURE_COLUMNS):
        result[column] = int(np.count_nonzero(~np.isfinite(matrix[:, index])))
    return result


def _timestamp_parse_errors(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for column in columns:
        errors = 0
        for row in rows:
            raw = str(row.get(column) or "").strip()
            if not raw:
                errors += 1
                continue
            try:
                datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                errors += 1
        result[column] = errors
    return result


def validate_pattern_events(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = [
        "sensor_id",
        "household_id",
        "event_type",
        "severity",
        "severity_score",
        "confidence",
        "is_night",
        "is_meaningful",
        "pattern_label",
        "avg_sound_level",
        "max_sound_level",
        "avg_vibration",
        "duration_ms",
        "sample_count",
        "started_at",
        "ended_at",
        "status",
        "recent_count_10min",
    ]
    if not rows:
        raise ValueError("pattern event csv is empty")
    columns = set(rows[0].keys())
    missing = [col for col in required if col not in columns]
    missing_values = {
        col: sum(1 for row in rows if str(row.get(col) or "").strip() == "")
        for col in required
        if col in columns
    }
    return {
        "row_count": len(rows),
        "columns": sorted(columns),
        "missing_required_columns": missing,
        "missing_values": missing_values,
        "timestamp_parse_errors": _timestamp_parse_errors(rows, ["started_at", "ended_at"]),
        "event_type_distribution": _count_values(rows, "event_type"),
        "severity_distribution": _count_values(rows, "severity"),
        "pattern_label_distribution": _count_values(rows, "pattern_label"),
        "is_meaningful_distribution": _count_values(rows, "is_meaningful"),
        "usage": "validation_only_not_training_feature_or_target",
    }


def _count_values(rows: Sequence[Mapping[str, Any]], column: str) -> dict[str, int]:
    counter = Counter(str(row.get(column) or "") for row in rows)
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _label_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return _count_values(rows, TARGET_COLUMN)


def _split_rows_time_based(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestamp_column: str = "sensor_timestamp",
) -> SplitRows:
    sorted_rows = sort_rows_by_timestamp(rows, timestamp_column)
    total = len(sorted_rows)
    train_end = int(total * 0.70)
    valid_end = int(total * 0.85)
    if train_end <= 0 or valid_end <= train_end or valid_end >= total:
        raise ValueError(f"invalid 70/15/15 split for row_count={total}")
    return SplitRows(
        train=[dict(row) for row in sorted_rows[:train_end]],
        valid=[dict(row) for row in sorted_rows[train_end:valid_end]],
        test=[dict(row) for row in sorted_rows[valid_end:]],
    )


def _split_summary(split: SplitRows) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in [("train", split.train), ("validation", split.valid), ("test", split.test)]:
        result[name] = {
            "row_count": len(rows),
            "label_distribution": _label_distribution(rows),
            "sensor_id_distribution": _count_values(rows, "sensor_id"),
            "time_period_distribution": _count_values(rows, "time_period"),
            "timestamp_min": rows[0].get("sensor_timestamp") if rows else None,
            "timestamp_max": rows[-1].get("sensor_timestamp") if rows else None,
        }
    return result


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([LABEL_TO_ID[str(row[TARGET_COLUMN])] for row in rows], dtype=int)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    label_ids = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    per_class: dict[str, Any] = {}
    for idx, label in enumerate(CLASS_NAMES):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_ids).tolist(),
    }


def _predict_rows(model: LGBMClassifier, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = build_feature_matrix(rows)
    prob = np.asarray(model.predict_proba(x), dtype=float)
    pred = np.argmax(prob, axis=1)
    return pred, prob


def _prediction_distribution(labels: Sequence[str]) -> dict[str, int]:
    counter = Counter(str(label) for label in labels)
    return {label: int(counter.get(label, 0)) for label in CLASS_NAMES}


def _normalize_ts(value: Any) -> str:
    parsed = str(value or "").strip().replace("T", " ")
    if not parsed:
        return ""
    try:
        dt = datetime.fromisoformat(parsed.replace("Z", "+00:00"))
    except ValueError:
        return parsed[:19]
    return dt.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")


def _pattern_join_key(row: Mapping[str, Any], timestamp_field: str) -> tuple[str, str]:
    return (str(row.get("sensor_id") or "").strip(), _normalize_ts(row.get(timestamp_field)))


def _rows_with_predictions(
    model: LGBMClassifier,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pred, prob = _predict_rows(model, rows)
    enriched: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        top_idx = int(pred[idx])
        item = dict(row)
        item["model_prediction"] = ID_TO_LABEL[top_idx]
        item["model_prediction_proba"] = float(prob[idx][top_idx])
        enriched.append(item)
    return enriched


def validate_patterns_after_training(
    *,
    model: LGBMClassifier,
    dataset_rows: Sequence[Mapping[str, Any]],
    pattern_rows: Sequence[Mapping[str, Any]],
    mismatch_csv: Path,
) -> dict[str, Any]:
    predicted_rows = _rows_with_predictions(model, dataset_rows)
    main_by_key = {
        _pattern_join_key(row, "sensor_timestamp"): row
        for row in predicted_rows
    }

    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_patterns: list[dict[str, Any]] = []
    matched_keys: set[tuple[str, str]] = set()
    for pattern in pattern_rows:
        key = _pattern_join_key(pattern, "started_at")
        main = main_by_key.get(key)
        if main is None:
            unmatched_patterns.append(dict(pattern))
            continue
        matched.append((main, dict(pattern)))
        matched_keys.add(key)

    no_pattern_rows = [
        row for row in predicted_rows if _pattern_join_key(row, "sensor_timestamp") not in matched_keys
    ]

    by_event_type = _prediction_breakdown(matched, "event_type")
    by_severity = _prediction_breakdown(matched, "severity")
    by_pattern_label = _prediction_breakdown(matched, "pattern_label")
    by_is_meaningful = _prediction_breakdown(matched, "is_meaningful")

    mismatch_rows = _build_mismatch_rows(matched, no_pattern_rows)
    _write_csv(
        mismatch_csv,
        mismatch_rows,
        [
            "mismatch_reason",
            "sensor_id",
            "household_id",
            "sensor_timestamp",
            "sound_level",
            "vibration_raw",
            "vibration_acc_mps2",
            "vibration_dbv",
            "impact_count_in_window",
            "repeated_impact_flag",
            "noise_risk_level",
            "model_prediction",
            "model_prediction_proba",
            "pattern_event_type",
            "pattern_severity",
            "pattern_label",
            "pattern_is_meaningful",
        ],
    )

    return {
        "pattern_event_row_count": len(pattern_rows),
        "matched_pattern_event_count": len(matched),
        "match_rate": (len(matched) / len(pattern_rows)) if pattern_rows else 0.0,
        "unmatched_pattern_event_count": len(unmatched_patterns),
        "unmatched_pattern_examples": unmatched_patterns[:5],
        "matched_prediction_distribution": _prediction_distribution(
            [main["model_prediction"] for main, _ in matched]
        ),
        "no_pattern_prediction_distribution": _prediction_distribution(
            [row["model_prediction"] for row in no_pattern_rows]
        ),
        "by_event_type": by_event_type,
        "by_severity": by_severity,
        "by_pattern_label": by_pattern_label,
        "by_is_meaningful": by_is_meaningful,
        "mismatch_case_count": len(mismatch_rows),
        "mismatch_cases_csv": str(mismatch_csv),
    }


def _prediction_breakdown(
    matched: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    pattern_column: str,
) -> dict[str, dict[str, int]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for main, pattern in matched:
        buckets[str(pattern.get(pattern_column) or "")].append(str(main["model_prediction"]))
    return {
        key: _prediction_distribution(values)
        for key, values in sorted(buckets.items())
    }


def _mismatch_base_row(
    reason: str,
    main: Mapping[str, Any],
    pattern: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "mismatch_reason": reason,
        "sensor_id": main.get("sensor_id", ""),
        "household_id": main.get("household_id", ""),
        "sensor_timestamp": main.get("sensor_timestamp", ""),
        "sound_level": main.get("sound_level", ""),
        "vibration_raw": main.get("vibration_raw", ""),
        "vibration_acc_mps2": main.get("vibration_acc_mps2", ""),
        "vibration_dbv": main.get("vibration_dbv", ""),
        "impact_count_in_window": main.get("impact_count_in_window", ""),
        "repeated_impact_flag": main.get("repeated_impact_flag", ""),
        "noise_risk_level": main.get("noise_risk_level", ""),
        "model_prediction": main.get("model_prediction", ""),
        "model_prediction_proba": main.get("model_prediction_proba", ""),
        "pattern_event_type": (pattern or {}).get("event_type", ""),
        "pattern_severity": (pattern or {}).get("severity", ""),
        "pattern_label": (pattern or {}).get("pattern_label", ""),
        "pattern_is_meaningful": (pattern or {}).get("is_meaningful", ""),
    }


def _build_mismatch_rows(
    matched: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    no_pattern_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for main, pattern in matched:
        prediction = str(main["model_prediction"])
        if str(pattern.get("severity")) == "critical" and prediction in {"normal", "caution"}:
            rows.append(_mismatch_base_row("critical_pattern_predicted_low", main, pattern))
        if str(pattern.get("pattern_label")) == "night_repeated_impact" and prediction == "normal":
            rows.append(_mismatch_base_row("night_repeated_impact_predicted_normal", main, pattern))
        if str(pattern.get("event_type")) == "impact_noise" and prediction == "normal":
            rows.append(_mismatch_base_row("impact_noise_predicted_normal", main, pattern))
        if prediction == "high" and str(pattern.get("severity")) == "low":
            rows.append(_mismatch_base_row("high_prediction_pattern_low", main, pattern))

    for main in no_pattern_rows:
        if str(main.get("model_prediction")) == "high":
            rows.append(_mismatch_base_row("high_prediction_without_pattern_event", main, None))
    return rows


def _feature_importance_rows(model: LGBMClassifier) -> list[dict[str, Any]]:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    rows = [
        {
            "feature": feature,
            "gain": float(gain[idx]),
            "split": int(split[idx]),
        }
        for idx, feature in enumerate(FEATURE_COLUMNS)
    ]
    return sorted(rows, key=lambda row: (-float(row["gain"]), row["feature"]))


def _confusion_matrix_rows(matrix: list[list[int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(CLASS_NAMES):
        row = {"actual": label}
        for pred_idx, pred_label in enumerate(CLASS_NAMES):
            row[f"pred_{pred_label}"] = int(matrix[idx][pred_idx])
        rows.append(row)
    return rows


def _write_training_report(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    pattern_validation: Mapping[str, Any],
    feature_importance_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# LightGBM Noise Risk Training Report",
        "",
        "## Scope",
        "- Training dataset: `training_dataset_feature_spec_2_0_0.csv`",
        "- Pattern event CSV is used for post-training validation only.",
        "- Target `noise_risk_level` is an internal pseudo-label, not legal or externally verified ground truth.",
        "",
        "## Dataset",
        f"- Feature spec version: `{metadata['feature_spec_version']}`",
        f"- Target column: `{metadata['target_column']}`",
        f"- Feature count: `{len(metadata['feature_names'])}`",
        f"- Train rows: `{metadata['split']['train']['row_count']}`",
        f"- Validation rows: `{metadata['split']['validation']['row_count']}`",
        f"- Test rows: `{metadata['split']['test']['row_count']}`",
        "",
        "## Test Metrics",
        f"- Accuracy: `{metrics['accuracy']:.6f}`",
        f"- Macro F1: `{metrics['macro_f1']:.6f}`",
        f"- Weighted F1: `{metrics['weighted_f1']:.6f}`",
        "",
        "## Feature Importance Top 10",
    ]
    for row in list(feature_importance_rows)[:10]:
        lines.append(f"- `{row['feature']}`: gain={float(row['gain']):.4f}, split={row['split']}")
    lines.extend(
        [
            "",
            "## Pattern Validation",
            f"- Pattern rows: `{pattern_validation['pattern_event_row_count']}`",
            f"- Matched rows: `{pattern_validation['matched_pattern_event_count']}`",
            f"- Match rate: `{pattern_validation['match_rate']:.6f}`",
            f"- Mismatch case rows: `{pattern_validation['mismatch_case_count']}`",
            "",
            "## Limitations",
            "- `noise_risk_level` is generated from current internal alert rules.",
            "- Pattern event labels are previous pattern-classification outputs and are not used for training.",
            "- This model reproduces the current alert-rule pseudo-labels; it is not a verified inter-floor-noise ground-truth model.",
            "- Future supervised labels from real complaints, user feedback, or administrator verification are required.",
        ]
    )
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_noise_risk_model(
    *,
    dataset: Path,
    pattern_events: Path,
    artifact_dir: Path,
    report_dir: Path,
    seed: int = 42,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    rows = load_csv_rows(dataset)
    validation = validate_training_dataset(rows)
    pattern_rows = load_csv_rows(pattern_events)
    pattern_validation_input = validate_pattern_events(pattern_rows)

    split = _split_rows_time_based(rows, timestamp_column="sensor_timestamp")
    x_train = build_feature_matrix(split.train)
    y_train = _labels(split.train)
    x_valid = build_feature_matrix(split.valid)
    y_valid = _labels(split.valid)
    x_test = build_feature_matrix(split.test)
    y_test = _labels(split.test)

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        feature_name=FEATURE_COLUMNS,
    )

    y_prob = np.asarray(model.predict_proba(x_test), dtype=float)
    y_pred = np.argmax(y_prob, axis=1)
    test_metrics = _metrics(y_test, y_pred, y_prob)

    _ensure_dir(artifact_dir)
    _ensure_dir(report_dir)

    artifact_path = artifact_dir / f"{MODEL_NAME}.pkl"
    metadata_path = artifact_dir / f"{MODEL_NAME}.metadata.json"
    label_map_path = artifact_dir / f"{MODEL_NAME}.label_map.json"
    confusion_path = report_dir / "lgbm_feature_spec_2_0_0_confusion_matrix.csv"
    feature_importance_path = report_dir / "lgbm_feature_spec_2_0_0_feature_importance.csv"
    training_report_path = report_dir / "lgbm_feature_spec_2_0_0_training_report.md"
    pattern_report_path = report_dir / "pattern_event_validation_report.md"
    mismatch_csv_path = report_dir / "pattern_validation_mismatch_cases.csv"

    feature_importance = _feature_importance_rows(model)
    pattern_validation = validate_patterns_after_training(
        model=model,
        dataset_rows=rows,
        pattern_rows=pattern_rows,
        mismatch_csv=mismatch_csv_path,
    )

    artifact_payload = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "target_column": TARGET_COLUMN,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "model_name": MODEL_NAME,
    }
    with artifact_path.open("wb") as f:
        pickle.dump(artifact_payload, f)

    split_summary = _split_summary(split)
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "dataset_path": str(dataset),
        "pattern_events_path": str(pattern_events),
        "pattern_events_usage": "post_training_validation_only_not_feature_not_target",
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "feature_names": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "target_classes": CLASS_NAMES,
        "label_mapping": LABEL_TO_ID,
        "categorical_features": [],
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "split_method": "time_based_sensor_timestamp_70_15_15",
        "split": split_summary,
        "model_params": {
            "objective": "multiclass",
            "num_class": len(CLASS_NAMES),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "class_weight": "balanced",
            "random_state": seed,
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        },
        "dataset_validation": validation,
        "pattern_event_validation_input": pattern_validation_input,
        "pattern_validation": pattern_validation,
        "test_metrics": test_metrics,
        "pseudo_label": {
            "used": True,
            "target": TARGET_COLUMN,
            "is_ground_truth": False,
            "description": "Internal alert-rule pseudo-label, not legal or externally verified ground truth.",
        },
        "legal_caution": (
            "This model is not a legal noise violation detector. It reproduces "
            "current internal alert-rule pseudo-labels."
        ),
        "artifacts": {
            "model_pickle": str(artifact_path),
            "metadata_json": str(metadata_path),
            "label_map_json": str(label_map_path),
            "training_report_md": str(training_report_path),
            "confusion_matrix_csv": str(confusion_path),
            "feature_importance_csv": str(feature_importance_path),
            "pattern_validation_report_md": str(pattern_report_path),
            "mismatch_cases_csv": str(mismatch_csv_path),
        },
    }

    _write_json(metadata_path, metadata)
    _write_json(label_map_path, {"label_to_id": LABEL_TO_ID, "id_to_label": CLASS_NAMES})
    _write_csv(confusion_path, _confusion_matrix_rows(test_metrics["confusion_matrix"]), ["actual", *[f"pred_{label}" for label in CLASS_NAMES]])
    _write_csv(feature_importance_path, feature_importance, ["feature", "gain", "split"])
    _write_training_report(
        training_report_path,
        metadata=metadata,
        metrics=test_metrics,
        pattern_validation=pattern_validation,
        feature_importance_rows=feature_importance,
    )
    _write_pattern_report(pattern_report_path, pattern_validation)

    return {
        "metadata": metadata,
        "pattern_validation": pattern_validation,
    }


def _write_pattern_report(path: Path, pattern_validation: Mapping[str, Any]) -> None:
    def _append_breakdown(title: str, breakdown: Mapping[str, Mapping[str, int]], lines: list[str]) -> None:
        lines.extend(["", f"## {title}"])
        for key, distribution in breakdown.items():
            compact = ", ".join(f"{label}={count}" for label, count in distribution.items())
            lines.append(f"- `{key}`: {compact}")

    lines = [
        "# Pattern Event Validation Report",
        "",
        "Pattern event rows are used only after training for reference analysis.",
        "They are not training features, target labels, or label correction sources.",
        "",
        f"- Pattern rows: `{pattern_validation['pattern_event_row_count']}`",
        f"- Matched rows: `{pattern_validation['matched_pattern_event_count']}`",
        f"- Match rate: `{pattern_validation['match_rate']:.6f}`",
        f"- Unmatched rows: `{pattern_validation['unmatched_pattern_event_count']}`",
        f"- Mismatch case rows: `{pattern_validation['mismatch_case_count']}`",
        "",
        "## Prediction Distribution For Matched Pattern Rows",
    ]
    for label, count in pattern_validation["matched_prediction_distribution"].items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Prediction Distribution For Rows Without Pattern Event"])
    for label, count in pattern_validation["no_pattern_prediction_distribution"].items():
        lines.append(f"- `{label}`: {count}")
    _append_breakdown("By Event Type", pattern_validation["by_event_type"], lines)
    _append_breakdown("By Severity", pattern_validation["by_severity"], lines)
    _append_breakdown("By Pattern Label", pattern_validation["by_pattern_label"], lines)
    _append_breakdown("By Is Meaningful", pattern_validation["by_is_meaningful"], lines)
    lines.extend(
        [
            "",
            "## Interpretation",
            "Differences are review signals only. Pattern event columns are previous pattern-classification outputs, not verified ground truth.",
        ]
    )
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_noise_risk_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise ValueError("feature_spec_version mismatch")
    if list(payload.get("feature_names") or []) != FEATURE_COLUMNS:
        raise ValueError("feature_names mismatch")
    return payload


def predict_noise_risk_from_rows(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model = artifact["model"]
    pred, prob = _predict_rows(model, rows)
    id_to_label = {int(k): str(v) for k, v in dict(artifact["id_to_label"]).items()}
    output: list[dict[str, Any]] = []
    for idx, pred_id in enumerate(pred):
        label = id_to_label[int(pred_id)]
        output.append(
            {
                "model_prediction": label,
                "model_prediction_proba": float(prob[idx][int(pred_id)]),
            }
        )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LightGBM noise_risk_level model.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--pattern-events", required=True, type=Path)
    parser.add_argument("--artifact-dir", default=Path("ai/artifacts"), type=Path)
    parser.add_argument("--report-dir", default=Path("ai/artifacts/reports"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n-estimators", default=400, type=int)
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--num-leaves", default=31, type=int)
    parser.add_argument("--early-stopping-rounds", default=50, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_noise_risk_model(
        dataset=args.dataset,
        pattern_events=args.pattern_events,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
