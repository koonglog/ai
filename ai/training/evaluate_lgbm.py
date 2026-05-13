from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from ai.training.train_utils import (
    build_feature_matrix,
    dump_json,
    label_distribution,
    load_csv_rows,
    pick_timestamp_column,
    sort_rows_by_timestamp,
    time_split_rows,
)


def _safe_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)[
                0
            ]
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": None,
        "roc_auc": None,
    }
    if len(set(int(v) for v in y_true.tolist())) >= 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def _safe_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    label_ids = list(range(len(label_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0,
    )
    per_class: dict[str, Any] = {}
    for idx, label in enumerate(label_names):
        per_class[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "roc_auc_ovr": None,
    }
    try:
        if len(label_names) == 2:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", labels=label_ids)
            )
    except ValueError:
        metrics["roc_auc_ovr"] = None
    return metrics


def evaluate_models(
    *,
    dataset: Path,
    model_dir: Path,
    out_json: Path,
    train_ratio: float = 0.8,
    meaningful_threshold: float = 0.5,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    rows = load_csv_rows(dataset)
    if not rows:
        raise ValueError(f"empty dataset: {dataset}")

    ts_col = pick_timestamp_column(rows, preferred=timestamp_column)
    sorted_rows = sort_rows_by_timestamp(rows, ts_col)
    _, valid_rows = time_split_rows(sorted_rows, train_ratio=train_ratio)

    x_valid = build_feature_matrix(valid_rows)
    y_valid_binary = np.asarray(
        [int(row.get("is_meaningful_label") or 0) for row in valid_rows],
        dtype=int,
    )

    meaningful_model_path = model_dir / "is_meaningful_model.txt"
    event_type_model_path = model_dir / "event_type_model.txt"
    label_map_path = model_dir / "event_type_label_map.json"

    if not meaningful_model_path.exists():
        raise FileNotFoundError(f"missing model: {meaningful_model_path}")
    if not event_type_model_path.exists():
        raise FileNotFoundError(f"missing model: {event_type_model_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"missing label map: {label_map_path}")

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_label = list(label_map.get("id_to_label") or [])
    if not id_to_label:
        raise ValueError("event_type_label_map.json has empty id_to_label")
    label_to_id = {label: idx for idx, label in enumerate(id_to_label)}

    meaningful_booster = lgb.Booster(model_file=str(meaningful_model_path))
    meaningful_prob = np.asarray(meaningful_booster.predict(x_valid), dtype=float)
    meaningful_pred = (meaningful_prob >= meaningful_threshold).astype(int)
    meaningful_metrics = _safe_binary_metrics(y_valid_binary, meaningful_pred, meaningful_prob)

    event_type_rows = [
        row
        for row in valid_rows
        if int(row.get("is_meaningful_label") or 0) == 1
        and str(row.get("event_type_label") or "") in label_to_id
    ]
    if not event_type_rows:
        raise ValueError("no event_type-evaluable rows in validation split")

    x_event = build_feature_matrix(event_type_rows)
    y_event = np.asarray(
        [label_to_id[str(row.get("event_type_label"))] for row in event_type_rows],
        dtype=int,
    )

    event_type_booster = lgb.Booster(model_file=str(event_type_model_path))
    event_prob = np.asarray(event_type_booster.predict(x_event), dtype=float)
    if event_prob.ndim == 1:
        event_prob = np.vstack([1.0 - event_prob, event_prob]).T
    event_pred = np.argmax(event_prob, axis=1)
    event_metrics = _safe_multiclass_metrics(y_event, event_pred, event_prob, id_to_label)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset),
        "model_dir": str(model_dir),
        "timestamp_column": ts_col,
        "train_ratio": train_ratio,
        "validation_rows": len(valid_rows),
        "validation_binary_distribution": label_distribution(y_valid_binary.tolist()),
        "event_type_validation_rows": len(event_type_rows),
        "event_type_validation_distribution": label_distribution(y_event.tolist()),
        "meaningful_threshold": meaningful_threshold,
        "is_meaningful_metrics": meaningful_metrics,
        "event_type_metrics": event_metrics,
    }

    dump_json(out_json, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LightGBM models on time-based validation split."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--meaningful-threshold", type=float, default=0.5)
    parser.add_argument("--timestamp-column", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = evaluate_models(
        dataset=args.dataset,
        model_dir=args.model_dir,
        out_json=args.out_json,
        train_ratio=args.train_ratio,
        meaningful_threshold=args.meaningful_threshold,
        timestamp_column=args.timestamp_column,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
