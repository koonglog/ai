from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ai.ml_features import FEATURE_COLUMNS, write_feature_spec
from ai.training.train_utils import (
    build_feature_matrix,
    dump_json,
    extract_labels,
    label_distribution,
    load_csv_rows,
    pick_timestamp_column,
    sort_rows_by_timestamp,
    time_split_rows,
)


def _label_from_row(row: dict[str, Any]) -> int:
    return int(row.get("is_meaningful_label") or 0)


def _safe_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": None,
        "roc_auc": None,
    }
    unique_classes = set(int(v) for v in y_true.tolist())
    if len(unique_classes) >= 2:
        metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def train_is_meaningful_model(
    *,
    dataset: Path,
    out_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 50,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    rows = load_csv_rows(dataset)
    if not rows:
        raise ValueError(f"empty dataset: {dataset}")

    ts_col = pick_timestamp_column(rows, preferred=timestamp_column)
    sorted_rows = sort_rows_by_timestamp(rows, ts_col)
    train_rows, valid_rows = time_split_rows(sorted_rows, train_ratio=train_ratio)

    x_train = build_feature_matrix(train_rows)
    y_train = extract_labels(train_rows, _label_from_row)
    x_valid = build_feature_matrix(valid_rows)
    y_valid = extract_labels(valid_rows, _label_from_row)

    if len(set(y_train.tolist())) < 2:
        raise ValueError(
            "train split has only one class for is_meaningful_label. "
            "Collect more data or adjust split."
        )

    model = LGBMClassifier(
        objective="binary",
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
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
    )

    y_prob = model.predict_proba(x_valid)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _safe_binary_metrics(y_valid, y_pred, y_prob)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "is_meaningful_model.txt"
    metrics_path = out_dir / "is_meaningful_metrics.json"
    write_feature_spec(out_dir / "feature_spec.json")

    model.booster_.save_model(str(model_path))

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "is_meaningful_binary_classification",
        "dataset_path": str(dataset),
        "timestamp_column": ts_col,
        "feature_columns": FEATURE_COLUMNS,
        "train_ratio": train_ratio,
        "seed": seed,
        "model_params": {
            "objective": "binary",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "class_weight": "balanced",
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        },
        "data_summary": {
            "total_rows": len(sorted_rows),
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "train_label_distribution": label_distribution(y_train.tolist()),
            "valid_label_distribution": label_distribution(y_valid.tolist()),
        },
        "validation_metrics": metrics,
        "artifacts": {
            "model_txt": str(model_path),
            "metrics_json": str(metrics_path),
            "feature_spec_json": str(out_dir / "feature_spec.json"),
        },
    }

    dump_json(metrics_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model for is_meaningful binary classification."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--timestamp-column", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = train_is_meaningful_model(
        dataset=args.dataset,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        early_stopping_rounds=args.early_stopping_rounds,
        timestamp_column=args.timestamp_column,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
