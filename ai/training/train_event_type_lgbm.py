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
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ai.ml_features import FEATURE_COLUMNS, write_feature_spec
from ai.training.train_utils import (
    build_feature_matrix,
    dump_json,
    label_distribution,
    load_csv_rows,
    pick_timestamp_column,
    sort_rows_by_timestamp,
    time_split_rows,
)


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


def train_event_type_model(
    *,
    dataset: Path,
    out_dir: Path,
    classes: list[str] | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    early_stopping_rounds: int = 50,
    timestamp_column: str | None = None,
) -> dict[str, Any]:
    class_names = classes or ["daily_noise", "repeated_vibration"]
    label_to_id = {name: idx for idx, name in enumerate(class_names)}

    rows = load_csv_rows(dataset)
    if not rows:
        raise ValueError(f"empty dataset: {dataset}")

    filtered = [
        row
        for row in rows
        if int(row.get("is_meaningful_label") or 0) == 1
        and str(row.get("event_type_label") or "") in label_to_id
    ]
    if len(filtered) < 30:
        raise ValueError(
            f"not enough meaningful labeled rows for event_type training: {len(filtered)}"
        )

    ts_col = pick_timestamp_column(filtered, preferred=timestamp_column)
    sorted_rows = sort_rows_by_timestamp(filtered, ts_col)
    train_rows, valid_rows = time_split_rows(sorted_rows, train_ratio=train_ratio)

    x_train = build_feature_matrix(train_rows)
    y_train = np.asarray(
        [label_to_id[str(row.get("event_type_label"))] for row in train_rows],
        dtype=int,
    )
    x_valid = build_feature_matrix(valid_rows)
    y_valid = np.asarray(
        [label_to_id[str(row.get("event_type_label"))] for row in valid_rows],
        dtype=int,
    )

    train_unique = set(int(v) for v in y_train.tolist())
    required = set(range(len(class_names)))
    if train_unique != required:
        missing = sorted(required - train_unique)
        raise ValueError(
            "train split is missing one or more classes for event_type training. "
            f"missing={missing}, present={sorted(train_unique)}"
        )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(class_names),
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
    )

    y_prob = model.predict_proba(x_valid)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = _safe_multiclass_metrics(y_valid, y_pred, y_prob, class_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "event_type_model.txt"
    metrics_path = out_dir / "event_type_metrics.json"
    label_map_path = out_dir / "event_type_label_map.json"
    write_feature_spec(out_dir / "feature_spec.json")

    model.booster_.save_model(str(model_path))
    dump_json(label_map_path, {"label_to_id": label_to_id, "id_to_label": class_names})

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "event_type_multiclass_classification",
        "dataset_path": str(dataset),
        "timestamp_column": ts_col,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": class_names,
        "train_ratio": train_ratio,
        "seed": seed,
        "model_params": {
            "objective": "multiclass",
            "num_class": len(class_names),
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "class_weight": "balanced",
            "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        },
        "data_summary": {
            "total_rows_input": len(rows),
            "total_rows_filtered": len(sorted_rows),
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
            "label_map_json": str(label_map_path),
        },
    }
    dump_json(metrics_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model for event_type classification."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--classes",
        type=str,
        default="daily_noise,repeated_vibration",
        help="comma-separated class labels for training target",
    )
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
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    result = train_event_type_model(
        dataset=args.dataset,
        out_dir=args.out_dir,
        classes=classes,
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
