from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ai.ml_features import FEATURE_COLUMNS, feature_vector_from_event, event_features_from_sensor_row

TIMESTAMP_CANDIDATES = ("sensor_timestamp", "timestamp", "received_at")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def parse_timestamp(value: str | None) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def pick_timestamp_column(
    rows: Sequence[Mapping[str, Any]],
    preferred: str | None = None,
) -> str:
    if preferred:
        return preferred

    for candidate in TIMESTAMP_CANDIDATES:
        if any((row.get(candidate) or "").strip() for row in rows):
            return candidate
    return TIMESTAMP_CANDIDATES[0]


def sort_rows_by_timestamp(
    rows: Sequence[Mapping[str, Any]],
    timestamp_col: str,
) -> list[dict[str, Any]]:
    def _key(row: Mapping[str, Any]) -> tuple[int, datetime]:
        parsed = parse_timestamp(str(row.get(timestamp_col) or ""))
        if parsed is None:
            return (1, datetime.max.replace(tzinfo=timezone.utc))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, parsed)

    return [dict(row) for row in sorted(rows, key=_key)]


def time_split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_ratio: float = 0.8,
    min_train_rows: int = 20,
    min_valid_rows: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.5 <= train_ratio < 1.0):
        raise ValueError("train_ratio must be in range [0.5, 1.0)")
    total = len(rows)
    if total < (min_train_rows + min_valid_rows):
        raise ValueError(
            f"not enough rows for time split: total={total}, "
            f"required>={min_train_rows + min_valid_rows}"
        )

    split_idx = int(total * train_ratio)
    split_idx = max(min_train_rows, split_idx)
    split_idx = min(total - min_valid_rows, split_idx)

    train_rows = [dict(row) for row in rows[:split_idx]]
    valid_rows = [dict(row) for row in rows[split_idx:]]
    if not train_rows or not valid_rows:
        raise ValueError("time split failed to create both train and valid sets")
    return train_rows, valid_rows


def build_feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    vectors: list[list[float]] = []
    for row in rows:
        event = event_features_from_sensor_row(row)
        vectors.append(feature_vector_from_event(event))
    if not vectors:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=float)
    return np.asarray(vectors, dtype=float)


def extract_labels(
    rows: Sequence[Mapping[str, Any]],
    label_fn: Callable[[Mapping[str, Any]], int],
) -> np.ndarray:
    return np.asarray([int(label_fn(row)) for row in rows], dtype=int)


def label_distribution(labels: Sequence[int]) -> dict[str, int]:
    counter = Counter(int(v) for v in labels)
    return {str(k): int(v) for k, v in sorted(counter.items())}
