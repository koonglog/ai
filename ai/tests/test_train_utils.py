from __future__ import annotations

from ai.training.train_utils import (
    label_distribution,
    pick_timestamp_column,
    sort_rows_by_timestamp,
    time_split_rows,
)


def test_pick_timestamp_column_prefers_sensor_timestamp() -> None:
    rows = [
        {"sensor_timestamp": "", "timestamp": "2026-05-13T10:00:00+09:00"},
        {"sensor_timestamp": "2026-05-13T10:01:00+09:00", "timestamp": ""},
    ]
    assert pick_timestamp_column(rows) == "sensor_timestamp"


def test_sort_rows_by_timestamp_orders_ascending() -> None:
    rows = [
        {"sensor_timestamp": "2026-05-13 12:10:00.000000", "value": "b"},
        {"sensor_timestamp": "2026-05-13 12:00:00.000000", "value": "a"},
        {"sensor_timestamp": "2026-05-13 12:20:00.000000", "value": "c"},
    ]
    sorted_rows = sort_rows_by_timestamp(rows, "sensor_timestamp")
    assert [row["value"] for row in sorted_rows] == ["a", "b", "c"]


def test_time_split_rows_uses_ratio_and_minimums() -> None:
    rows = [{"sensor_timestamp": f"2026-05-13 12:{i:02d}:00.000000"} for i in range(40)]
    train_rows, valid_rows = time_split_rows(
        rows,
        train_ratio=0.75,
        min_train_rows=20,
        min_valid_rows=10,
    )
    assert len(train_rows) == 30
    assert len(valid_rows) == 10


def test_label_distribution_returns_string_keys() -> None:
    dist = label_distribution([1, 1, 0, 2, 2, 2])
    assert dist == {"0": 1, "1": 2, "2": 3}
