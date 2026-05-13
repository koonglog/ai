from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_timestamp(value: str | None) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    candidate = raw.replace("T", " ").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return raw
    return dt.isoformat(sep=" ")


def _build_join_key(sensor_id: str, ts: str) -> tuple[str, str]:
    return (sensor_id.strip(), _normalize_timestamp(ts))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_lgbm_dataset(
    *,
    sensor_csv: Path,
    noise_csv: Path,
    out_csv: Path,
    out_summary: Path | None = None,
) -> dict[str, Any]:
    if not sensor_csv.exists():
        raise FileNotFoundError(f"sensor csv not found: {sensor_csv}")
    if not noise_csv.exists():
        raise FileNotFoundError(f"noise csv not found: {noise_csv}")

    noise_index: dict[tuple[str, str], dict[str, str]] = {}
    noise_row_count = 0
    noise_rows_missing_key = 0

    with noise_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            noise_row_count += 1
            sensor_id = (row.get("sensor_id") or "").strip()
            started_at = row.get("started_at")
            key = _build_join_key(sensor_id, started_at or "")
            if not key[0] or not key[1]:
                noise_rows_missing_key += 1
                continue
            if key in noise_index:
                raise ValueError(
                    "duplicate noise join key detected: "
                    f"sensor_id={key[0]}, started_at={key[1]}"
                )
            noise_index[key] = row

    sensor_row_count = 0
    matched_noise_event_count = 0
    output_rows: list[dict[str, str]] = []

    with sensor_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        sensor_fields = list(reader.fieldnames or [])
        if not sensor_fields:
            raise ValueError("sensor csv has no header")

        for row in reader:
            sensor_row_count += 1
            sensor_id = (row.get("sensor_id") or "").strip()
            sensor_timestamp = row.get("sensor_timestamp") or ""
            key = _build_join_key(sensor_id, sensor_timestamp)
            matched = noise_index.get(key)
            if matched is not None:
                matched_noise_event_count += 1

            output = dict(row)
            output["is_meaningful_label"] = "1" if matched is not None else "0"
            output["event_type_label"] = (matched.get("event_type") if matched else "") or ""
            output["severity_label"] = (matched.get("severity") if matched else "") or ""
            output["noise_event_id"] = (matched.get("id") if matched else "") or ""
            output["noise_event_started_at"] = (
                (matched.get("started_at") if matched else "") or ""
            )
            output_rows.append(output)

    orphan_noise_event_count = len(noise_index) - matched_noise_event_count
    output_row_count = len(output_rows)
    positive_label_count = sum(
        1 for row in output_rows if row.get("is_meaningful_label") == "1"
    )

    out_fields = sensor_fields + [
        "is_meaningful_label",
        "event_type_label",
        "severity_label",
        "noise_event_id",
        "noise_event_started_at",
    ]

    _ensure_parent(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "join_key": {
            "sensor_table": ["sensor_id", "sensor_timestamp"],
            "noise_table": ["sensor_id", "started_at"],
        },
        "sensor_row_count": sensor_row_count,
        "noise_row_count": noise_row_count,
        "noise_rows_missing_key": noise_rows_missing_key,
        "output_row_count": output_row_count,
        "matched_noise_event_count": matched_noise_event_count,
        "positive_label_count": positive_label_count,
        "orphan_noise_event_count": orphan_noise_event_count,
        "integrity_checks": {
            "output_equals_sensor_rows": output_row_count == sensor_row_count,
            "positive_equals_matched": positive_label_count == matched_noise_event_count,
        },
        "paths": {
            "sensor_csv": str(sensor_csv),
            "noise_csv": str(noise_csv),
            "output_csv": str(out_csv),
        },
    }

    summary_path = out_summary
    if summary_path is not None:
        _ensure_parent(summary_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        summary["paths"]["summary_json"] = str(summary_path)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LightGBM training dataset by joining "
            "raw sensor readings and noise events."
        )
    )
    parser.add_argument("--sensor-csv", required=True, type=Path)
    parser.add_argument("--noise-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-summary", required=False, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_lgbm_dataset(
        sensor_csv=args.sensor_csv,
        noise_csv=args.noise_csv,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
