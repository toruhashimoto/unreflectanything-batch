"""Per-image processing logs.

Writes, under ``<output>/logs/``:
  * ``process_log.jsonl`` — one JSON object per image (full detail).
  * ``process_log.csv``   — flat summary table (Excel-friendly).
  * ``errors.csv``        — only the failed images, for quick triage.
  * ``run_summary.json``  — run-level totals + configuration.

Every successful/failed record carries ``processed_by: "UnReflectAnything"`` so a
downstream consumer can always tell a frame was AI-processed (the task's explicit
requirement). No torch import here — pure stdlib.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESSED_BY = "UnReflectAnything"

# Columns for the flat CSV (kept stable for downstream tooling).
_CSV_FIELDS = [
    "timestamp",
    "status",
    "processed_by",
    "source",
    "output",
    "model",
    "model_version",
    "device",
    "input_w",
    "input_h",
    "output_w",
    "output_h",
    "duration_sec",
    "threshold",
    "dilation",
    "jpeg_quality",
    "mean_luma_before",
    "mean_luma_after",
    "mean_luma_delta",
    "highlight_ratio_before",
    "highlight_ratio_after",
    "highlight_ratio_delta",
    "mean_abs_diff",
    "error",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLogger:
    """Append-as-you-go logger so a crash mid-batch still leaves a usable log."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "process_log.jsonl"
        self.csv_path = self.log_dir / "process_log.csv"
        self.errors_path = self.log_dir / "errors.csv"
        self.summary_path = self.log_dir / "run_summary.json"

        self._counts = {"ok": 0, "skipped": 0, "error": 0}

        # Fresh files each run.
        self.jsonl_path.write_text("", encoding="utf-8")
        self._csv_fh = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv = csv.DictWriter(self._csv_fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        self._csv.writeheader()
        self._err_fh = self.errors_path.open("w", newline="", encoding="utf-8")
        self._err = csv.DictWriter(
            self._err_fh, fieldnames=["timestamp", "source", "error"], extrasaction="ignore"
        )
        self._err.writeheader()

    def log(self, record: dict[str, Any]) -> None:
        record.setdefault("timestamp", utc_now_iso())
        record.setdefault("processed_by", PROCESSED_BY)
        status = record.get("status", "ok")
        if status in self._counts:
            self._counts[status] += 1

        # JSONL (full record).
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Flatten nested fields for the CSV row.
        flat = dict(record)
        size_in = record.get("input_size") or [None, None]
        size_out = record.get("output_size") or [None, None]
        flat["input_w"], flat["input_h"] = size_in[0], size_in[1]
        flat["output_w"], flat["output_h"] = size_out[0], size_out[1]
        for k, v in (record.get("params") or {}).items():
            flat.setdefault(k, v)
        for k, v in (record.get("metrics") or {}).items():
            flat.setdefault(k, v)
        self._csv.writerow(flat)
        self._csv_fh.flush()

        if status == "error":
            self._err.writerow(
                {
                    "timestamp": flat["timestamp"],
                    "source": record.get("source", ""),
                    "error": record.get("error", ""),
                }
            )
            self._err_fh.flush()

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def finalize(self, summary: dict[str, Any]) -> None:
        summary = {**summary, "counts": self.counts, "finished_at": utc_now_iso()}
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            self._csv_fh.close()
            self._err_fh.close()
        except Exception:  # noqa: BLE001
            pass

    # Context-manager sugar so callers can guarantee files are flushed/closed.
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._csv_fh.close()
            self._err_fh.close()
        except Exception:  # noqa: BLE001
            pass
