import json

from src.logger import RunLogger, PROCESSED_BY


def test_logger_writes_all_files_and_counts(tmp_path):
    with RunLogger(tmp_path / "logs") as log:
        log.log({
            "status": "ok",
            "source": "a.jpg",
            "output": "out/a.jpg",
            "input_size": [100, 80],
            "output_size": [100, 80],
            "params": {"threshold": 0.3},
            "metrics": {"mean_luma_before": 100.0, "mean_luma_after": 90.0},
        })
        log.log({"status": "skipped", "source": "b.jpg"})
        log.log({"status": "error", "source": "c.jpg", "error": "boom"})
        assert log.counts == {"ok": 1, "skipped": 1, "error": 1}
        log.finalize({"model": "UnReflectAnything"})

    logs = tmp_path / "logs"
    assert (logs / "process_log.jsonl").exists()
    assert (logs / "process_log.csv").exists()
    assert (logs / "errors.csv").exists()
    assert (logs / "run_summary.json").exists()

    lines = (logs / "process_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["processed_by"] == PROCESSED_BY  # always stamped

    summary = json.loads((logs / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {"ok": 1, "skipped": 1, "error": 1}

    err_rows = (logs / "errors.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(err_rows) == 2  # header + one error
