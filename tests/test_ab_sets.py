"""Torch-free test for the A/B-set orchestrator's report builder."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_ab_sets  # noqa: E402


def test_build_report_table_and_json():
    results = [
        {"set": "original", "dir": "w/original", "images": 5, "masks": False},
        {"set": "luma", "dir": "w/luma", "images": 5, "masks": True,
         "mean_excluded_pct": 4.2, "warn_images": 0, "danger_images": 0},
        {"set": "reflectmask", "dir": "w/reflectmask", "images": 0,
         "skipped": "skipped (--skip-model)"},
    ]
    md, obj = make_ab_sets.build_report(Path("w"), Path("in"), results)
    # every set appears, plus the comparison guidance
    for name in ("original", "luma", "reflectmask"):
        assert f"`{name}`" in md
    assert "Enable masks for alignment" in md
    assert "ab_colmap.py" in md
    assert "skipped (--skip-model)" in md
    # json round-trips the raw set records
    assert obj["sets"] == results
    assert obj["input"] == "in" and obj["work"] == "w"
