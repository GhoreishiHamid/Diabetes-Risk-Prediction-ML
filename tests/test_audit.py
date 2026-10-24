"""The audit should handle missing data and report the deduplication accurately."""
from pathlib import Path

import pandas as pd
import pytest

from src.audit import render_markdown, summarize_dataset
from src.data import CATEGORICAL_COLUMNS


def test_audit_counts_original_unique_and_classes(tmp_path: Path):
    positive = {column: "Yes" for column in CATEGORICAL_COLUMNS}
    positive.update({"age": 55, "class": "Positive"})
    negative = {column: "No" for column in CATEGORICAL_COLUMNS}
    negative.update({"age": 32, "class": "Negative"})
    csv = tmp_path / "data.csv"
    pd.DataFrame([positive, positive.copy(), negative]).to_csv(csv, index=False)

    audit = summarize_dataset(csv)
    assert audit["original_rows"] == 3
    assert audit["unique_rows"] == 2
    assert audit["exact_duplicate_rows"] == 1
    assert audit["raw_class_counts"] == {"negative": 1, "positive": 2}
    assert audit["deduplicated_class_counts"] == {"negative": 1, "positive": 1}
    assert "Exact duplicate rows removed" in render_markdown(audit)


def test_audit_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        summarize_dataset(tmp_path / "missing.csv")
