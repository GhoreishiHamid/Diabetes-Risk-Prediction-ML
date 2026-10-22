from pathlib import Path

import pandas as pd
import pytest

from src.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, load_data, normalize_column


def test_normalize_column():
    assert normalize_column(" Sudden Weight Loss ") == "sudden_weight_loss"
    assert normalize_column("class") == "class"


def test_load_data_normalizes_targets_and_deduplicates(tmp_path: Path):
    rows = []
    for i, label in enumerate(["Positive", "Negative", "Positive"]):
        row = {name: "Yes" for name in CATEGORICAL_COLUMNS}
        row["Gender"] = "Female"
        row.pop("gender", None)
        row.update({"Age": 35 + i, "class": label})
        rows.append(row)
    rows.append(rows[0].copy())
    file = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(file, index=False)
    X, y = load_data(file)
    assert len(X) == len(y) == 3
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.tolist()) == {0, 1}


def test_missing_dataset_gives_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_data(tmp_path / "missing.csv")
