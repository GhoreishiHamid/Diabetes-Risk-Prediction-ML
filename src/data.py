"""Dataset input, validation, and column normalization."""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


DATASET_COLUMNS = [
    "age", "gender", "polyuria", "polydipsia", "sudden_weight_loss",
    "weakness", "polyphagia", "genital_thrush", "visual_blurring", "itching",
    "irritability", "delayed_healing", "partial_paresis", "muscle_stiffness",
    "alopecia", "obesity", "class",
]
FEATURE_COLUMNS = [column for column in DATASET_COLUMNS if column != "class"]
NUMERIC_COLUMNS = ["age"]
CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column != "age"]


def normalize_column(name: str) -> str:
    """Normalize UCI CSV headers to stable snake_case names."""
    name = str(name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def load_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load UCI dataset, validate schema, and return predictors + binary label."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run `python scripts/download_data.py` "
            "or download diabetes_data_upload.csv from UCI dataset 529."
        )

    frame = pd.read_csv(path, na_values=["?", "", "NA", "N/A"])
    frame = frame.rename(columns=normalize_column)
    missing = sorted(set(DATASET_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"The CSV is missing expected columns: {missing}")

    # Exact duplicate rows can otherwise appear in both train/test splits.
    frame = frame[DATASET_COLUMNS].drop_duplicates().reset_index(drop=True)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip().str.lower()
        frame[column] = frame[column].replace({"": pd.NA, "?": pd.NA})

    labels = frame["class"].astype("string").str.strip().str.lower()
    mapped = labels.map({"positive": 1, "negative": 0})
    if mapped.isna().any():
        invalid = labels[mapped.isna()].unique().tolist()
        raise ValueError(f"Unexpected or missing class labels: {invalid}")
    if mapped.nunique() < 2:
        raise ValueError("Both positive and negative labels are needed.")
    return frame[FEATURE_COLUMNS].copy(), mapped.astype(int).rename("class")
