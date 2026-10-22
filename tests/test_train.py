import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import CATEGORICAL_COLUMNS
from src.train import run_experiment


def test_end_to_end_on_synthetic_data(tmp_path: Path):
    """Smoke test using GENERATED data, never reported as a real model result."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(90):
        label = "Positive" if i % 2 else "Negative"
        row = {name: ("Yes" if rng.random() < (0.7 if i % 2 else 0.3) else "No")
               for name in CATEGORICAL_COLUMNS}
        row["gender"] = "Male" if i % 3 else "Female"
        row["age"] = int(rng.integers(20, 65))
        row["class"] = label
        rows.append(row)
    csv_path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out = tmp_path / "outputs"
    report = run_experiment(csv_path, out, cv_folds=2)
    assert report["selected_model_by_train_cv"] in {"KNN", "Decision Tree"}
    assert (out / "best_model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert (out / "metrics.csv").exists()
    assert (out / "figures" / "roc_comparison.png").exists()
    for name in ("knn", "decision_tree"):
        assert (out / "figures" / f"{name}_confusion_matrix.png").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert 0 <= metrics["models"]["KNN"]["holdout"]["f1"] <= 1
