"""Run a reproducible KNN vs Decision Tree comparison on UCI dataset 529.

Usage:
    python -m src.train --data data/raw/diabetes_data_upload.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.data import load_data
from src.modeling import model_specs


def evaluate(estimator, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = estimator.predict(X_test)
    probabilities = estimator.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ),
    }


def run_experiment(data_path: Path, output_dir: Path, seed: int = 42,
                   test_size: float = 0.2, cv_folds: int = 5) -> dict:
    X, y = load_data(data_path)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    if y_train.value_counts().min() < cv_folds:
        raise ValueError("Not enough samples of each class for cross-validation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    report = {
        "dataset": "UCI Early Stage Diabetes Risk Prediction, id=529",
        "seed": seed,
        "n_samples_after_exact_deduplication": len(X),
        "train_count": len(X_train),
        "test_count": len(X_test),
        "test_fraction": test_size,
        "cv_folds": cv_folds,
        "model_selection_metric": "train_cv_f1",
        "models": {},
    }
    summary_rows = []
    trained = {}

    for name, spec in model_specs(seed).items():
        search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["parameters"],
            scoring="f1",
            cv=cv,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        report["models"][name] = {
            "train_cv_f1": float(search.best_score_),
            "best_parameters": search.best_params_,
            "holdout": metrics,
        }
        summary_rows.append({
            "model": name,
            "train_cv_f1": float(search.best_score_),
            **{key: value for key, value in metrics.items()
               if isinstance(value, float)},
        })
        trained[name] = search.best_estimator_
        safe_name = name.lower().replace(" ", "_")
        cm_display = ConfusionMatrixDisplay(
            confusion_matrix=np.array(metrics["confusion_matrix"]),
            display_labels=["Negative", "Positive"],
        )
        cm_display.plot(cmap="Blues", values_format="d", colorbar=False)
        plt.title(f"{name} — held-out confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / f"{safe_name}_confusion_matrix.png", dpi=160)
        plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, estimator in trained.items():
        RocCurveDisplay.from_estimator(estimator, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC curves — held-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "roc_comparison.png", dpi=160)
    plt.close(fig)

    # Select from training CV ONLY, not from the test-set outcomes.
    selected_name = max(report["models"], key=lambda name: report["models"][name]["train_cv_f1"])
    report["selected_model_by_train_cv"] = selected_name
    joblib.dump(trained[selected_name], output_dir / "best_model.joblib")
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics.csv", index=False)
    report_rows = [
        "# Reproduced experiment results",
        "",
        "> Automatically generated by `python -m src.train` from the supplied CSV. "
        "Verify the input is the original UCI dataset before publishing these numbers.",
        "",
        f"- Dataset: {report['dataset']}",
        f"- Rows after exact duplicate removal: {len(X)}",
        f"- Training rows: {len(X_train)}; test rows: {len(X_test)}",
        f"- Random seed: {seed}; CV folds: {cv_folds}",
        "- Model selection: training-CV F1 (test set not used to choose model)",
        "",
        "| Model | Train CV F1 | Test F1 | Test ROC-AUC | Test balanced accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_rows.append(
            f"| {row['model']} | {row['train_cv_f1']:.3f} | {row['f1']:.3f} "
            f"| {row['roc_auc']:.3f} | {row['balanced_accuracy']:.3f} |"
        )
    report_rows.extend([
        "", f"**Model selected by training CV:** {selected_name}",
        "", "## Held-out evaluation plots", "",
        "![Held-out ROC comparison](figures/roc_comparison.png)", "",
        "![KNN confusion matrix](figures/knn_confusion_matrix.png)", "",
        "![Decision Tree confusion matrix](figures/decision_tree_confusion_matrix.png)",
        "", "## Limitations", "",
        "This single-site questionnaire dataset is small and not suitable for "
        "clinical diagnosis. Results are from one stratified hold-out split and "
        "should not be generalized to new populations.", "",
    ])
    (output_dir / "report.md").write_text("\n".join(report_rows), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/diabetes_data_upload.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    report = run_experiment(args.data, args.output, args.seed, args.test_size, args.cv_folds)
    print(f"Training records: {report['train_count']}; held-out records: {report['test_count']}")
    print(f"Model chosen by training CV: {report['selected_model_by_train_cv']}")
    for name, summary in report["models"].items():
        metrics = summary["holdout"]
        print(f"{name}: CV F1={summary['train_cv_f1']:.3f}, "
              f"test F1={metrics['f1']:.3f}, test ROC-AUC={metrics['roc_auc']:.3f}")
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
