# Experimental design and interpretation

## Research objective

Compare K-Nearest Neighbors (KNN) and Decision Tree classifiers on an early-stage diabetes questionnaire classification dataset. This is a reproducible undergraduate machine learning experiment, **not a clinical prediction system**.

## Dataset and preprocessing

The [UCI Early Stage Diabetes Risk Prediction dataset](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset) has 520 questionnaire records, 16 predictor variables, and a binary outcome. The download script retrieves the source CSV; the source data are excluded from this repository.

The project removes exact duplicate rows **before splitting** into training and test sets. In the original experiment, this left 251 rows. See the generated [`results/data_audit.md`](../results/data_audit.md) for the precise counts and per-class distributions. The dataset has no patient identifiers accessible to this experiment. Two identical symptom profiles may belong to different people, so this preprocessing choice must not be misrepresented as verified patient-level deduplication.

Feature preparation occurs **inside each fitted scikit-learn pipeline**. Numeric age is median-imputed and standardized. Categorical columns are mode-imputed and one-hot encoded; unseen categories are ignored at prediction time. This prevents the preprocessing transformations from being fitted using the held-out data or cross-validation validation folds.

## Model selection and evaluation

A stratified, fixed-seed 80/20 split separates training and held-out test data. Hyperparameters are optimized with `GridSearchCV`, using 5-fold stratified cross-validation **only on training data**. Both models are scored with F1 for selection. The chosen model maximizes mean training-cross-validation F1 (not test-set F1).

The held-out test split is evaluated **once** after selection, with F1, ROC-AUC, balanced accuracy, precision, recall, and a confusion matrix. The test results are reported for both models for comparison, but are not used to choose the saved model.

## Observed results

| Model | CV F1 | Test F1 | Test ROC-AUC | Test balanced accuracy |
|---|---:|---:|---:|---:|
| KNN | 0.920 | 0.852 | 0.954 | 0.871 |
| Decision Tree | 0.923 | 0.886 | 0.912 | 0.818 |

The Decision Tree was selected on training CV F1. Its held-out F1 was higher; KNN's ROC-AUC and balanced accuracy were higher. This **does not** establish a universally best model.

## Threats to validity

- Only 51 records remained in the test partition after deduplication; a small test sample increases uncertainty.
- Exact-row deduplication is not equivalent to confirming independence of patients or observations.
- One random split is insufficient to characterize performance variability or external generalization.
- A single-hospital, symptom-based questionnaire dataset may contain selection bias and is not representative of screening populations.
- No clinical utility, real-world diagnostic accuracy, prospective risk estimate, or calibrated probability is claimed.

## Reproduction and scope

From the repository root, install `requirements.txt`, run the dataset download, then `python -m src.audit`, `python -m src.train`, and `python -m pytest -q`. The audit and training scripts produce text, JSON, metrics, and plots under `results/`.

The test suite includes generated synthetic examples for software functionality. **Synthetic test results are not used or described as medical dataset results.**
