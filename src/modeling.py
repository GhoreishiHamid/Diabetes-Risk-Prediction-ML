"""Leakage-resistant scikit-learn pipelines and model search spaces."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])


def model_specs(seed: int = 42) -> dict:
    return {
        "KNN": {
            "estimator": Pipeline([
                ("preprocess", make_preprocessor()),
                ("classifier", KNeighborsClassifier()),
            ]),
            "parameters": {
                "classifier__n_neighbors": [3, 5, 7, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        },
        "Decision Tree": {
            "estimator": Pipeline([
                ("preprocess", make_preprocessor()),
                ("classifier", DecisionTreeClassifier(random_state=seed)),
            ]),
            "parameters": {
                "classifier__max_depth": [3, 5, 7, None],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    }
