from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd


def _prepare_inference_matrix(model_artifact: dict[str, Any], dataset: pd.DataFrame) -> pd.DataFrame:
    feature_columns = model_artifact["feature_columns"]
    categorical_columns = set(model_artifact.get("categorical_columns", []))
    category_levels = model_artifact.get("category_levels", {})
    numeric_fill_values = model_artifact.get("numeric_fill_values", {})

    features = dataset[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_columns:
            values = features[column].fillna("missing").astype(str)
            categories = category_levels.get(column, [])
            features[column] = pd.Categorical(values, categories=categories)
        else:
            fill_value = float(numeric_fill_values.get(column, 0.0))
            features[column] = features[column].fillna(fill_value).astype(float)

    return features


def predict_scores(model_artifact: dict[str, Any], dataset: pd.DataFrame) -> list[float]:
    feature_columns = model_artifact["feature_columns"]
    for column in feature_columns:
        if column not in dataset.columns:
            raise ValueError(f"Missing feature column '{column}' in dataset")

    model = lgb.Booster(model_str=model_artifact["model_text"])
    features = _prepare_inference_matrix(model_artifact, dataset)
    return model.predict(features).tolist()


def predict_labels(model_artifact: dict[str, Any], dataset: pd.DataFrame, threshold: float | None = None) -> list[int]:
    active_threshold = float(model_artifact["threshold"] if threshold is None else threshold)
    return [1 if score >= active_threshold else 0 for score in predict_scores(model_artifact, dataset)]
