from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd

from src.evaluation.metrics import binary_f1_score
from src.models.threshold import find_best_threshold


EXCLUDED_COLUMNS = {"id_user", "timestamp_reg", "email", "is_fraud"}


def _select_feature_columns(dataset: pd.DataFrame, target_column: str) -> list[str]:
    excluded = set(EXCLUDED_COLUMNS)
    excluded.add(target_column)
    return [column for column in dataset.columns if column not in excluded]


def _split_feature_types(dataset: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    for column in feature_columns:
        if (
            pd.api.types.is_object_dtype(dataset[column])
            or pd.api.types.is_string_dtype(dataset[column])
            or isinstance(dataset[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dataset[column])
        ):
            categorical_columns.append(column)
        else:
            numeric_columns.append(column)
    return categorical_columns, numeric_columns


def _prepare_training_matrix(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = dataset[feature_columns].copy()
    category_levels: dict[str, list[str]] = {}

    for column in categorical_columns:
        values = features[column].fillna("missing").astype(str)
        categories = sorted(values.unique().tolist())
        category_levels[column] = categories
        features[column] = pd.Categorical(values, categories=categories)

    numeric_fill_values: dict[str, float] = {}
    for column in numeric_columns:
        fill_value = float(features[column].median()) if not features[column].dropna().empty else 0.0
        numeric_fill_values[column] = fill_value
        features[column] = features[column].fillna(fill_value).astype(float)

    metadata = {
        "category_levels": category_levels,
        "numeric_fill_values": numeric_fill_values,
    }
    return features, metadata


def _prepare_inference_matrix(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    features = dataset[feature_columns].copy()
    categorical_set = set(categorical_columns)

    for column in feature_columns:
        if column in categorical_set:
            values = features[column].fillna("missing").astype(str)
            categories = metadata["category_levels"].get(column, [])
            features[column] = pd.Categorical(values, categories=categories)
        else:
            fill_value = float(metadata["numeric_fill_values"].get(column, 0.0))
            features[column] = features[column].fillna(fill_value).astype(float)

    return features


def train_baseline_model(
    dataset: pd.DataFrame,
    target_column: str = "is_fraud",
    iterations: int = 400,
    fast_mode: bool = False,
    validation_dataset: pd.DataFrame | None = None,
    early_stopping_rounds: int = 20,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    max_depth: int | None = None,
    min_child_samples: int | None = None,
) -> dict[str, Any]:
    if target_column not in dataset.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    active_iterations = min(iterations, 75) if fast_mode else iterations
    feature_columns = _select_feature_columns(dataset, target_column)
    if not feature_columns:
        raise ValueError("No feature columns available for training")

    categorical_columns, numeric_columns = _split_feature_types(dataset, feature_columns)
    features, metadata = _prepare_training_matrix(dataset, feature_columns, categorical_columns, numeric_columns)
    target = dataset[target_column].astype(int)

    positive_count = max(int(target.sum()), 1)
    negative_count = max(int((target == 0).sum()), 1)
    scale_pos_weight = negative_count / positive_count
    active_learning_rate = learning_rate if learning_rate is not None else (0.05 if not fast_mode else 0.08)
    active_num_leaves = num_leaves if num_leaves is not None else (31 if not fast_mode else 23)
    active_max_depth = max_depth if max_depth is not None else -1
    active_min_child_samples = min_child_samples if min_child_samples is not None else 50

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=active_iterations,
        learning_rate=active_learning_rate,
        num_leaves=active_num_leaves,
        max_depth=active_max_depth,
        min_child_samples=active_min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )

    fit_kwargs: dict[str, Any] = {"categorical_feature": categorical_columns}
    validation_target = None
    validation_features: pd.DataFrame | None = None
    validation_scores: list[float] | None = None
    train_class_count = int(target.nunique())
    if validation_dataset is not None and target_column in validation_dataset.columns:
        validation_target = validation_dataset[target_column].astype(int)
        validation_class_count = int(validation_target.nunique())
        if train_class_count > 1 and validation_class_count > 1:
            validation_features = _prepare_inference_matrix(
                validation_dataset,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                metadata=metadata,
            )
            fit_kwargs["eval_set"] = [(validation_features, validation_target)]
            fit_kwargs["eval_metric"] = "binary_logloss"
            fit_kwargs["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    model.fit(features, target, **fit_kwargs)

    best_iteration = model.best_iteration_ or active_iterations
    train_scores = model.predict_proba(features, num_iteration=best_iteration)[:, 1].tolist()
    train_labels = None
    validation_f1 = None

    if validation_features is not None and validation_target is not None:
        validation_scores = model.predict_proba(validation_features, num_iteration=best_iteration)[:, 1].tolist()
        assert validation_scores is not None
        threshold_info = find_best_threshold(validation_target.tolist(), validation_scores)
        validation_labels = [1 if score >= threshold_info["threshold"] else 0 for score in validation_scores]
        validation_f1 = binary_f1_score(validation_target.tolist(), validation_labels)
    else:
        threshold_info = find_best_threshold(target.tolist(), train_scores)

    train_labels = [1 if score >= threshold_info["threshold"] else 0 for score in train_scores]

    feature_processors = []
    for column in feature_columns:
        if column in categorical_columns:
            feature_processors.append(
                {
                    "type": "categorical",
                    "categories": metadata["category_levels"][column],
                }
            )
        else:
            feature_processors.append(
                {
                    "type": "numeric",
                    "fill_value": metadata["numeric_fill_values"][column],
                }
            )

    feature_importances = model.feature_importances_.tolist()

    return {
        "model_type": "lightgbm",
        "model_text": model.booster_.model_to_string(num_iteration=best_iteration),
        "feature_columns": feature_columns,
        "feature_weights": feature_importances,
        "feature_processors": feature_processors,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "category_levels": metadata["category_levels"],
        "numeric_fill_values": metadata["numeric_fill_values"],
        "threshold": threshold_info["threshold"],
        "global_rate": float(target.mean()),
        "train_f1": binary_f1_score(target.tolist(), train_labels),
        "validation_f1": validation_f1,
        "training_iterations": active_iterations,
        "best_iteration": best_iteration,
        "fast_mode": fast_mode,
        "learning_rate": active_learning_rate,
        "num_leaves": active_num_leaves,
        "max_depth": active_max_depth,
        "min_child_samples": active_min_child_samples,
        "feature_importances": {name: int(value) for name, value in zip(feature_columns, feature_importances)},
    }
