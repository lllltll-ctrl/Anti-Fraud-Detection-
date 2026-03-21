from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PipelineConfig
from src.evaluation.metrics import binary_f1_score
from src.evaluation.validation import stratified_split_users
from src.feature_cache import load_or_build_feature_table
from src.data.loaders import load_users
from src.models.predict import predict_labels
from src.models.train import train_baseline_model
from src.submission.make_submission import build_submission


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sample_feature_table(
    feature_table: pd.DataFrame,
    target_column: str,
    max_negative_samples: int | None = None,
    max_positive_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_negative_samples is None and max_positive_samples is None:
        return feature_table

    sampled_parts = []
    for target_value, group in feature_table.groupby(target_column, group_keys=False):
        if int(target_value) == 1:
            limit = max_positive_samples
        else:
            limit = max_negative_samples
        if limit is None or len(group) <= limit:
            sampled_parts.append(group)
        else:
            sampled_parts.append(group.sample(n=limit, random_state=random_state))

    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_features(
    train_users_path: str | Path,
    train_transactions_path: str | Path,
    base_dir: str | Path,
    test_users_path: str | Path | None = None,
    test_transactions_path: str | Path | None = None,
) -> dict[str, Any]:
    config = PipelineConfig(base_dir=Path(base_dir))
    config.ensure_directories()

    train_feature_table = load_or_build_feature_table(
        users_path=train_users_path,
        transactions_path=train_transactions_path,
        cache_path=config.processed_dir / "train_features.csv",
        is_train=True,
    )

    result: dict[str, Any] = {
        "train_cache_path": str(config.processed_dir / "train_features.csv"),
        "train_rows": len(train_feature_table),
    }

    if test_users_path is not None and test_transactions_path is not None:
        test_feature_table = load_or_build_feature_table(
            users_path=test_users_path,
            transactions_path=test_transactions_path,
            cache_path=config.processed_dir / "test_features.csv",
            is_train=False,
        )
        result["test_cache_path"] = str(config.processed_dir / "test_features.csv")
        result["test_rows"] = len(test_feature_table)

    return result


def validate_from_cache(
    base_dir: str | Path,
    train_cache_path: str | Path | None = None,
    iterations: int = 400,
    fast_mode: bool = False,
    max_negative_samples: int | None = None,
    max_positive_samples: int | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    max_depth: int | None = None,
    min_child_samples: int | None = None,
) -> dict[str, Any]:
    config = PipelineConfig(base_dir=Path(base_dir))
    config.ensure_directories()
    active_train_cache_path = Path(train_cache_path) if train_cache_path is not None else config.processed_dir / "train_features.csv"

    train_feature_table = load_users(active_train_cache_path)
    train_feature_table = _sample_feature_table(
        train_feature_table,
        target_column="is_fraud",
        max_negative_samples=max_negative_samples,
        max_positive_samples=max_positive_samples,
    )

    train_split, valid_split = stratified_split_users(
        train_feature_table,
        target_column="is_fraud",
        valid_size=0.2,
        random_state=42,
    )

    model_artifact = train_baseline_model(
        train_split,
        target_column="is_fraud",
        iterations=iterations,
        fast_mode=fast_mode,
        validation_dataset=valid_split,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
    )
    valid_labels = predict_labels(model_artifact, valid_split.drop(columns=["is_fraud"]))
    validation_f1 = binary_f1_score(valid_split["is_fraud"].tolist(), valid_labels)

    model_path = config.models_dir / "baseline_model.json"
    metrics_path = config.reports_dir / "baseline_metrics.json"
    _write_json(model_path, model_artifact)
    _write_json(
        metrics_path,
        {
            "validation_f1": validation_f1,
            "threshold": model_artifact["threshold"],
            "feature_count": len(model_artifact["feature_columns"]),
            "training_iterations": model_artifact["training_iterations"],
            "best_iteration": model_artifact["best_iteration"],
            "fast_mode": model_artifact["fast_mode"],
            "learning_rate": model_artifact["learning_rate"],
            "num_leaves": model_artifact["num_leaves"],
            "max_depth": model_artifact["max_depth"],
            "min_child_samples": model_artifact["min_child_samples"],
        },
    )

    result = {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "validation_f1": validation_f1,
        "threshold": model_artifact["threshold"],
    }
    return result


def run_validation_pipeline(
    train_users_path: str | Path,
    train_transactions_path: str | Path,
    base_dir: str | Path,
    iterations: int = 400,
    fast_mode: bool = False,
    max_negative_samples: int | None = None,
    max_positive_samples: int | None = None,
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    max_depth: int | None = None,
    min_child_samples: int | None = None,
) -> dict[str, Any]:
    feature_result = build_features(
        train_users_path=train_users_path,
        train_transactions_path=train_transactions_path,
        base_dir=base_dir,
    )
    return validate_from_cache(
        base_dir=base_dir,
        train_cache_path=feature_result["train_cache_path"],
        iterations=iterations,
        fast_mode=fast_mode,
        max_negative_samples=max_negative_samples,
        max_positive_samples=max_positive_samples,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
    )


def run_training_pipeline(
    train_users_path: str | Path,
    train_transactions_path: str | Path,
    test_users_path: str | Path,
    test_transactions_path: str | Path,
    base_dir: str | Path,
) -> dict[str, Any]:
    feature_result = build_features(
        train_users_path=train_users_path,
        train_transactions_path=train_transactions_path,
        test_users_path=test_users_path,
        test_transactions_path=test_transactions_path,
        base_dir=base_dir,
    )
    result = validate_from_cache(base_dir=base_dir, train_cache_path=feature_result["train_cache_path"])
    config = PipelineConfig(base_dir=Path(base_dir))
    validation_f1 = result["validation_f1"]
    train_feature_table = load_users(feature_result["train_cache_path"])
    full_model_artifact = train_baseline_model(train_feature_table, iterations=400, fast_mode=False)
    full_model_artifact["threshold"] = result["threshold"]
    model_path = Path(result["model_path"])
    _write_json(model_path, full_model_artifact)

    test_users = load_users(test_users_path)
    test_dataset = load_users(feature_result["test_cache_path"])

    model_artifact = json.loads(model_path.read_text(encoding="utf-8"))

    submission = build_submission(test_users, predict_labels(model_artifact, test_dataset))
    submission_path = config.submissions_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    result["submission_path"] = str(submission_path)
    result["validation_f1"] = validation_f1
    return result
