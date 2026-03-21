from pathlib import Path

import json
import pandas as pd

from src.config import PipelineConfig
from src.pipeline import build_features, run_training_pipeline, run_validation_pipeline, validate_from_cache


def test_pipeline_config_creates_expected_output_paths(tmp_path: Path):
    config = PipelineConfig(base_dir=tmp_path)

    assert config.artifacts_dir == tmp_path / "artifacts"
    assert config.models_dir == tmp_path / "artifacts" / "models"
    assert config.submissions_dir == tmp_path / "artifacts" / "submissions"
    assert config.reports_dir == tmp_path / "artifacts" / "reports"
    assert config.processed_dir == tmp_path / "artifacts" / "processed"


def test_run_training_pipeline_writes_artifacts_and_submission(tmp_path: Path, sample_users_df, sample_test_users_df, sample_transactions_df):
    train_users_path = tmp_path / "train_users.csv"
    train_transactions_path = tmp_path / "train_transactions.csv"
    test_users_path = tmp_path / "test_users.csv"
    test_transactions_path = tmp_path / "test_transactions.csv"

    sample_users_df.to_csv(train_users_path, index=False)
    sample_transactions_df.to_csv(train_transactions_path, index=False)
    sample_test_users_df.to_csv(test_users_path, index=False)
    sample_transactions_df.iloc[:2].assign(id_user=[201, 201]).to_csv(test_transactions_path, index=False)

    result = run_training_pipeline(
        train_users_path=train_users_path,
        train_transactions_path=train_transactions_path,
        test_users_path=test_users_path,
        test_transactions_path=test_transactions_path,
        base_dir=tmp_path,
    )

    assert set(["model_path", "metrics_path", "submission_path", "validation_f1"]).issubset(result.keys())
    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert Path(result["submission_path"]).exists()

    metrics_payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert "validation_f1" in metrics_payload
    assert "threshold" in metrics_payload

    model_payload = json.loads(Path(result["model_path"]).read_text(encoding="utf-8"))
    assert "feature_processors" in model_payload
    assert len(model_payload["feature_columns"]) == len(model_payload["feature_processors"])

    submission_lines = Path(result["submission_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert submission_lines[0] == "id_user,is_fraud"
    assert len(submission_lines) == 3
    assert (tmp_path / "artifacts" / "processed" / "train_features.csv").exists()
    assert (tmp_path / "artifacts" / "processed" / "test_features.csv").exists()


def test_run_validation_pipeline_writes_metrics_without_submission_artifact(tmp_path: Path, sample_users_df, sample_transactions_df):
    train_users_path = tmp_path / "train_users.csv"
    train_transactions_path = tmp_path / "train_transactions.csv"

    sample_users_df.to_csv(train_users_path, index=False)
    sample_transactions_df.to_csv(train_transactions_path, index=False)

    result = run_validation_pipeline(
        train_users_path=train_users_path,
        train_transactions_path=train_transactions_path,
        base_dir=tmp_path,
    )

    assert set(["model_path", "metrics_path", "validation_f1"]).issubset(result.keys())
    assert "submission_path" not in result
    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()

    metrics_payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
    assert "validation_f1" in metrics_payload
    assert "threshold" in metrics_payload
    assert (tmp_path / "artifacts" / "processed" / "train_features.csv").exists()


def test_validate_from_cache_supports_fast_mode(tmp_path: Path, sample_users_df):
    config = PipelineConfig(base_dir=tmp_path)
    config.ensure_directories()
    cache_path = config.processed_dir / "train_features.csv"
    cached = sample_users_df.assign(email_domain="gmail.com", reg_hour=10, reg_weekday=2)
    for column in [
        "tx_count",
        "success_count",
        "fail_count",
        "amount_sum",
        "amount_mean",
        "unique_cards",
        "antifraud_count",
        "fraud_error_count",
        "fail_ratio",
        "card_country_mismatch_count",
        "payment_country_mismatch_count",
        "unique_card_countries",
        "unique_payment_countries",
        "max_users_per_card",
        "minutes_to_first_tx",
        "minutes_to_first_success",
        "tx_span_minutes",
        "mean_gap_minutes",
        "max_tx_per_day",
        "tx_in_first_24h",
        "fail_in_first_24h",
    ]:
        cached[column] = 0
    cached.to_csv(cache_path, index=False)

    result = validate_from_cache(base_dir=tmp_path, fast_mode=True, iterations=200)
    metrics_payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))

    assert metrics_payload["fast_mode"] is True
    assert metrics_payload["training_iterations"] == 75


def test_validate_from_cache_supports_sampling_controls(tmp_path: Path, sample_users_df):
    config = PipelineConfig(base_dir=tmp_path)
    config.ensure_directories()
    cache_path = config.processed_dir / "train_features.csv"
    cached = pd.concat([sample_users_df] * 20, ignore_index=True)
    cached["id_user"] = range(1, len(cached) + 1)
    cached = cached.assign(email_domain="gmail.com", email_is_freemail=1, reg_hour=10, reg_weekday=2)
    for column in [
        "tx_count",
        "success_count",
        "fail_count",
        "amount_sum",
        "amount_mean",
        "unique_cards",
        "antifraud_count",
        "fraud_error_count",
        "fraud_error_ratio",
        "has_prepaid_card",
        "fail_ratio",
        "card_country_mismatch_count",
        "payment_country_mismatch_count",
        "unique_card_countries",
        "unique_payment_countries",
        "max_users_per_card",
        "card_reuse_flag",
        "minutes_to_first_tx",
        "minutes_to_first_success",
        "tx_span_minutes",
        "mean_gap_minutes",
        "max_tx_per_day",
        "tx_in_first_24h",
        "fail_in_first_24h",
    ]:
        cached[column] = 0
    cached.to_csv(cache_path, index=False)

    result = validate_from_cache(base_dir=tmp_path, fast_mode=True, iterations=50, max_negative_samples=10, max_positive_samples=5)
    metrics_payload = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))

    assert metrics_payload["fast_mode"] is True
    assert Path(result["model_path"]).exists()


def test_build_features_writes_train_and_test_feature_tables(tmp_path: Path, sample_users_df, sample_test_users_df, sample_transactions_df):
    train_users_path = tmp_path / "train_users.csv"
    train_transactions_path = tmp_path / "train_transactions.csv"
    test_users_path = tmp_path / "test_users.csv"
    test_transactions_path = tmp_path / "test_transactions.csv"

    sample_users_df.to_csv(train_users_path, index=False)
    sample_transactions_df.to_csv(train_transactions_path, index=False)
    sample_test_users_df.to_csv(test_users_path, index=False)
    sample_transactions_df.iloc[:2].assign(id_user=[201, 201]).to_csv(test_transactions_path, index=False)

    result = build_features(
        train_users_path=train_users_path,
        train_transactions_path=train_transactions_path,
        test_users_path=test_users_path,
        test_transactions_path=test_transactions_path,
        base_dir=tmp_path,
    )

    assert Path(result["train_cache_path"]).exists()
    assert Path(result["test_cache_path"]).exists()
    assert result["train_rows"] == len(sample_users_df)
    assert result["test_rows"] == len(sample_test_users_df)


def test_validate_from_cache_reads_cached_train_table_only(tmp_path: Path, sample_users_df):
    config = PipelineConfig(base_dir=tmp_path)
    config.ensure_directories()
    cache_path = config.processed_dir / "train_features.csv"
    sample_users_df.assign(email_domain="gmail.com", reg_hour=10, reg_weekday=2).to_csv(cache_path, index=False)

    cached = pd.read_csv(cache_path)
    for column in [
        "tx_count",
        "success_count",
        "fail_count",
        "amount_sum",
        "amount_mean",
        "unique_cards",
        "antifraud_count",
        "fraud_error_count",
        "fail_ratio",
        "card_country_mismatch_count",
        "payment_country_mismatch_count",
        "unique_card_countries",
        "unique_payment_countries",
        "max_users_per_card",
        "minutes_to_first_tx",
        "minutes_to_first_success",
        "tx_span_minutes",
        "mean_gap_minutes",
        "max_tx_per_day",
        "tx_in_first_24h",
        "fail_in_first_24h",
    ]:
        if column not in cached.columns:
            cached[column] = 0
    cached.to_csv(cache_path, index=False)

    result = validate_from_cache(base_dir=tmp_path)

    assert Path(result["model_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert "submission_path" not in result
