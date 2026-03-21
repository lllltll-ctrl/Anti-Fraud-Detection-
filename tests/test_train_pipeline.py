from src.features.build_dataset import build_model_dataset
from src.models.predict import predict_labels, predict_scores
from src.models.train import train_baseline_model


def test_train_baseline_model_returns_serializable_artifact(sample_users_df, sample_transactions_df):
    dataset = build_model_dataset(sample_users_df, sample_transactions_df, is_train=True)

    artifact = train_baseline_model(dataset, target_column="is_fraud")

    assert set(["feature_columns", "feature_weights", "threshold", "global_rate", "feature_processors"]).issubset(artifact.keys())
    assert len(artifact["feature_columns"]) > 0
    assert len(artifact["feature_columns"]) == len(artifact["feature_weights"])
    assert len(artifact["feature_columns"]) == len(artifact["feature_processors"])
    assert 0.0 <= artifact["threshold"] <= 1.0


def test_predict_scores_and_labels_match_input_length(sample_users_df, sample_transactions_df):
    train_dataset = build_model_dataset(sample_users_df, sample_transactions_df, is_train=True)
    test_dataset = build_model_dataset(sample_users_df.drop(columns=["is_fraud"]), sample_transactions_df, is_train=False)
    artifact = train_baseline_model(train_dataset, target_column="is_fraud")

    scores = predict_scores(artifact, test_dataset)
    labels = predict_labels(artifact, test_dataset)

    assert len(scores) == len(test_dataset)
    assert len(labels) == len(test_dataset)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert set(labels).issubset({0, 1})


def test_predict_scores_handles_unseen_categories(sample_users_df, sample_transactions_df):
    train_dataset = build_model_dataset(sample_users_df, sample_transactions_df, is_train=True)
    artifact = train_baseline_model(train_dataset, target_column="is_fraud")

    inference_dataset = train_dataset.drop(columns=["is_fraud"]).copy()
    inference_dataset.loc[:, "traffic_type"] = ["brand_new_channel"] * len(inference_dataset)
    inference_dataset.loc[:, "email_domain"] = ["newdomain.example"] * len(inference_dataset)

    scores = predict_scores(artifact, inference_dataset)

    assert len(scores) == len(inference_dataset)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_train_baseline_model_supports_fast_mode(sample_users_df, sample_transactions_df):
    dataset = build_model_dataset(sample_users_df, sample_transactions_df, is_train=True)

    artifact = train_baseline_model(dataset, target_column="is_fraud", iterations=200, fast_mode=True)

    assert artifact["training_iterations"] == 75
    assert artifact["fast_mode"] is True
