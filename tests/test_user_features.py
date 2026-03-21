from src.features.user_features import build_user_features


def test_build_user_features_creates_email_and_time_columns(sample_users_df):
    features = build_user_features(sample_users_df)

    assert set(["id_user", "email_domain", "email_is_freemail", "reg_hour", "reg_weekday"]).issubset(features.columns)
    assert features.loc[features["id_user"] == 101, "email_domain"].item() == "gmail.com"
    assert features.loc[features["id_user"] == 101, "email_is_freemail"].item() == 1
    assert features.loc[features["id_user"] == 102, "reg_hour"].item() == 15


def test_build_user_features_preserves_one_row_per_user(sample_users_df):
    features = build_user_features(sample_users_df)

    assert len(features) == len(sample_users_df)
    assert features["id_user"].is_unique
