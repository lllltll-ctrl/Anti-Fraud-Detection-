from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_user": [101, 102, 103],
            "timestamp_reg": [
                "2025-01-01 10:00:00+00:00",
                "2025-01-02 15:30:00+00:00",
                "2025-01-03 22:45:00+00:00",
            ],
            "email": [
                "alice@gmail.com",
                "bob@outlook.com",
                "carol@yahoo.com",
            ],
            "gender": ["female", "male", "female"],
            "reg_country": ["Portugal", "Brazil", "Germany"],
            "traffic_type": ["organic", "cpa", "ppc"],
            "is_fraud": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_test_users_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_user": [201, 202],
            "timestamp_reg": [
                "2025-02-01 08:00:00+00:00",
                "2025-02-02 19:15:00+00:00",
            ],
            "email": ["dave@gmail.com", "emma@hotmail.com"],
            "gender": ["male", "female"],
            "reg_country": ["Spain", "France"],
            "traffic_type": ["organic", "unknown"],
        }
    )


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_user": [101, 101, 102, 103, 103],
            "timestamp_tr": [
                "2025-01-01 11:00:00+00:00",
                "2025-01-01 12:00:00+00:00",
                "2025-01-02 16:00:00+00:00",
                "2025-01-03 23:00:00+00:00",
                "2025-01-04 00:00:00+00:00",
            ],
            "amount": [10.0, 20.0, 5.0, 7.5, 8.5],
            "status": ["success", "fail", "fail", "success", "fail"],
            "transaction_type": [
                "card_init",
                "google-pay",
                "card_init",
                "card_recurring",
                "card_recurring",
            ],
            "error_group": ["", "antifraud", "fraud", "", "do not honor"],
            "currency": ["EUR", "EUR", "USD", "EUR", "EUR"],
            "card_brand": ["VISA", "VISA", "MASTERCARD", "VISA", "VISA"],
            "card_type": ["DEBIT", "DEBIT", "DEBIT", "CREDIT", "CREDIT"],
            "card_country": ["Portugal", "Portugal", "Brazil", "Germany", "Germany"],
            "card_holder": ["alice", "alice", "bob", "carol", "carol"],
            "card_mask_hash": ["card_a", "card_a", "card_b", "card_c", "card_d"],
            "payment_country": ["Portugal", "Portugal", "Brazil", "Germany", "Germany"],
        }
    )


@pytest.fixture
def temp_csv_dir(tmp_path: Path) -> Path:
    return tmp_path
