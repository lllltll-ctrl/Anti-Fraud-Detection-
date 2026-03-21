from __future__ import annotations

import random

import pandas as pd


def stratified_split_users(
    users: pd.DataFrame,
    target_column: str,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1")

    rng = random.Random(random_state)
    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []

    for _, group in users.groupby(target_column):
        indices = list(group.index)
        rng.shuffle(indices)
        valid_count = max(1, int(round(len(indices) * valid_size)))
        valid_indices = set(indices[:valid_count])
        valid_parts.append(group.loc[list(valid_indices)])
        train_parts.append(group.loc[[index for index in indices if index not in valid_indices]])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return train_df, valid_df
