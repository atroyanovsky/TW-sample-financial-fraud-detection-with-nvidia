# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Clean and encode TabFormer data."""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "numpy==1.26.0",
        "category_encoders==2.6.0",
        "scikit-learn==1.4.0",
        "pyarrow==15.0.0",
    ],
)
def clean_and_encode_data(
    raw_data: Input[Dataset],
    cleaned_data: Output[Dataset],
    id_transformer_artifact: Output[Artifact],
    metrics: Output[Metrics],
    under_sample: bool = True,
    fraud_ratio: float = 0.1,
) -> dict:
    """Clean raw TabFormer data and encode identifiers.

    Performs:
    - Column renaming for cleaner access
    - Missing value handling (XX for strings, 0 for zip)
    - Amount field cleanup (remove $, convert to float)
    - Fraud label encoding (Yes->1, No->0)
    - Time parsing (HH:MM -> minutes since midnight)
    - Card ID combination (User * max_cards + Card)
    - Binary encoding of merchant/user/MCC identifiers
    - Optional undersampling of majority class
    - Duplicate removal from non-fraud transactions

    Args:
        raw_data: Input raw dataset artifact
        cleaned_data: Output cleaned dataset artifact
        id_transformer_artifact: Output fitted transformer for ID columns
        metrics: Output metrics artifact
        under_sample: Whether to undersample majority class
        fraud_ratio: Target fraud ratio when undersampling

    Returns:
        Dict with cleaning statistics and transformer column info
    """
    import pickle

    import numpy as np
    import pandas as pd
    from category_encoders import BinaryEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Constants
    COL_MERCHANT = "Merchant"
    COL_STATE = "State"
    COL_CITY = "City"
    COL_ZIP = "Zip"
    COL_ERROR = "Errors"
    COL_CHIP = "Chip"
    COL_FRAUD = "Fraud"
    COL_AMOUNT = "Amount"
    COL_TIME = "Time"
    COL_CARD = "Card"
    COL_USER = "User"
    COL_MCC = "MCC"

    UNKNOWN_STRING_MARKER = "XX"
    UNKNOWN_ZIP_CODE = 0
    MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_CARD, COL_MCC]

    COLUMN_RENAME_MAP = {
        "Merchant Name": COL_MERCHANT,
        "Merchant State": COL_STATE,
        "Merchant City": COL_CITY,
        "Errors?": COL_ERROR,
        "Use Chip": COL_CHIP,
        "Is Fraud?": COL_FRAUD,
    }

    # Load data
    data = pd.read_parquet(raw_data.path)
    original_count = len(data)
    print(f"Loaded {original_count:,} records")

    # Rename columns
    data.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Validate markers don't already exist
    assert UNKNOWN_STRING_MARKER not in set(data[COL_STATE].unique())
    assert UNKNOWN_STRING_MARKER not in set(data[COL_ERROR].unique())
    assert float(0) not in set(data[COL_ZIP].unique())
    assert 0 not in set(data[COL_ZIP].unique())

    # Handle missing values
    data[COL_STATE] = data[COL_STATE].fillna(UNKNOWN_STRING_MARKER)
    data[COL_ERROR] = data[COL_ERROR].fillna(UNKNOWN_STRING_MARKER)
    data[COL_ZIP] = data[COL_ZIP].fillna(UNKNOWN_ZIP_CODE)

    assert data.isnull().sum().sum() == 0, "Unexpected null values remain"

    # Clean Amount field
    data[COL_AMOUNT] = data[COL_AMOUNT].str.replace("$", "").astype("float")

    # Encode fraud label
    data[COL_FRAUD] = data[COL_FRAUD].map({"No": 0, "Yes": 1}).astype("int8")

    # Remove commas in error descriptions
    data[COL_ERROR] = data[COL_ERROR].str.replace(",", "")

    # Parse time to minutes since midnight
    time_parts = data[COL_TIME].str.split(":", expand=True)
    time_parts[0] = time_parts[0].astype("int32")
    time_parts[1] = time_parts[1].astype("int32")
    data[COL_TIME] = (time_parts[0] * 60) + time_parts[1]
    data[COL_TIME] = data[COL_TIME].astype("int32")

    # Convert Merchant to str
    data[COL_MERCHANT] = data[COL_MERCHANT].astype("str")

    # Combine User and Card for unique card IDs
    max_cards = len(data[COL_CARD].unique())
    data[COL_CARD] = (data[COL_USER] * max_cards + data[COL_CARD]).astype("int")

    # Fit binary encoder for ID columns
    nr_unique_card = data[COL_CARD].nunique()
    nr_unique_merchant = data[COL_MERCHANT].nunique()
    nr_unique_mcc = data[COL_MCC].nunique()
    nr_elements = max(nr_unique_merchant, nr_unique_card)

    # Create fitting dataframe with all unique values
    data_ids = pd.DataFrame(
        {
            COL_CARD: [data[COL_CARD].iloc[0]] * nr_elements,
            COL_MERCHANT: [data[COL_MERCHANT].iloc[0]] * nr_elements,
            COL_MCC: [data[COL_MCC].iloc[0]] * nr_elements,
        }
    )
    data_ids.loc[np.arange(nr_unique_card), COL_CARD] = data[COL_CARD].unique()
    data_ids.loc[np.arange(nr_unique_merchant), COL_MERCHANT] = data[
        COL_MERCHANT
    ].unique()
    data_ids.loc[np.arange(nr_unique_mcc), COL_MCC] = data[COL_MCC].unique()
    data_ids = data_ids[MERCHANT_AND_USER_COLS].astype("category")

    id_bin_encoder = Pipeline(
        steps=[
            ("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))
        ]
    )
    id_transformer = ColumnTransformer(
        transformers=[("binary", id_bin_encoder, MERCHANT_AND_USER_COLS)],
        remainder="passthrough",
    )

    pd.set_option("future.no_silent_downcasting", True)
    id_transformer = id_transformer.fit(data_ids)

    # Transform ID columns
    preprocessed_id_data = id_transformer.transform(
        data[MERCHANT_AND_USER_COLS].astype("category")
    )

    id_columns = [
        name.split("__")[1]
        for name in id_transformer.get_feature_names_out(MERCHANT_AND_USER_COLS)
    ]

    preprocessed_id_df = pd.DataFrame(preprocessed_id_data, columns=id_columns)
    data = pd.concat(
        [data.reset_index(drop=True), preprocessed_id_df.reset_index(drop=True)], axis=1
    )

    # Remove duplicate non-fraud transactions
    nominal_predictors = [
        COL_ERROR,
        COL_CARD,
        COL_CHIP,
        COL_CITY,
        COL_ZIP,
        COL_MCC,
        COL_MERCHANT,
    ]
    fraud_data = data[data[COL_FRAUD] == 1]
    non_fraud_data = data[data[COL_FRAUD] == 0].drop_duplicates(
        subset=nominal_predictors
    )
    data = pd.concat([non_fraud_data, fraud_data])

    after_dedup_count = len(data)
    print(f"After dedup: {after_dedup_count:,} records")

    # Undersample majority class
    if under_sample:
        fraud_df = data[data[COL_FRAUD] == 1]
        non_fraud_df = data[data[COL_FRAUD] == 0]
        nr_non_fraud = min(len(non_fraud_df), int(len(fraud_df) / fraud_ratio))
        data = pd.concat([fraud_df, non_fraud_df.sample(nr_non_fraud, random_state=42)])

    final_count = len(data)
    fraud_count = data[COL_FRAUD].sum()

    # Shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(
        f"Final dataset: {final_count:,} records, {fraud_count:,} fraud ({100 * fraud_count / final_count:.2f}%)"
    )

    # Save outputs
    data.to_parquet(cleaned_data.path, index=False)

    with open(id_transformer_artifact.path, "wb") as f:
        pickle.dump(
            {
                "transformer": id_transformer,
                "columns": id_columns,
                "merchant_user_cols": MERCHANT_AND_USER_COLS,
            },
            f,
        )

    # Log metrics
    metrics.log_metric("original_records", original_count)
    metrics.log_metric("after_dedup_records", after_dedup_count)
    metrics.log_metric("final_records", final_count)
    metrics.log_metric("fraud_records", int(fraud_count))
    metrics.log_metric("fraud_rate", float(fraud_count / final_count))

    return {
        "original_records": original_count,
        "final_records": final_count,
        "fraud_records": int(fraud_count),
        "fraud_rate": float(fraud_count / final_count),
        "id_columns": id_columns,
    }
