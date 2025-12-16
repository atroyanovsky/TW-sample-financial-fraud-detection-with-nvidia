# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Fit feature transformers on training data."""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output


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
def fit_transformers(
    train_data: Input[Dataset],
    feature_transformer_artifact: Output[Artifact],
    one_hot_threshold: int = 8,
) -> dict:
    """Fit feature transformers on training data.

    Creates a ColumnTransformer that:
    - Binary encodes categorical columns with > one_hot_threshold categories
    - One-hot encodes categorical columns with <= one_hot_threshold categories
    - Robust scales numerical columns (Amount)

    The fitted transformer is saved for use in downstream components.

    Args:
        train_data: Input training dataset artifact
        feature_transformer_artifact: Output fitted transformer artifact
        one_hot_threshold: Max categories for one-hot encoding (else binary)

    Returns:
        Dict with transformer configuration
    """
    import pickle

    import pandas as pd
    from category_encoders import BinaryEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler

    # Column definitions
    COL_AMOUNT = "Amount"
    COL_FRAUD = "Fraud"
    COL_ERROR = "Errors"
    COL_CARD = "Card"
    COL_CHIP = "Chip"
    COL_CITY = "City"
    COL_ZIP = "Zip"
    COL_MCC = "MCC"
    COL_MERCHANT = "Merchant"

    # These are already binary-encoded in clean step, exclude from predictor columns
    MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_CARD, COL_MCC]

    numerical_predictors = [COL_AMOUNT]
    nominal_predictors = [
        COL_ERROR,
        COL_CARD,
        COL_CHIP,
        COL_CITY,
        COL_ZIP,
        COL_MCC,
        COL_MERCHANT,
    ]

    # Remove already-encoded columns
    nominal_predictors = [
        c for c in nominal_predictors if c not in MERCHANT_AND_USER_COLS
    ]
    predictor_columns = numerical_predictors + nominal_predictors

    # Load training data
    train_df = pd.read_parquet(train_data.path)
    pdf_training = train_df[predictor_columns + [COL_FRAUD]]

    print(f"Fitting transformers on {len(pdf_training):,} training records")
    print(f"Numerical predictors: {numerical_predictors}")
    print(f"Nominal predictors: {nominal_predictors}")

    # Determine encoding strategy per column
    columns_for_binary = []
    columns_for_onehot = []

    for col in nominal_predictors:
        n_unique = train_df[col].nunique()
        if n_unique <= one_hot_threshold:
            columns_for_onehot.append(col)
            print(f"  {col}: {n_unique} unique -> one-hot")
        else:
            columns_for_binary.append(col)
            print(f"  {col}: {n_unique} unique -> binary")

    # Mark categorical columns
    pdf_training[nominal_predictors] = pdf_training[nominal_predictors].astype(
        "category"
    )

    # Build transformer pipelines
    bin_encoder = Pipeline(
        steps=[
            ("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))
        ]
    )
    one_hot_encoder = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    robust_scaler = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("robust", RobustScaler()),
        ]
    )

    # Compose transformer
    transformers = []
    if columns_for_binary:
        transformers.append(("binary", bin_encoder, columns_for_binary))
    if columns_for_onehot:
        transformers.append(("onehot", one_hot_encoder, columns_for_onehot))
    transformers.append(("robust", robust_scaler, [COL_AMOUNT]))

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
    )

    # Fit transformer
    pd.set_option("future.no_silent_downcasting", True)
    transformer = transformer.fit(pdf_training[predictor_columns])

    # Get output column names
    output_columns = [
        name.split("__")[1] if "__" in name else name
        for name in transformer.get_feature_names_out(predictor_columns)
    ]

    # Determine column types
    type_mapping = {}
    for col in output_columns:
        base = col.split("_")[0]
        if base in nominal_predictors:
            type_mapping[col] = "int8"
        elif col in numerical_predictors or col == COL_AMOUNT:
            type_mapping[col] = "float"

    print(f"Output columns: {len(output_columns)}")

    # Save transformer
    with open(feature_transformer_artifact.path, "wb") as f:
        pickle.dump(
            {
                "transformer": transformer,
                "output_columns": output_columns,
                "type_mapping": type_mapping,
                "predictor_columns": predictor_columns,
                "nominal_predictors": nominal_predictors,
                "numerical_predictors": numerical_predictors,
            },
            f,
        )

    return {
        "num_output_columns": len(output_columns),
        "columns_binary_encoded": columns_for_binary,
        "columns_onehot_encoded": columns_for_onehot,
        "columns_scaled": [COL_AMOUNT],
    }
