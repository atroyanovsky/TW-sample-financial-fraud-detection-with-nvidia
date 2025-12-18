"""
xgboost_data_generation.py
-------------------------
Module for generating XGBoost-ready features and exporting train/validation/test splits from cleaned TabFormer data.
"""

from typing import Dict, Any, Optional, Tuple
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from category_encoders import BinaryEncoder

import sys

# Ensure src directory is in path for config import
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.config import DataConfig, default_config


def generate_xgboost_features(
    cleaned_data_bundle: Dict[str, Any],
    output_dir: Optional[str] = None,
    data_split_year: Optional[int] = None,
    config: Optional[DataConfig] = None
) -> Tuple[ColumnTransformer, list]:
    """Generate XGBoost features from cleaned data.
    
    Args:
        cleaned_data_bundle: Output from load_and_clean_tabformer
        output_dir: Directory to save processed data. If None, uses config.output_dir
        data_split_year: Year to split data. If None, uses config.test_split_year
        config: Configuration object. If None, uses config from cleaned_data_bundle or default
        
    Returns:
        Tuple of (transformer, feature_columns) for use in GNN generation
    """
    if config is None:
        config = cleaned_data_bundle.get('config', default_config)
        
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "xgboost")
        os.makedirs(output_dir, exist_ok=True)
        
    if data_split_year is None:
        data_split_year = config.test_split_year
    """
    Generate XGBoost-ready features and export train/validation/test splits to CSV.

    Args:
        cleaned_data_bundle (dict): Output from load_and_clean_tabformer.
        output_dir (str): Directory to write XGBoost CSV files.
        data_split_year (int): Year to split data into train/validation/test splits.
    """
    data = cleaned_data_bundle["data"]
    id_transformer = cleaned_data_bundle["id_transformer"]
    columns_of_transformed_id_data = cleaned_data_bundle[
        "columns_of_transformed_id_data"
    ]
    id_col_type_mapping = cleaned_data_bundle["id_col_type_mapping"]

    # Feature selection from config
    numerical_predictors = config.numerical_features
    nominal_predictors = config.categorical_features + [
        col for col in [config.COL_CARD, config.COL_MERCHANT, config.COL_MCC] 
        if col not in config.categorical_features
    ]
    predictor_columns = numerical_predictors + nominal_predictors
    target_column = [config.COL_FRAUD]

    # Remove ID columns from predictors (they are encoded separately)
    predictor_columns = list(set(predictor_columns) - set(config.MERCHANT_AND_USER_COLS))
    nominal_predictors = list(set(nominal_predictors) - set(config.MERCHANT_AND_USER_COLS))

    # Temporal split using config
    training_idx = data[config.COL_YEAR] < data_split_year
    validation_idx = data[config.COL_YEAR] == data_split_year
    test_idx = data[config.COL_YEAR] > data_split_year

    # Categorical encoding strategy
    columns_for_binary_encoding = []
    columns_for_one_hot_encoding = []
    for col in nominal_predictors:
        if len(data[col].unique()) <= 8:
            columns_for_one_hot_encoding.append(col)
        else:
            columns_for_binary_encoding.append(col)

    # Encoders and scalers
    bin_encoder = Pipeline(
        [("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))]
    )
    one_hot_encoder = Pipeline([("onehot", OneHotEncoder())])
    robust_scaler = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("robust", RobustScaler()),
        ]
    )
    transformer = ColumnTransformer(
        [
            ("binary", bin_encoder, columns_for_binary_encoding),
            ("onehot", one_hot_encoder, columns_for_one_hot_encoding),
            ("robust", robust_scaler, [config.COL_AMOUNT]),
        ],
        remainder="passthrough",
    )

    pdf_training = data[training_idx][predictor_columns + target_column].copy()
    # Mark categorical column as "category"
    pdf_training[nominal_predictors] = pdf_training[nominal_predictors].astype(
        "category"
    )

    pd.set_option("future.no_silent_downcasting", True)
    transformer = transformer.fit(pdf_training[predictor_columns])

    # transformed column names
    columns_of_transformed_txs = list(
        map(
            lambda name: name.split("__")[1],
            list(transformer.get_feature_names_out(predictor_columns)),
        )
    )
    
    type_mapping = {}
    for col in columns_of_transformed_txs:
        if col.split("_")[0] in nominal_predictors:
            type_mapping[col] = "int8"
        elif col in numerical_predictors:
            type_mapping[col] = "float"
        elif col in target_column:
            type_mapping[col] = data.dtypes.to_dict()[col]

    # Transform splits
    def transform_and_concat(idx, label_df):
        X = transformer.transform(data[idx][predictor_columns])
        X_df = pd.DataFrame(X, columns=columns_of_transformed_txs)
        id_df = pd.DataFrame(
            id_transformer.transform(data[idx][config.MERCHANT_AND_USER_COLS]),
            columns=columns_of_transformed_id_data,
        )
        out = pd.concat([X_df, id_df], axis=1)
        out[config.COL_FRAUD] = label_df[config.COL_FRAUD].values
        out = out.astype(type_mapping)
        return out

    train_df = transform_and_concat(training_idx, data[training_idx])
    val_df = transform_and_concat(validation_idx, data[validation_idx])
    test_df = transform_and_concat(test_idx, data[test_idx])
    # Output
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "training.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    # Save the transformer and feature info to the bundle for GNN
    cleaned_data_bundle.update({
        'xgb_transformer': transformer,
        'xgb_feature_columns': columns_of_transformed_txs,
        'output_dir': output_dir
    })

    return transformer, columns_of_transformed_txs
