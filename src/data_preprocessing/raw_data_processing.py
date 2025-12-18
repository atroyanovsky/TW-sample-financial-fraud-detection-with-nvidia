"""
raw_data_processing.py
---------------------
Module for loading, cleaning, and encoding the raw TabFormer credit card transaction data.
"""

from typing import Dict, Any
import pyarrow  # Must import before cudf to avoid ArrowKeyError
import cudf
import pandas as pd
import numpy as np
import os
import scipy.stats as ss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders import BinaryEncoder
import sys

# Ensure src directory is in path for config import
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
    
from config.config import DataConfig, default_config


def cramers_v(x, y):
    """ "
    Compute correlation of categorical field x with target y.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r's_V
    """
    confusion_matrix = cudf.crosstab(x, y).to_numpy()
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

def load_and_clean_tabformer(
    base_path: str, 
    csv_name: str, 
    config: DataConfig = None
) -> Dict[str, Any]:
    """Load, clean, and encode the raw TabFormer credit card transaction data.
    
    Args:
        base_path: Directory containing 'raw' subfolder with CSV file
        csv_name: Name of the CSV file in the 'raw' subfolder
        config: Configuration object. If None, uses default config.
    """
    if config is None:
        config = default_config
        config.input_path = os.path.join(base_path, "raw", csv_name)
    """
    Load, clean, and encode the raw TabFormer credit card transaction data.

    Args:
        base_path (str): Directory containing 'raw' subfolder with CSV file.
        csv_name (str): Name of the CSV file in the 'raw' subfolder.
        under_sample (bool): Whether to under-sample non-fraud class. Default True.
        fraud_ratio (float): Desired fraud:non-fraud ratio if under-sampling. Default 0.1.

    Returns:
        dict: Contains cleaned DataFrame, fitted id_transformer, and metadata.
    """
    tabformer_raw_file_path = os.path.join(base_path, "raw", csv_name)
    data = pd.read_csv(tabformer_raw_file_path)
    # Rename columns using config mapping
    data.rename(columns=config.RAW_COLUMN_MAPPING, inplace=True)
    # Handle missing values
    assert config.UNKNOWN_STRING_MARKER not in set(data[config.COL_STATE].unique())
    assert config.UNKNOWN_STRING_MARKER not in set(data[config.COL_ERROR].unique())
    assert float(0) not in set(data[config.COL_ZIP].unique())
    assert 0 not in set(data[config.COL_ZIP].unique())
    data[config.COL_STATE] = data[config.COL_STATE].fillna(config.UNKNOWN_STRING_MARKER)
    data[config.COL_ERROR] = data[config.COL_ERROR].fillna(config.UNKNOWN_STRING_MARKER)
    data[config.COL_ZIP] = data[config.COL_ZIP].fillna(config.UNKNOWN_ZIP_CODE)
    assert data.isnull().sum().sum() == 0
    # Amount cleanup
    data[config.COL_AMOUNT] = (
        data[config.COL_AMOUNT].str.replace("$", "", regex=False).astype("float")
    )
    # Fraud label to int
    fraud_to_binary = {"No": 0, "Yes": 1}
    data[config.COL_FRAUD] = data[config.COL_FRAUD].map(fraud_to_binary).astype("int8")
    # Remove commas in error descriptions
    data[config.COL_ERROR] = data[config.COL_ERROR].str.replace(",", "")
    # Time conversion
    T = data[config.COL_TIME].str.split(":", expand=True)
    T[0] = T[0].astype("int32")
    T[1] = T[1].astype("int32")
    data[config.COL_TIME] = (T[0] * 60) + T[1]
    data[config.COL_TIME] = data[config.COL_TIME].astype("int32")
    del T
    # Merchant to string
    data[config.COL_MERCHANT] = data[config.COL_MERCHANT].astype("str")
    # Combine User and Card to unique Card ID
    max_nr_cards_per_user = len(data[config.COL_CARD].unique())
    data[config.COL_CARD] = data[config.COL_USER] * max_nr_cards_per_user + data[config.COL_CARD]
    data[config.COL_CARD] = data[config.COL_CARD].astype("int")
    # Prepare for ID encoding
    data_ids = pd.DataFrame()
    nr_unique_card = data[config.COL_CARD].unique().shape[0]
    nr_unique_merchant = data[config.COL_MERCHANT].unique().shape[0]
    nr_unique_mcc = data[config.COL_MCC].unique().shape[0]
    nr_elements = max(nr_unique_merchant, nr_unique_card)
    data_ids[config.COL_CARD] = [data[config.COL_CARD][0]] * nr_elements
    data_ids[config.COL_MERCHANT] = [data[config.COL_MERCHANT][0]] * nr_elements
    data_ids[config.COL_MCC] = [data[config.COL_MCC][0]] * nr_elements
    data_ids.loc[np.arange(nr_unique_card), config.COL_CARD] = data[config.COL_CARD].unique()
    data_ids.loc[np.arange(nr_unique_merchant), config.COL_MERCHANT] = data[
        config.COL_MERCHANT
    ].unique()
    data_ids.loc[np.arange(nr_unique_mcc), config.COL_MCC] = data[config.COL_MCC].unique()
    data_ids = data_ids[config.MERCHANT_AND_USER_COLS].astype("category")
    # Binary encoding for IDs
    id_bin_encoder = Pipeline(
        [("binary", BinaryEncoder(handle_missing="value", handle_unknown="value"))]
    )
    id_transformer = ColumnTransformer(
        [
            ("binary", id_bin_encoder, config.MERCHANT_AND_USER_COLS),
        ],
        remainder="passthrough",
    )
    pd.set_option("future.no_silent_downcasting", True)
    id_transformer = id_transformer.fit(data_ids)
    preprocessed_id_data_raw = id_transformer.transform(
        data[config.MERCHANT_AND_USER_COLS].astype("category")
    )
    columns_of_transformed_id_data = list(
        map(
            lambda name: name.split("__")[1],
            list(id_transformer.get_feature_names_out(config.MERCHANT_AND_USER_COLS)),
        )
    )
    id_col_type_mapping = {
        col: "int8"
        for col in columns_of_transformed_id_data
        if col.split("_")[0] in config.MERCHANT_AND_USER_COLS
    }
    assert data_ids.isnull().sum().sum() == 0
    preprocessed_id_data = pd.DataFrame(
        preprocessed_id_data_raw, columns=columns_of_transformed_id_data
    )
    data = pd.concat(
        [data.reset_index(drop=True), preprocessed_id_data.reset_index(drop=True)],
        axis=1,
    )
    del data_ids, preprocessed_id_data_raw
    # ##### Compute correlation of different fields with target
    sparse_factor = 1
    columns_to_compute_corr = [
        config.COL_CARD,
        config.COL_CHIP,
        config.COL_ERROR,
        config.COL_STATE,
        config.COL_CITY,
        config.COL_ZIP,
        config.COL_MCC,
        config.COL_MERCHANT,
        config.COL_USER,
        config.COL_DAY,
        config.COL_MONTH,
        config.COL_YEAR,
    ]
    for c1 in columns_to_compute_corr:
        for c2 in [config.COL_FRAUD]:
            coff = 100 * cramers_v(data[c1][::sparse_factor], data[c2][::sparse_factor])
            print("Correlation ({}, {}) = {:6.2f}%".format(c1, c2, coff))

    # ### Correlation of target with numerical columns
    for col in [config.COL_TIME, config.COL_AMOUNT]:
        r_pb, p_value = ss.pointbiserialr(
            # data[COL_FRAUD].to_pandas(), data[col].to_pandas()
            data[config.COL_FRAUD],
            data[col],
        )
        print("r_pb ({}) = {:3.2f} with p_value {:3.2f}".format(col, r_pb, p_value))

    nominal_predictors = [
        config.COL_ERROR,
        config.COL_CARD,
        config.COL_CHIP,
        config.COL_CITY,
        config.COL_ZIP,
        config.COL_MCC,
        config.COL_MERCHANT,
    ]
    # #### Remove duplicates non-fraud data points
    # Remove duplicates data points
    fraud_data = data[data[config.COL_FRAUD] == 1]
    data = data[data[config.COL_FRAUD] == 0]

    data = data.drop_duplicates(subset=nominal_predictors)
    data = pd.concat([data, fraud_data])

    # Optional: under-sample
    if config.under_sample:
        fraud_df = data[data[config.COL_FRAUD] == 1]
        non_fraud_df = data[data[config.COL_FRAUD] == 0]
        nr_non_fraud_samples = min(
            (len(data) - len(fraud_df)), int(len(fraud_df) / config.fraud_ratio)
        )
        data = pd.concat(
            [fraud_df, non_fraud_df.sample(nr_non_fraud_samples, random_state=config.random_seed)]
        )
    # Shuffle
    data = data.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    return {
        "data": data,
        "id_transformer": id_transformer,
        "columns_of_transformed_id_data": columns_of_transformed_id_data,
        "id_col_type_mapping": id_col_type_mapping,
        "raw_base_path": base_path,
        "csv_name": csv_name,
        "under_sample": config.under_sample,
        "fraud_ratio": config.fraud_ratio,
        "config": config,
    }
