"""
raw_data_processing.py
---------------------
Module for loading, cleaning, and encoding the IEEE-CIS Fraud Detection dataset.
This dataset comes from the Kaggle IEEE-CIS Fraud Detection competition.

The IEEE-CIS dataset has two main files:
- train_transaction.csv: Transaction data with features V1-V339, C1-C14, D1-D15, M1-M9
- train_identity.csv: Identity data with device info and id features

This module transforms the data into a format compatible with the XGBoost and GNN
data generation steps used in this fraud detection pipeline.
"""

from typing import Dict, Any, Optional
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
from datetime import datetime, timedelta

# Ensure src directory is in path for config import
_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from config.config import DataConfig, default_config


def cramers_v(x, y):
    """
    Compute correlation of categorical field x with target y.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r's_V
    """
    confusion_matrix = cudf.crosstab(x, y).to_numpy()
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


def create_ieee_cis_config() -> DataConfig:
    """Create a DataConfig customized for the IEEE-CIS dataset."""
    config = DataConfig(
        input_path="",
        output_dir="processed_data",
        test_split_year=2018,  # We'll use time-based split via TransactionDT
        under_sample=True,
        fraud_ratio=0.1,
        random_seed=42,
    )
    
    # Override column mappings for IEEE-CIS
    config.COL_USER = "card1"  # Primary card identifier
    config.COL_CARD = "Card"   # Combined card identifier
    config.COL_AMOUNT = "Amount"
    config.COL_MCC = "ProductCD"  # Product code as category
    config.COL_TIME = "Time"
    config.COL_DAY = "Day"
    config.COL_MONTH = "Month"
    config.COL_YEAR = "Year"
    config.COL_MERCHANT = "Merchant"  # Synthetic merchant from addr fields
    config.COL_STATE = "addr2"  # addr2 is often state/country
    config.COL_CITY = "addr1"   # addr1 is often city/region
    config.COL_ZIP = "dist1"    # dist1 as proxy for geographic zone
    config.COL_ERROR = "DeviceType"  # Device type as error proxy
    config.COL_CHIP = "card4"   # Card network as chip proxy
    config.COL_FRAUD = "Fraud"
    
    # Update column groups
    config.MERCHANT_AND_USER_COLS = [config.COL_MERCHANT, config.COL_CARD, config.COL_MCC]
    
    # No column renaming needed - we create columns with target names
    config.RAW_COLUMN_MAPPING = {}
    
    # Feature columns specific to IEEE-CIS
    config.categorical_features = [config.COL_ERROR, config.COL_CHIP, 
                                   config.COL_CITY, config.COL_ZIP]
    config.numerical_features = [config.COL_AMOUNT]
    config.id_columns = [config.COL_MERCHANT, config.COL_CARD, config.COL_MCC]
    
    return config


def load_and_clean_ieee_cis(
    base_path: str,
    transaction_csv: str = "train_transaction.csv",
    identity_csv: str = "train_identity.csv",
    config: Optional[DataConfig] = None,
    sample_fraction: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load, clean, and encode the IEEE-CIS Fraud Detection dataset.

    Args:
        base_path: Directory containing 'raw' subfolder with CSV files
        transaction_csv: Name of transaction CSV file in 'raw' subfolder
        identity_csv: Name of identity CSV file in 'raw' subfolder
        config: Configuration object. If None, creates IEEE-CIS specific config.
        sample_fraction: Optional fraction to sample (for faster testing). None = use all data.

    Returns:
        dict: Contains cleaned DataFrame, fitted id_transformer, and metadata.
    """
    if config is None:
        config = create_ieee_cis_config()
    
    config.input_path = os.path.join(base_path, "raw", transaction_csv)
    
    # Load transaction data
    transaction_path = os.path.join(base_path, "raw", transaction_csv)
    print(f"Loading transaction data from {transaction_path}...")
    
    # Read with appropriate dtypes to reduce memory
    dtype_dict = {f'V{i}': 'float32' for i in range(1, 340)}
    dtype_dict.update({f'C{i}': 'float32' for i in range(1, 15)})
    dtype_dict.update({f'D{i}': 'float32' for i in range(1, 16)})
    dtype_dict.update({f'M{i}': 'object' for i in range(1, 10)})
    dtype_dict['TransactionAmt'] = 'float32'
    dtype_dict['TransactionDT'] = 'int64'
    dtype_dict['isFraud'] = 'int8'
    
    transactions = pd.read_csv(transaction_path, dtype=dtype_dict)
    print(f"  Loaded {len(transactions)} transactions")
    
    # Load identity data
    identity_path = os.path.join(base_path, "raw", identity_csv)
    print(f"Loading identity data from {identity_path}...")
    identity = pd.read_csv(identity_path)
    print(f"  Loaded {len(identity)} identity records")
    
    # Merge transaction and identity data
    print("Merging transaction and identity data...")
    data = transactions.merge(identity, on='TransactionID', how='left')
    print(f"  Merged data has {len(data)} rows")
    
    del transactions, identity  # Free memory
    
    # Optional sampling for faster development
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"Sampling {sample_fraction*100:.1f}% of data...")
        data = data.sample(frac=sample_fraction, random_state=config.random_seed)
        data = data.reset_index(drop=True)
        print(f"  Sampled data has {len(data)} rows")
    
    # ========== Transform columns to expected schema ==========
    
    # Fraud label
    data[config.COL_FRAUD] = data['isFraud'].astype('int8')
    
    # Amount
    data[config.COL_AMOUNT] = data['TransactionAmt'].astype('float32')
    
    # Time conversion: TransactionDT is seconds from a reference point
    # We'll convert to Day, Month, Year, Time (minutes from midnight)
    print("Converting time features...")
    reference_date = datetime(2017, 11, 30)  # Approximate start date of dataset
    data['datetime'] = data['TransactionDT'].apply(
        lambda x: reference_date + timedelta(seconds=int(x))
    )
    data['RealYear'] = data['datetime'].dt.year.astype('int16')
    data[config.COL_MONTH] = data['datetime'].dt.month.astype('int8')
    data[config.COL_DAY] = data['datetime'].dt.day.astype('int8')
    data[config.COL_TIME] = (data['datetime'].dt.hour * 60 + data['datetime'].dt.minute).astype('int32')
    data.drop(columns=['datetime'], inplace=True)
    
    # Create synthetic Year column for train/val/test split
    # IEEE-CIS data spans ~6 months, so we map TransactionDT to "virtual years"
    # to be compatible with the year-based splitting in xgboost/gnn generation
    # Split: 60% train (Year<2018), 20% val (Year==2018), 20% test (Year>2018)
    print("Creating virtual year column for data splitting...")
    dt_min = data['TransactionDT'].min()
    dt_max = data['TransactionDT'].max()
    dt_range = dt_max - dt_min
    
    # Map to virtual years: 2017 (train), 2018 (val), 2019 (test)
    # 60% → 2017, 20% → 2018, 20% → 2019
    train_threshold = dt_min + 0.6 * dt_range
    val_threshold = dt_min + 0.8 * dt_range
    
    data[config.COL_YEAR] = 2017  # Default to training year
    data.loc[data['TransactionDT'] >= train_threshold, config.COL_YEAR] = 2018
    data.loc[data['TransactionDT'] >= val_threshold, config.COL_YEAR] = 2019
    data[config.COL_YEAR] = data[config.COL_YEAR].astype('int16')
    
    print(f"  Virtual year distribution:")
    print(f"    2017 (train): {(data[config.COL_YEAR] == 2017).sum()} samples")
    print(f"    2018 (val):   {(data[config.COL_YEAR] == 2018).sum()} samples")
    print(f"    2019 (test):  {(data[config.COL_YEAR] == 2019).sum()} samples")
    
    # Card identifier: combine card1, card2, card3 for unique card ID
    print("Creating card identifiers...")
    data['card2_filled'] = data['card2'].fillna(-1).astype('int32')
    data['card3_filled'] = data['card3'].fillna(-1).astype('int32')
    data['card1_int'] = data['card1'].astype('int32')
    
    # Create unique card ID
    max_card2 = data['card2_filled'].max() + 2
    max_card3 = data['card3_filled'].max() + 2
    data[config.COL_CARD] = (
        data['card1_int'] * max_card2 * max_card3 + 
        data['card2_filled'] * max_card3 + 
        data['card3_filled']
    ).astype('int64')
    
    # User column (same as card for this dataset)
    data[config.COL_USER] = data['card1'].astype('int32')
    
    # MCC / Product category
    data[config.COL_MCC] = data['ProductCD'].astype('str')
    
    # Merchant: Create synthetic merchant from addr1 + ProductCD
    # This gives us a merchant-like identifier for the graph structure
    print("Creating merchant identifiers...")
    data['addr1_filled'] = data['addr1'].fillna(-1).astype('int32')
    data['addr2_filled'] = data['addr2'].fillna(-1).astype('int32')
    data[config.COL_MERCHANT] = (
        data['ProductCD'].astype(str) + '_' + 
        data['addr1_filled'].astype(str) + '_' +
        data['addr2_filled'].astype(str)
    )
    
    # City (addr1) and State (addr2)
    data[config.COL_CITY] = data['addr1_filled'].astype('int32')
    data[config.COL_STATE] = data['addr2_filled'].astype('str')
    
    # Zip proxy (dist1)
    data[config.COL_ZIP] = data['dist1'].fillna(config.UNKNOWN_ZIP_CODE).astype('float32')
    
    # Error/Device type
    data[config.COL_ERROR] = data['DeviceType'].fillna(config.UNKNOWN_STRING_MARKER).astype('str')
    
    # Chip (card4 - card network)
    data[config.COL_CHIP] = data['card4'].fillna(config.UNKNOWN_STRING_MARKER).astype('str')
    
    # ========== Handle missing values in key columns ==========
    print("Handling missing values...")
    
    # Ensure no NaN in required columns
    for col in [config.COL_STATE, config.COL_ERROR, config.COL_CHIP]:
        if data[col].isna().any():
            data[col] = data[col].fillna(config.UNKNOWN_STRING_MARKER)
    
    # ========== Binary encoding for IDs ==========
    print("Setting up ID encoding...")
    
    # Prepare data for ID encoding
    data_ids = pd.DataFrame()
    nr_unique_card = data[config.COL_CARD].unique().shape[0]
    nr_unique_merchant = data[config.COL_MERCHANT].unique().shape[0]
    nr_unique_mcc = data[config.COL_MCC].unique().shape[0]
    nr_elements = max(nr_unique_merchant, nr_unique_card, nr_unique_mcc)
    
    print(f"  Unique cards: {nr_unique_card}")
    print(f"  Unique merchants: {nr_unique_merchant}")
    print(f"  Unique MCCs: {nr_unique_mcc}")
    
    # Initialize with default values
    data_ids[config.COL_CARD] = [data[config.COL_CARD].iloc[0]] * nr_elements
    data_ids[config.COL_MERCHANT] = [data[config.COL_MERCHANT].iloc[0]] * nr_elements
    data_ids[config.COL_MCC] = [data[config.COL_MCC].iloc[0]] * nr_elements
    
    # Fill with unique values
    data_ids.loc[np.arange(nr_unique_card), config.COL_CARD] = data[config.COL_CARD].unique()
    data_ids.loc[np.arange(nr_unique_merchant), config.COL_MERCHANT] = data[config.COL_MERCHANT].unique()
    data_ids.loc[np.arange(nr_unique_mcc), config.COL_MCC] = data[config.COL_MCC].unique()
    data_ids = data_ids[config.MERCHANT_AND_USER_COLS].astype('category')
    
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
    
    preprocessed_id_data = pd.DataFrame(
        preprocessed_id_data_raw, columns=columns_of_transformed_id_data
    )
    
    data = pd.concat(
        [data.reset_index(drop=True), preprocessed_id_data.reset_index(drop=True)],
        axis=1,
    )
    
    del data_ids, preprocessed_id_data_raw
    
    # ========== Compute correlation of different fields with target ==========
    print("\nComputing feature correlations with fraud target...")
    sparse_factor = max(1, len(data) // 100000)  # Sample for large datasets
    
    columns_to_compute_corr = [
        config.COL_CARD,
        config.COL_CHIP,
        config.COL_ERROR,
        config.COL_STATE,
        config.COL_CITY,
        config.COL_MCC,
        config.COL_MERCHANT,
        config.COL_USER,
        config.COL_DAY,
        config.COL_MONTH,
        config.COL_YEAR,
    ]
    
    for c1 in columns_to_compute_corr:
        if c1 in data.columns:
            try:
                coff = 100 * cramers_v(
                    cudf.Series(data[c1].values[::sparse_factor]), 
                    cudf.Series(data[config.COL_FRAUD].values[::sparse_factor])
                )
                print(f"Correlation ({c1}, {config.COL_FRAUD}) = {coff:6.2f}%")
            except Exception as e:
                print(f"Could not compute correlation for {c1}: {e}")
    
    # Correlation of target with numerical columns
    for col in [config.COL_TIME, config.COL_AMOUNT]:
        try:
            r_pb, p_value = ss.pointbiserialr(
                data[config.COL_FRAUD],
                data[col],
            )
            print(f"r_pb ({col}) = {r_pb:3.2f} with p_value {p_value:3.2f}")
        except Exception as e:
            print(f"Could not compute point-biserial for {col}: {e}")
    
    # ========== Remove duplicates and balance data ==========
    print("\nRemoving duplicate non-fraud data points...")
    nominal_predictors = [
        config.COL_ERROR,
        config.COL_CARD,
        config.COL_CHIP,
        config.COL_CITY,
        config.COL_MCC,
        config.COL_MERCHANT,
    ]
    
    # Keep fraud data separate
    fraud_data = data[data[config.COL_FRAUD] == 1]
    non_fraud_data = data[data[config.COL_FRAUD] == 0]
    
    # Remove duplicates from non-fraud
    non_fraud_data = non_fraud_data.drop_duplicates(subset=nominal_predictors)
    
    # Recombine
    data = pd.concat([non_fraud_data, fraud_data])
    print(f"  Data after deduplication: {len(data)} rows")
    print(f"  Fraud cases: {len(fraud_data)}, Non-fraud: {len(non_fraud_data)}")
    
    # Optional: under-sample
    if config.under_sample:
        print(f"\nUnder-sampling to fraud ratio: {config.fraud_ratio}")
        fraud_df = data[data[config.COL_FRAUD] == 1]
        non_fraud_df = data[data[config.COL_FRAUD] == 0]
        nr_non_fraud_samples = min(
            len(non_fraud_df), int(len(fraud_df) / config.fraud_ratio)
        )
        data = pd.concat(
            [fraud_df, non_fraud_df.sample(nr_non_fraud_samples, random_state=config.random_seed)]
        )
        print(f"  Data after under-sampling: {len(data)} rows")
    
    # Shuffle
    data = data.sample(frac=1, random_state=config.random_seed).reset_index(drop=True)
    
    print(f"\nFinal dataset shape: {data.shape}")
    print(f"Fraud rate: {data[config.COL_FRAUD].mean()*100:.2f}%")
    
    return {
        "data": data,
        "id_transformer": id_transformer,
        "columns_of_transformed_id_data": columns_of_transformed_id_data,
        "id_col_type_mapping": id_col_type_mapping,
        "raw_base_path": base_path,
        "csv_name": transaction_csv,
        "under_sample": config.under_sample,
        "fraud_ratio": config.fraud_ratio,
        "config": config,
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process IEEE-CIS Fraud Detection dataset")
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/IEEE_CIS",
        help="Path to IEEE_CIS data directory (containing 'raw' subfolder)"
    )
    parser.add_argument(
        "--sample", 
        type=float, 
        default=None,
        help="Fraction of data to sample (e.g., 0.1 for 10%%)"
    )
    parser.add_argument(
        "--no-undersample",
        action="store_true",
        help="Disable under-sampling of non-fraud class"
    )
    
    args = parser.parse_args()
    
    config = create_ieee_cis_config()
    if args.no_undersample:
        config.under_sample = False
    
    result = load_and_clean_ieee_cis(
        base_path=args.data_path,
        config=config,
        sample_fraction=args.sample
    )
    
    print("\n" + "="*50)
    print("Processing complete!")
    print(f"Output data shape: {result['data'].shape}")
    print(f"Columns: {list(result['data'].columns)}")
