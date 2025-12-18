"""
gnn_data_generation.py
---------------------
Module for generating GNN graph data for link prediction on a bipartite User-Merchant graph.
Transactions are represented as edges between users and merchants.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import numpy as np
import pandas as pd
import pyarrow  # Must import before cudf to avoid ArrowKeyError
import cudf
import sys

# Ensure src directory is in path for config import
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
    
from config.config import DataConfig, default_config


def create_feature_mask(columns, start_mask_id=0):
    """Create a feature mask mapping columns to their base feature groups.
    
    Args:
        columns: List of column names
        start_mask_id: Starting mask ID value (for chaining multiple masks)
        
    Returns:
        Tuple of (mask_mapping dict, feature_mask array)
    """
    mask_mapping = {}
    mask_values = []
    current_mask = start_mask_id

    for col in columns:
        # For encoded columns, assume the base is before the underscore
        if "_" in col:
            base_feature = col.split("_")[0]
        else:
            base_feature = col  # For non-encoded columns, use the column name directly

        # Assign a new mask value if this base feature hasn't been seen before
        if base_feature not in mask_mapping:
            mask_mapping[base_feature] = current_mask
            current_mask += 1

        # Append the mask value for this column
        mask_values.append(mask_mapping[base_feature])

    # Convert list to numpy array for further processing if needed
    feature_mask = np.array(mask_values)

    return mask_mapping, feature_mask


def generate_gnn_graph_data(
    cleaned_data_bundle: Dict[str, Any],
    output_dir: Optional[str] = None,
    data_split_year: Optional[int] = None,
    xgb_transformer=None,
    columns_of_transformed_txs: Optional[List[str]] = None,
    config: Optional[DataConfig] = None
) -> Tuple[Dict, Dict, Dict]:
    """Generate GNN graph data for link prediction on a bipartite User-Merchant graph.
    
    In the link prediction approach:
    - Users and Merchants are nodes with their own features
    - Transactions are edges between users and merchants
    - Edge features are the transaction attributes
    - Edge labels are the fraud indicators
    
    Args:
        cleaned_data_bundle: Output from load_and_clean_tabformer
        output_dir: Directory to save graph data. If None, uses config.output_dir
        data_split_year: Year to split data. If None, uses config.test_split_year
        xgb_transformer: Transformer from XGBoost feature generation
        columns_of_transformed_txs: Feature columns from XGBoost (transaction features)
        config: Configuration object. If None, uses config from cleaned_data_bundle or default
        
    Returns:
        Tuple of (user_mask_map, merchant_mask_map, transaction_mask_map)
    """
    if config is None:
        config = cleaned_data_bundle.get('config', default_config)
        
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "gnn")
        os.makedirs(output_dir, exist_ok=True)
        
    if data_split_year is None:
        data_split_year = config.test_split_year
        
    # Get transformer and feature columns from XGBoost if not provided
    if xgb_transformer is None:
        xgb_transformer = cleaned_data_bundle.get('xgb_transformer')
    if columns_of_transformed_txs is None:
        columns_of_transformed_txs = cleaned_data_bundle.get('xgb_feature_columns', [])

    data = cleaned_data_bundle["data"]
    id_transformer = cleaned_data_bundle["id_transformer"]
    columns_of_transformed_id_data = cleaned_data_bundle["columns_of_transformed_id_data"]
    
    # Build type mapping for transaction features
    numerical_predictors = [config.COL_AMOUNT]
    nominal_predictors = [
        config.COL_ERROR, config.COL_CARD, config.COL_CHIP, 
        config.COL_CITY, config.COL_ZIP, config.COL_MCC, config.COL_MERCHANT
    ]
    predictor_columns = list(set(nominal_predictors) - set(config.MERCHANT_AND_USER_COLS)) + numerical_predictors
    
    type_mapping = {}
    for col in columns_of_transformed_txs:
        if col.split("_")[0] in nominal_predictors:
            type_mapping[col] = "int8"
        elif col in numerical_predictors:
            type_mapping[col] = "float"
    
    # Split data using config
    training_idx = data[config.COL_YEAR] < data_split_year
    validation_idx = data[config.COL_YEAR] == data_split_year
    test_idx = data[config.COL_YEAR] > data_split_year

    # Separate user and merchant feature columns
    user_feature_columns = [c for c in columns_of_transformed_id_data if c.startswith("Card")]
    mx_feature_columns = [c for c in columns_of_transformed_id_data if not c.startswith("Card")]

    def process_gnn_split(sub_data, out_dir, is_test=False):
        """Process a data split for bipartite link prediction GNN."""
        sub_data = sub_data.copy().reset_index(drop=True)
        sub_data[config.COL_TRANSACTION_ID] = sub_data.index
        
        # Create merchant ID mapping
        merchant_name_to_id = dict(
            zip(sub_data[config.COL_MERCHANT].unique(), 
                np.arange(len(sub_data[config.COL_MERCHANT].unique())))
        )
        sub_data[config.COL_MERCHANT_ID] = sub_data[config.COL_MERCHANT].map(merchant_name_to_id)
        
        # Create user ID mapping
        id_to_consecutive_id = dict(
            zip(sub_data[config.COL_CARD].unique(), 
                np.arange(len(sub_data[config.COL_CARD].unique())))
        )
        sub_data[config.COL_USER_ID] = sub_data[config.COL_CARD].map(id_to_consecutive_id)
        
        # Print ID ranges
        id_range = sub_data[config.COL_TRANSACTION_ID].min(), sub_data[config.COL_TRANSACTION_ID].max()
        print(f"Transaction ID range {id_range}")
        id_range = sub_data[config.COL_MERCHANT_ID].min(), sub_data[config.COL_MERCHANT_ID].max()
        print(f"Merchant ID range {id_range}")
        id_range = sub_data[config.COL_USER_ID].min(), sub_data[config.COL_USER_ID].max()
        print(f"User ID range {id_range}")
        
        # ---- Create Edges (User to Merchant) in COO format ----
        U_2_M = cudf.DataFrame()
        U_2_M[config.COL_GRAPH_SRC] = sub_data[config.COL_USER_ID]
        U_2_M[config.COL_GRAPH_DST] = sub_data[config.COL_MERCHANT_ID]
        Edge = cudf.concat([U_2_M])
        
        os.makedirs(os.path.join(out_dir, "edges"), exist_ok=True)
        out_path = os.path.join(out_dir, "edges/user_to_merchant.csv")
        Edge.to_csv(out_path, header=True, index=False)
        
        # ---- Transaction features (become edge attributes) ----
        transaction_feature_df = pd.DataFrame(
            xgb_transformer.transform(sub_data[predictor_columns]),
            columns=columns_of_transformed_txs,
        ).astype(type_mapping)
        transaction_feature_df[config.COL_FRAUD] = sub_data[config.COL_FRAUD]
        
        # ---- Node features ----
        # Merchant features
        data_merchant = sub_data[[config.COL_MERCHANT, config.COL_MCC, config.COL_CARD]].drop_duplicates(
            subset=[config.COL_MERCHANT]
        )
        data_merchant[config.COL_MERCHANT_ID] = data_merchant[config.COL_MERCHANT].map(merchant_name_to_id)
        data_merchant_sorted = data_merchant.sort_values(by=config.COL_MERCHANT_ID)
        
        # User features
        data_user = sub_data[[config.COL_MERCHANT, config.COL_MCC, config.COL_CARD]].drop_duplicates(
            subset=[config.COL_CARD]
        )
        data_user[config.COL_USER_ID] = data_user[config.COL_CARD].map(id_to_consecutive_id)
        data_user_sorted = data_user.sort_values(by=config.COL_USER_ID)
        
        preprocessed_merchant_data = pd.DataFrame(
            id_transformer.transform(data_merchant_sorted[config.MERCHANT_AND_USER_COLS]),
            columns=columns_of_transformed_id_data,
        )[mx_feature_columns]
        
        preprocessed_user_data = pd.DataFrame(
            id_transformer.transform(data_user_sorted[config.MERCHANT_AND_USER_COLS]),
            columns=columns_of_transformed_id_data,
        )[user_feature_columns]
        
        # ---- Write node features ----
        os.makedirs(os.path.join(out_dir, "nodes"), exist_ok=True)
        
        # User features
        out_path = os.path.join(out_dir, "nodes/user.csv")
        preprocessed_user_data.to_csv(out_path, header=True, index=False, columns=user_feature_columns)
        
        # Merchant features
        out_path = os.path.join(out_dir, "nodes/merchant.csv")
        preprocessed_merchant_data.to_csv(out_path, header=True, index=False, columns=mx_feature_columns)
        
        # ---- Write edge labels (fraud indicator) ----
        out_path = os.path.join(out_dir, "edges/user_to_merchant_label.csv")
        transaction_feature_df[[config.COL_FRAUD]].to_csv(
            out_path, header=True, index=False, columns=[config.COL_FRAUD]
        )
        
        # ---- Write edge attributes (transaction features) ----
        out_path = os.path.join(out_dir, "edges/user_to_merchant_attr.csv")
        transaction_feature_df[columns_of_transformed_txs].to_csv(
            out_path, header=True, index=False, columns=columns_of_transformed_txs
        )
        
        return user_feature_columns, mx_feature_columns, columns_of_transformed_txs

    # ---- Process train+val data ----
    data_all = data.copy()
    train_val_data = pd.concat([data[training_idx], data[validation_idx]])
    train_val_data.reset_index(inplace=True, drop=True)
    
    user_feat_cols, mx_feat_cols, tx_feat_cols = process_gnn_split(train_val_data, output_dir)
    
    # ---- Process test data ----
    test_data = data_all[test_idx].copy()
    test_out_dir = os.path.join(output_dir, "test_gnn")
    process_gnn_split(test_data, test_out_dir, is_test=True)
    
    # ---- Create and save feature masks for test data ----
    user_mask_map, user_mask = create_feature_mask(user_feat_cols, 0)
    mx_mask_map, mx_mask = create_feature_mask(mx_feat_cols, np.max(user_mask) + 1)
    tx_mask_map, tx_mask = create_feature_mask(tx_feat_cols, np.max(mx_mask) + 1)
    
    np.savetxt(
        os.path.join(test_out_dir, "nodes/user_feature_mask.csv"),
        user_mask, delimiter=",", fmt="%d"
    )
    np.savetxt(
        os.path.join(test_out_dir, "nodes/merchant_feature_mask.csv"),
        mx_mask, delimiter=",", fmt="%d"
    )
    np.savetxt(
        os.path.join(test_out_dir, "edges/user_to_merchant_feature_mask.csv"),
        tx_mask, delimiter=",", fmt="%d"
    )
    
    return user_mask_map, mx_mask_map, tx_mask_map 