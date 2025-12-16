# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Prepare GNN graph data (edges, node features, masks)."""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "numpy==1.26.0",
        "pyarrow==15.0.0",
    ],
)
def prepare_gnn_datasets(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    test_data: Input[Dataset],
    feature_transformer_artifact: Input[Artifact],
    id_transformer_artifact: Input[Artifact],
    gnn_train_edges: Output[Dataset],
    gnn_train_user_features: Output[Dataset],
    gnn_train_merchant_features: Output[Dataset],
    gnn_train_edge_features: Output[Dataset],
    gnn_train_edge_labels: Output[Dataset],
    gnn_test_edges: Output[Dataset],
    gnn_test_user_features: Output[Dataset],
    gnn_test_merchant_features: Output[Dataset],
    gnn_test_edge_features: Output[Dataset],
    gnn_test_edge_labels: Output[Dataset],
    feature_masks_artifact: Output[Artifact],
    metrics: Output[Metrics],
) -> dict:
    """Prepare graph data for GNN training and testing.

    Creates a tripartite graph structure where:
    - User nodes (identified by Card)
    - Merchant nodes (identified by Merchant)
    - Edges represent transactions (User -> Merchant)

    Output artifacts:
    - Edge lists in COO format (src, dst)
    - Node features for users and merchants
    - Edge features (transaction attributes)
    - Edge labels (fraud indicator)
    - Feature masks for aggregation

    Args:
        train_data: Training + validation data for GNN training graph
        validation_data: Validation data (combined with train for graph)
        test_data: Test data for separate test graph
        feature_transformer_artifact: Fitted feature transformer
        id_transformer_artifact: Fitted ID transformer
        gnn_train_*: Output artifacts for training graph
        gnn_test_*: Output artifacts for test graph
        feature_masks_artifact: Output artifact with feature mask arrays
        metrics: Output metrics artifact

    Returns:
        Dict with graph statistics
    """
    import pickle

    import numpy as np
    import pandas as pd

    # Constants
    COL_MERCHANT = "Merchant"
    COL_CARD = "Card"
    COL_MCC = "MCC"
    COL_FRAUD = "Fraud"
    COL_MERCHANT_ID = "Merchant_ID"
    COL_USER_ID = "User_ID"
    COL_GRAPH_SRC = "src"
    COL_GRAPH_DST = "dst"
    MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_CARD, COL_MCC]

    def create_feature_mask(columns: list, start_id: int = 0) -> tuple:
        """Create feature mask mapping columns to group IDs."""
        mask_mapping = {}
        mask_values = []
        current_mask = start_id

        for col in columns:
            base = col.split("_")[0] if "_" in col else col
            if base not in mask_mapping:
                mask_mapping[base] = current_mask
                current_mask += 1
            mask_values.append(mask_mapping[base])

        return mask_mapping, np.array(mask_values)

    # Load transformers
    with open(feature_transformer_artifact.path, "rb") as f:
        feature_config = pickle.load(f)

    with open(id_transformer_artifact.path, "rb") as f:
        id_config = pickle.load(f)

    transformer = feature_config["transformer"]
    tx_feature_columns = feature_config["output_columns"]
    type_mapping = feature_config["type_mapping"]
    predictor_columns = feature_config["predictor_columns"]

    id_transformer = id_config["transformer"]
    id_columns = id_config["columns"]

    # Separate user vs merchant ID columns
    user_feature_columns = [c for c in id_columns if c.startswith("Card")]
    merchant_feature_columns = [c for c in id_columns if not c.startswith("Card")]

    def prepare_graph(data: pd.DataFrame, prefix: str) -> dict:
        """Prepare graph structures for a single dataset."""
        data = data.reset_index(drop=True)

        # Create consecutive IDs for merchants
        merchant_to_id = {m: i for i, m in enumerate(data[COL_MERCHANT].unique())}
        data[COL_MERCHANT_ID] = data[COL_MERCHANT].map(merchant_to_id)

        # Create consecutive IDs for users (cards)
        card_to_id = {c: i for i, c in enumerate(data[COL_CARD].unique())}
        data[COL_USER_ID] = data[COL_CARD].map(card_to_id)

        n_users = data[COL_USER_ID].max() + 1
        n_merchants = data[COL_MERCHANT_ID].max() + 1
        n_edges = len(data)

        print(
            f"{prefix}: {n_users:,} users, {n_merchants:,} merchants, {n_edges:,} edges"
        )

        # Create edge list (User -> Merchant)
        edges = pd.DataFrame(
            {
                COL_GRAPH_SRC: data[COL_USER_ID],
                COL_GRAPH_DST: data[COL_MERCHANT_ID],
            }
        )

        # Transform transaction features (edge attributes)
        tx_features = pd.DataFrame(
            transformer.transform(data[predictor_columns]),
            columns=tx_feature_columns,
        )
        for col, dtype in type_mapping.items():
            if col in tx_features.columns:
                tx_features[col] = tx_features[col].astype(dtype)

        # Edge labels
        edge_labels = data[[COL_FRAUD]].copy()

        # Get unique merchants sorted by ID
        merchant_data = (
            data[[COL_MERCHANT, COL_MCC, COL_CARD, COL_MERCHANT_ID]]
            .drop_duplicates(subset=[COL_MERCHANT])
            .sort_values(COL_MERCHANT_ID)
        )

        # Get unique users sorted by ID
        user_data = (
            data[[COL_MERCHANT, COL_MCC, COL_CARD, COL_USER_ID]]
            .drop_duplicates(subset=[COL_CARD])
            .sort_values(COL_USER_ID)
        )

        # Transform ID features
        merchant_id_features = pd.DataFrame(
            id_transformer.transform(
                merchant_data[MERCHANT_AND_USER_COLS].astype("category")
            ),
            columns=id_columns,
        )[merchant_feature_columns]

        user_id_features = pd.DataFrame(
            id_transformer.transform(
                user_data[MERCHANT_AND_USER_COLS].astype("category")
            ),
            columns=id_columns,
        )[user_feature_columns]

        return {
            "edges": edges,
            "user_features": user_id_features,
            "merchant_features": merchant_id_features,
            "edge_features": tx_features,
            "edge_labels": edge_labels,
            "stats": {
                "n_users": int(n_users),
                "n_merchants": int(n_merchants),
                "n_edges": int(n_edges),
                "n_fraud": int(edge_labels[COL_FRAUD].sum()),
            },
        }

    # Load and combine train + validation for GNN training graph
    train_df = pd.read_parquet(train_data.path)
    val_df = pd.read_parquet(validation_data.path)
    train_combined = pd.concat([train_df, val_df], ignore_index=True)

    # Prepare training graph
    train_graph = prepare_graph(train_combined, "Train")
    train_graph["edges"].to_csv(gnn_train_edges.path, index=False)
    train_graph["user_features"].to_csv(gnn_train_user_features.path, index=False)
    train_graph["merchant_features"].to_csv(
        gnn_train_merchant_features.path, index=False
    )
    train_graph["edge_features"].to_csv(gnn_train_edge_features.path, index=False)
    train_graph["edge_labels"].to_csv(gnn_train_edge_labels.path, index=False)

    # Load and prepare test graph
    test_df = pd.read_parquet(test_data.path)
    test_graph = prepare_graph(test_df, "Test")
    test_graph["edges"].to_csv(gnn_test_edges.path, index=False)
    test_graph["user_features"].to_csv(gnn_test_user_features.path, index=False)
    test_graph["merchant_features"].to_csv(gnn_test_merchant_features.path, index=False)
    test_graph["edge_features"].to_csv(gnn_test_edge_features.path, index=False)
    test_graph["edge_labels"].to_csv(gnn_test_edge_labels.path, index=False)

    # Create feature masks
    user_mask_map, user_mask = create_feature_mask(user_feature_columns, 0)
    merchant_mask_map, merchant_mask = create_feature_mask(
        merchant_feature_columns, np.max(user_mask) + 1
    )
    tx_mask_map, tx_mask = create_feature_mask(
        tx_feature_columns, np.max(merchant_mask) + 1
    )

    # Save feature masks
    with open(feature_masks_artifact.path, "wb") as f:
        pickle.dump(
            {
                "user_mask": user_mask,
                "merchant_mask": merchant_mask,
                "transaction_mask": tx_mask,
                "user_mask_map": user_mask_map,
                "merchant_mask_map": merchant_mask_map,
                "transaction_mask_map": tx_mask_map,
                "user_columns": user_feature_columns,
                "merchant_columns": merchant_feature_columns,
                "transaction_columns": tx_feature_columns,
            },
            f,
        )

    # Log metrics
    metrics.log_metric("train_users", train_graph["stats"]["n_users"])
    metrics.log_metric("train_merchants", train_graph["stats"]["n_merchants"])
    metrics.log_metric("train_edges", train_graph["stats"]["n_edges"])
    metrics.log_metric("test_users", test_graph["stats"]["n_users"])
    metrics.log_metric("test_merchants", test_graph["stats"]["n_merchants"])
    metrics.log_metric("test_edges", test_graph["stats"]["n_edges"])

    return {
        "train_graph": train_graph["stats"],
        "test_graph": test_graph["stats"],
        "feature_dimensions": {
            "user": len(user_feature_columns),
            "merchant": len(merchant_feature_columns),
            "transaction": len(tx_feature_columns),
        },
    }
