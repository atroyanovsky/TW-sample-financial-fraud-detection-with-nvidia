import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def print_tree(directory, prefix=""):
    """Recursively prints the directory tree starting at 'directory'."""
    # Retrieve a sorted list of entries in the directory
    entries = sorted(os.listdir(directory))
    entries_count = len(entries)
    
    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        # Determine the branch connector
        if index == entries_count - 1:
            connector = "└── "
            extension = "    "
        else:
            connector = "├── "
            extension = "│   "
        
        print(prefix + connector + entry)
        
        # If the entry is a directory, recursively print its contents
        if os.path.isdir(path):
            print_tree(path, prefix + extension)


def summarize_graph(gnn_dir):
    """Summarize the preprocessed GNN graph structure."""
    
    # Load training graph data
    edges_df = pd.read_csv(os.path.join(gnn_dir, "edges", "user_to_merchant.csv"))
    labels_df = pd.read_csv(os.path.join(gnn_dir, "edges", "user_to_merchant_label.csv"))
    users_df = pd.read_csv(os.path.join(gnn_dir, "nodes", "user.csv"))
    merchants_df = pd.read_csv(os.path.join(gnn_dir, "nodes", "merchant.csv"))
    
    # Load test graph data
    test_edges_df = pd.read_csv(os.path.join(gnn_dir, "test_gnn", "edges", "user_to_merchant.csv"))
    test_labels_df = pd.read_csv(os.path.join(gnn_dir, "test_gnn", "edges", "user_to_merchant_label.csv"))
    test_users_df = pd.read_csv(os.path.join(gnn_dir, "test_gnn", "nodes", "user.csv"))
    test_merchants_df = pd.read_csv(os.path.join(gnn_dir, "test_gnn", "nodes", "merchant.csv"))
    
    # Calculate statistics
    num_users = len(users_df)
    num_merchants = len(merchants_df)
    num_transactions = len(edges_df)
    num_fraud = labels_df['Fraud'].sum()
    fraud_rate = num_fraud / num_transactions * 100
    
    test_num_users = len(test_users_df)
    test_num_merchants = len(test_merchants_df)
    test_num_transactions = len(test_edges_df)
    test_num_fraud = test_labels_df['Fraud'].sum()
    test_fraud_rate = test_num_fraud / test_num_transactions * 100
    
    # User features and merchant features
    user_features = users_df.shape[1]
    merchant_features = merchants_df.shape[1]
    edge_attr_df = pd.read_csv(os.path.join(gnn_dir, "edges", "user_to_merchant_attr.csv"))
    edge_features = edge_attr_df.shape[1]
    
    # Print summary
    print("=" * 60)
    print("  GRAPH SUMMARY: User-Merchant Transaction Network")
    print("=" * 60)
    print()
    print("  Graph Type: Bipartite (Users ↔ Merchants)")
    print("  Edge Type:  Transactions (with fraud labels)")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                   TRAINING SET                      │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Users (nodes):        {num_users:>10,}                  │")
    print(f"  │  Merchants (nodes):    {num_merchants:>10,}                  │")
    print(f"  │  Transactions (edges): {num_transactions:>10,}                  │")
    print(f"  │  Fraudulent:           {num_fraud:>10,} ({fraud_rate:.2f}%)         │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                     TEST SET                        │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Users (nodes):        {test_num_users:>10,}                  │")
    print(f"  │  Merchants (nodes):    {test_num_merchants:>10,}                  │")
    print(f"  │  Transactions (edges): {test_num_transactions:>10,}                  │")
    print(f"  │  Fraudulent:           {test_num_fraud:>10,} ({test_fraud_rate:.2f}%)         │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │                 FEATURE DIMENSIONS                  │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  User features:        {user_features:>10}                  │")
    print(f"  │  Merchant features:    {merchant_features:>10}                  │")
    print(f"  │  Edge (tx) features:   {edge_features:>10}                  │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. Node distribution
    ax1 = axes[0]
    categories = ['Users', 'Merchants']
    train_counts = [num_users, num_merchants]
    test_counts = [test_num_users, test_num_merchants]
    x = range(len(categories))
    width = 0.35
    ax1.bar([i - width/2 for i in x], train_counts, width, label='Train', color='steelblue')
    ax1.bar([i + width/2 for i in x], test_counts, width, label='Test', color='coral')
    ax1.set_ylabel('Count')
    ax1.set_title('Node Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Fraud vs Normal transactions
    ax2 = axes[1]
    labels = ['Train', 'Test']
    fraud_counts = [num_fraud, test_num_fraud]
    normal_counts = [num_transactions - num_fraud, test_num_transactions - test_num_fraud]
    ax2.bar(labels, normal_counts, label='Normal', color='seagreen')
    ax2.bar(labels, fraud_counts, bottom=normal_counts, label='Fraud', color='crimson')
    ax2.set_ylabel('Transactions')
    ax2.set_title('Transaction Labels')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Fraud rate comparison
    ax3 = axes[2]
    ax3.bar(['Train', 'Test'], [fraud_rate, test_fraud_rate], color=['steelblue', 'coral'])
    ax3.set_ylabel('Fraud Rate (%)')
    ax3.set_title('Fraud Rate by Split')
    ax3.set_ylim(0, max(fraud_rate, test_fraud_rate) * 1.3)
    for i, v in enumerate([fraud_rate, test_fraud_rate]):
        ax3.text(i, v + 0.3, f'{v:.2f}%', ha='center', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'train': {'users': num_users, 'merchants': num_merchants, 'transactions': num_transactions, 'fraud': num_fraud},
        'test': {'users': test_num_users, 'merchants': test_num_merchants, 'transactions': test_num_transactions, 'fraud': test_num_fraud},
        'features': {'user': user_features, 'merchant': merchant_features, 'edge': edge_features}
    }


def extract_subgraph(full_data, edge_indices, gnn_data_dir):
    """
    Extract a subgraph containing only the specified edges and their connected nodes.
    
    Args:
        full_data: Full test graph data from load_hetero_graph()
        edge_indices: List of edge indices to include
        gnn_data_dir: Path to GNN data directory
    
    Returns:
        Subgraph data dict ready for inference
    """
    # Get edge data
    edge_index = full_data['edge_index_user_to_merchant']  # shape (2, num_edges)
    edge_attr = full_data['edge_attr_user_to_merchant']    # shape (num_edges, 38)
    
    # Extract selected edges
    selected_edge_index = edge_index[:, edge_indices]
    selected_edge_attr = edge_attr[edge_indices, :]
    
    # Find unique users and merchants in selected edges
    selected_users = np.unique(selected_edge_index[0, :])
    selected_merchants = np.unique(selected_edge_index[1, :])
    
    # Create mapping from old IDs to new contiguous IDs
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(selected_users)}
    merchant_id_map = {old_id: new_id for new_id, old_id in enumerate(selected_merchants)}
    
    # Remap edge indices to new contiguous IDs
    remapped_edge_index = np.zeros_like(selected_edge_index)
    for i in range(selected_edge_index.shape[1]):
        remapped_edge_index[0, i] = user_id_map[selected_edge_index[0, i]]
        remapped_edge_index[1, i] = merchant_id_map[selected_edge_index[1, i]]
    
    # Extract node features for selected nodes
    x_user = full_data['x_user'][selected_users, :]
    x_merchant = full_data['x_merchant'][selected_merchants, :]
    
    # Build subgraph data
    subgraph = {
        'x_user': x_user,
        'x_merchant': x_merchant,
        'feature_mask_user': full_data['feature_mask_user'],
        'feature_mask_merchant': full_data['feature_mask_merchant'],
        'edge_index_user_to_merchant': remapped_edge_index,
        'edge_attr_user_to_merchant': selected_edge_attr,
        'edge_feature_mask_user_to_merchant': full_data['edge_feature_mask_user_to_merchant'],
    }
    
    return subgraph

def create_batch_samples(full_data, batch_size, num_samples, gnn_data_dir):
    """Create multiple subgraph samples of a given batch size."""
    num_edges = full_data['edge_index_user_to_merchant'].shape[1]
    samples = []
    
    for _ in range(num_samples):
        # Randomly select batch_size edges
        edge_indices = np.random.choice(num_edges, size=min(batch_size, num_edges), replace=False)
        subgraph = extract_subgraph(full_data, edge_indices, gnn_data_dir)
        samples.append(subgraph)
    
    return samples

def print_subgraph_stats(subgraph, label=""):
    """Print statistics about a subgraph."""
    num_users = subgraph['x_user'].shape[0]
    num_merchants = subgraph['x_merchant'].shape[0]
    num_edges = subgraph['edge_index_user_to_merchant'].shape[1]
    print(f"{label}: {num_edges} transactions, {num_users} users, {num_merchants} merchants")