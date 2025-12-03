# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

"""
Visualize the bipartite User-Merchant graph from preprocessed TabFormer data.

Usage:
    python visualize_graph.py <tabformer_base_path> [--sample N] [--output FILE]

Example:
    python visualize_graph.py /path/to/tabformer --sample 100 --output graph.html
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install: pip install networkx matplotlib")
    sys.exit(1)

# Optional: interactive visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_graph_data(tabformer_base_path, use_test=False):
    """Load edge and node data from preprocessed GNN files."""
    # Check both possible locations for GNN data
    gnn_path = os.path.join(tabformer_base_path, "gnn")
    if not os.path.exists(gnn_path):
        gnn_path = os.path.join(tabformer_base_path, "raw", "gnn")
    if use_test:
        gnn_path = os.path.join(gnn_path, "test_gnn")
    
    edges_path = os.path.join(gnn_path, "edges", "user_to_merchant.csv")
    labels_path = os.path.join(gnn_path, "edges", "user_to_merchant_label.csv")
    raw_data_path = os.path.join(tabformer_base_path, "raw", "card_transaction.v1.csv")
    users_path = os.path.join(gnn_path, "nodes", "user.csv")
    merchants_path = os.path.join(gnn_path, "nodes", "merchant.csv")
    
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Edge file not found: {edges_path}\nRun preprocess_TabFormer_lp.py first.")
    
    edges = pd.read_csv(edges_path)
    labels = pd.read_csv(labels_path) if os.path.exists(labels_path) else None
    raw_data = pd.read_csv(raw_data_path) if os.path.exists(raw_data_path) else None
    users = pd.read_csv(users_path) if os.path.exists(users_path) else None
    merchants = pd.read_csv(merchants_path) if os.path.exists(merchants_path) else None
    
    return edges, labels, raw_data, users, merchants


def build_networkx_graph(edges, labels=None, raw_data=None, sample_n=None):
    """Build a NetworkX graph from edge data."""
    if sample_n and len(edges) > sample_n:
        sampled_edges = edges.sample(n=sample_n, random_state=42)
        sampled_idx = sampled_edges.index
        edges = sampled_edges
        if labels is not None:
            labels = labels.loc[sampled_idx]
        if raw_data is not None:
            raw_data = raw_data.loc[sampled_idx]
    
    G = nx.Graph()
    
    # Add nodes with type attribute
    unique_users = edges['src'].unique()
    unique_merchants = edges['dst'].unique()
    
    for u in unique_users:
        G.add_node(f"U_{u}", node_type="user", node_id=u)
    for m in unique_merchants:
        G.add_node(f"M_{m}", node_type="merchant", node_id=m)
    
    # Add edges with fraud label and raw transaction attributes
    for idx, row in edges.iterrows():
        fraud = labels.loc[idx, 'Fraud'] if labels is not None else 0
        
        # Get transaction attributes from raw data
        edge_attrs = {'fraud': fraud}
        if raw_data is not None:
            raw_row = raw_data.loc[idx]
            edge_attrs['amount'] = raw_row.get('Amount', 'N/A')
            edge_attrs['tx_type'] = raw_row.get('Use Chip', 'Unknown')
            edge_attrs['merchant_name'] = raw_row.get('Merchant Name', 'Unknown')
            edge_attrs['city'] = raw_row.get('Merchant City', 'Unknown')
            edge_attrs['state'] = raw_row.get('Merchant State', 'Unknown')
            edge_attrs['mcc'] = raw_row.get('MCC', 'Unknown')
            edge_attrs['date'] = f"{raw_row.get('Month', '?')}/{raw_row.get('Day', '?')}/{raw_row.get('Year', '?')}"
            edge_attrs['time'] = raw_row.get('Time', 'Unknown')
            edge_attrs['errors'] = raw_row.get('Errors?', '')
        
        G.add_edge(f"U_{row['src']}", f"M_{row['dst']}", **edge_attrs)
    
    return G


def visualize_matplotlib(G, output_path=None, title="User-Merchant Transaction Graph"):
    """Create a static matplotlib visualization."""
    plt.figure(figsize=(14, 10))
    
    # Separate nodes by type
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'user']
    merchant_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'merchant']
    
    # Layout: bipartite
    pos = {}
    for i, node in enumerate(user_nodes):
        pos[node] = (0, i * 2)
    for i, node in enumerate(merchant_nodes):
        pos[node] = (3, i * 2)
    
    # If too many nodes, use spring layout instead
    if len(G.nodes()) > 50:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='#4CAF50', 
                           node_size=300, label='Users', alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=merchant_nodes, node_color='#2196F3', 
                           node_size=300, label='Merchants', alpha=0.8)
    
    # Draw edges - color by fraud
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 1]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#CCCCCC', 
                           alpha=0.5, width=1)
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='#F44336', 
                           alpha=0.8, width=2, label='Fraud')
    
    plt.legend(scatterpoints=1, loc='upper left')
    plt.title(f"{title}\n({len(user_nodes)} users, {len(merchant_nodes)} merchants, "
              f"{len(G.edges())} transactions, {len(fraud_edges)} fraudulent)")
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved static visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_plotly(G, output_path=None, title="User-Merchant Transaction Graph"):
    """Create an interactive Plotly visualization."""
    if not HAS_PLOTLY:
        print("Plotly not installed. Install with: pip install plotly")
        return
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Edge traces - we need separate traces for hover to work on edges
    # Create edge midpoints for hover markers
    normal_edge_x, normal_edge_y = [], []
    fraud_edge_x, fraud_edge_y = [], []
    normal_mid_x, normal_mid_y, normal_hover = [], [], []
    fraud_mid_x, fraud_mid_y, fraud_hover = [], [], []
    
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        
        # Build hover text with rich transaction details
        fraud = d.get('fraud', 0)
        amount = d.get('amount', 'N/A')
        tx_type = d.get('tx_type', 'Unknown')
        merchant_name = d.get('merchant_name', 'Unknown')
        city = d.get('city', '')
        state = d.get('state', '')
        mcc = d.get('mcc', '')
        date = d.get('date', '')
        time = d.get('time', '')
        errors = d.get('errors', '')
        user_id = G.nodes[u].get('node_id', u)
        merchant_id = G.nodes[v].get('node_id', v)
        
        location = f"{city}, {state}" if city and state else "Unknown"
        
        hover_text = (f"<b>{'🚨 FRAUD' if fraud else 'Transaction'}</b><br>"
                      f"<b>Amount:</b> {amount}<br>"
                      f"<b>Type:</b> {tx_type}<br>"
                      f"<b>Date:</b> {date} {time}<br>"
                      f"<b>User ID:</b> {user_id}<br>"
                      f"<b>Merchant:</b> {merchant_name}<br>"
                      f"<b>Location:</b> {location}<br>"
                      f"<b>MCC:</b> {mcc}")
        if errors and str(errors) != 'nan':
            hover_text += f"<br><b>Errors:</b> {errors}"
        
        if fraud == 1:
            fraud_edge_x.extend([x0, x1, None])
            fraud_edge_y.extend([y0, y1, None])
            fraud_mid_x.append(mid_x)
            fraud_mid_y.append(mid_y)
            fraud_hover.append(hover_text)
        else:
            normal_edge_x.extend([x0, x1, None])
            normal_edge_y.extend([y0, y1, None])
            normal_mid_x.append(mid_x)
            normal_mid_y.append(mid_y)
            normal_hover.append(hover_text)
    
    # Edge lines
    normal_edge_trace = go.Scatter(
        x=normal_edge_x, y=normal_edge_y, mode='lines',
        line=dict(width=0.5, color='#CCCCCC'),
        hoverinfo='none', name='Normal', showlegend=True
    )
    
    fraud_edge_trace = go.Scatter(
        x=fraud_edge_x, y=fraud_edge_y, mode='lines',
        line=dict(width=2, color='#F44336'),
        hoverinfo='none', name='Fraud', showlegend=True
    )
    
    # Edge midpoint markers for hover - visible small dots
    normal_hover_trace = go.Scatter(
        x=normal_mid_x, y=normal_mid_y, mode='markers',
        marker=dict(size=12, color='#999999', symbol='circle', 
                    line=dict(width=1, color='white')),
        text=normal_hover, hoverinfo='text', 
        name='Normal Tx (hover)', showlegend=True
    )
    
    fraud_hover_trace = go.Scatter(
        x=fraud_mid_x, y=fraud_mid_y, mode='markers',
        marker=dict(size=14, color='#F44336', symbol='diamond',
                    line=dict(width=2, color='white')),
        text=fraud_hover, hoverinfo='text',
        name='Fraud Tx (hover)', showlegend=True
    )
    
    # Node traces
    user_x, user_y, user_text = [], [], []
    merchant_x, merchant_y, merchant_text = [], [], []
    
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        if data.get('node_type') == 'user':
            user_x.append(x)
            user_y.append(y)
            user_text.append(f"User {data.get('node_id')}<br>Connections: {G.degree(node)}")
        else:
            merchant_x.append(x)
            merchant_y.append(y)
            merchant_text.append(f"Merchant {data.get('node_id')}<br>Connections: {G.degree(node)}")
    
    user_trace = go.Scatter(
        x=user_x, y=user_y, mode='markers',
        marker=dict(size=10, color='#4CAF50', line=dict(width=1, color='white')),
        text=user_text, hoverinfo='text', name='Users'
    )
    
    merchant_trace = go.Scatter(
        x=merchant_x, y=merchant_y, mode='markers',
        marker=dict(size=10, color='#2196F3', line=dict(width=1, color='white')),
        text=merchant_text, hoverinfo='text', name='Merchants'
    )
    
    # Count fraud edges
    fraud_count = sum(1 for _, _, d in G.edges(data=True) if d.get('fraud', 0) == 1)
    
    fig = go.Figure(
        data=[normal_edge_trace, fraud_edge_trace, normal_hover_trace, fraud_hover_trace, user_trace, merchant_trace],
        layout=go.Layout(
            title=f"{title}<br>({len(user_x)} users, {len(merchant_x)} merchants, "
                  f"{len(G.edges())} transactions, {fraud_count} fraudulent)",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Saved interactive visualization to: {output_path}")
    else:
        fig.show()


def print_graph_stats(G, edges, labels):
    """Print summary statistics about the graph."""
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'user']
    merchant_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'merchant']
    
    fraud_count = labels['Fraud'].sum() if labels is not None else 0
    
    print("\n" + "="*50)
    print("GRAPH STATISTICS")
    print("="*50)
    print(f"Total nodes:        {G.number_of_nodes()}")
    print(f"  - Users:          {len(user_nodes)}")
    print(f"  - Merchants:      {len(merchant_nodes)}")
    print(f"Total edges:        {G.number_of_edges()}")
    print(f"  - Fraudulent:     {fraud_count} ({100*fraud_count/len(edges):.2f}%)")
    print(f"  - Normal:         {len(edges) - fraud_count}")
    print(f"Graph density:      {nx.density(G):.6f}")
    
    degrees = [d for _, d in G.degree()]
    print(f"Avg degree:         {np.mean(degrees):.2f}")
    print(f"Max degree:         {max(degrees)}")
    print("="*50 + "\n")


def visualize_labeled(G, output_path=None, title="Labeled Transaction Graph"):
    """Create a small labeled visualization with node and edge labels."""
    plt.figure(figsize=(16, 12))
    
    # Separate nodes by type
    user_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'user']
    merchant_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'merchant']
    
    # Use spring layout for better spacing
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_color='#4CAF50', 
                           node_size=800, label='Users', alpha=0.9)
    nx.draw_networkx_nodes(G, pos, nodelist=merchant_nodes, node_color='#2196F3', 
                           node_size=800, label='Merchants', alpha=0.9)
    
    # Draw edges - color by fraud
    fraud_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 1]
    normal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('fraud', 0) == 0]
    
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='#CCCCCC', 
                           alpha=0.7, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='#F44336', 
                           alpha=0.9, width=3)
    
    # Node labels (shortened)
    node_labels = {}
    for n, d in G.nodes(data=True):
        node_id = d.get('node_id', n)
        if d.get('node_type') == 'user':
            node_labels[n] = f"U{node_id}"
        else:
            node_labels[n] = f"M{node_id}"
    
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
    
    # Edge labels
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        edge_labels[(u, v)] = "FRAUD" if d.get('fraud', 0) == 1 else ""
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color='#D32F2F')
    
    fraud_count = len(fraud_edges)
    plt.legend(scatterpoints=1, loc='upper left', fontsize=10)
    plt.title(f"{title}\n({len(user_nodes)} users, {len(merchant_nodes)} merchants, "
              f"{len(G.edges())} transactions, {fraud_count} fraudulent)", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved labeled visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize TabFormer transaction graph")
    parser.add_argument("tabformer_path", help="Base path to TabFormer data directory")
    parser.add_argument("--sample", "-n", type=int, default=500,
                        help="Number of edges to sample for visualization (default: 500)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (.png for static, .html for interactive)")
    parser.add_argument("--test", action="store_true",
                        help="Use test set instead of training data")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Create interactive Plotly visualization")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics, no visualization")
    parser.add_argument("--labeled", "-l", action="store_true",
                        help="Create small labeled visualization (best with --sample 20-50)")
    
    args = parser.parse_args()
    
    print(f"Loading graph data from: {args.tabformer_path}")
    edges, labels, attrs, users, merchants = load_graph_data(args.tabformer_path, use_test=args.test)
    
    print(f"Building graph (sampling {args.sample} edges)...")
    G = build_networkx_graph(edges, labels, attrs, sample_n=args.sample)
    
    print_graph_stats(G, edges, labels)
    
    if args.stats_only:
        return
    
    if args.labeled:
        output = args.output or "graph_labeled.png"
        visualize_labeled(G, output)
    elif args.interactive or (args.output and args.output.endswith('.html')):
        visualize_plotly(G, args.output)
    else:
        output = args.output or "graph_visualization.png"
        visualize_matplotlib(G, output)


if __name__ == "__main__":
    main()
