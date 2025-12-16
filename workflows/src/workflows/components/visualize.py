# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Components for generating visualizations in KFP UI."""

from kfp import dsl
from kfp.dsl import HTML, Artifact, Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "matplotlib==3.8.0",
        "seaborn==0.13.0",
        "pyarrow==15.0.0",
    ],
)
def visualize_data_stats(
    cleaned_data: Input[Dataset],
    stats_html: Output[HTML],
    metrics: Output[Metrics],
) -> dict:
    """Generate data statistics visualization for KFP UI.

    Creates an HTML report with:
    - Fraud rate distribution
    - Transaction amount distribution
    - Transactions by year
    - Fraud vs non-fraud comparison

    Args:
        cleaned_data: Cleaned dataset from preprocessing
        stats_html: Output HTML artifact for visualization
        metrics: Output metrics

    Returns:
        Dict with summary statistics
    """
    import base64
    import io

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # Load data
    df = pd.read_parquet(cleaned_data.path)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Fraud distribution
    ax1 = axes[0, 0]
    fraud_counts = df["Fraud"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax1.pie(
        fraud_counts.values,
        labels=["Non-Fraud", "Fraud"],
        autopct="%1.2f%%",
        colors=colors,
        explode=(0, 0.1),
    )
    ax1.set_title("Fraud Distribution", fontsize=14, fontweight="bold")

    # 2. Amount distribution (log scale)
    ax2 = axes[0, 1]
    df_sample = df.sample(min(50000, len(df)), random_state=42)
    sns.histplot(
        data=df_sample,
        x="Amount",
        hue="Fraud",
        bins=50,
        ax=ax2,
        palette=colors,
        alpha=0.7,
    )
    ax2.set_xscale("log")
    ax2.set_title("Transaction Amount Distribution", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Amount ($, log scale)")

    # 3. Transactions by year
    ax3 = axes[1, 0]
    year_counts = df.groupby(["Year", "Fraud"]).size().unstack(fill_value=0)
    year_counts.plot(kind="bar", ax=ax3, color=colors, alpha=0.8)
    ax3.set_title("Transactions by Year", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Count")
    ax3.legend(["Non-Fraud", "Fraud"])
    ax3.tick_params(axis="x", rotation=0)

    # 4. Fraud rate by year
    ax4 = axes[1, 1]
    fraud_rate_by_year = df.groupby("Year")["Fraud"].mean() * 100
    fraud_rate_by_year.plot(kind="line", ax=ax4, marker="o", color="#e74c3c", linewidth=2)
    ax4.set_title("Fraud Rate by Year", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Fraud Rate (%)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Calculate summary stats
    total_records = len(df)
    fraud_count = df["Fraud"].sum()
    fraud_rate = fraud_count / total_records
    avg_amount = df["Amount"].mean()
    median_amount = df["Amount"].median()

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Statistics Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-card.fraud {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
            .stat-value {{ font-size: 28px; font-weight: bold; }}
            .stat-label {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}
            .viz {{ text-align: center; margin: 30px 0; }}
            .viz img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TabFormer Data Statistics</h1>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{total_records:,}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card fraud">
                    <div class="stat-value">{fraud_count:,}</div>
                    <div class="stat-label">Fraud Cases</div>
                </div>
                <div class="stat-card fraud">
                    <div class="stat-value">{fraud_rate:.4%}</div>
                    <div class="stat-label">Fraud Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${avg_amount:,.2f}</div>
                    <div class="stat-label">Avg Transaction</div>
                </div>
            </div>
            <div class="viz">
                <img src="data:image/png;base64,{img_base64}" alt="Data Visualizations">
            </div>
        </div>
    </body>
    </html>
    """

    with open(stats_html.path, "w") as f:
        f.write(html_content)

    # Log metrics
    metrics.log_metric("total_records", total_records)
    metrics.log_metric("fraud_count", int(fraud_count))
    metrics.log_metric("fraud_rate", float(fraud_rate))
    metrics.log_metric("avg_amount", float(avg_amount))
    metrics.log_metric("median_amount", float(median_amount))

    return {
        "total_records": total_records,
        "fraud_count": int(fraud_count),
        "fraud_rate": float(fraud_rate),
    }


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "numpy==1.26.0",
        "matplotlib==3.8.0",
        "networkx==3.2.0",
        "pyarrow==15.0.0",
    ],
)
def visualize_graph_structure(
    gnn_train_data: Input[Artifact],
    graph_html: Output[HTML],
    metrics: Output[Metrics],
) -> dict:
    """Generate graph structure visualization for KFP UI.

    Creates a visualization of the bipartite user-merchant graph
    structure showing a sample subgraph.

    Args:
        gnn_train_data: GNN training data artifact
        graph_html: Output HTML artifact for visualization
        metrics: Output metrics

    Returns:
        Dict with graph statistics
    """
    import base64
    import io
    import os

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd

    # Load edge data
    edges_path = os.path.join(gnn_train_data.path, "edges", "user_to_merchant.csv")
    if os.path.exists(edges_path):
        edges_df = pd.read_csv(edges_path)
    else:
        # Try alternate structure
        edges_df = pd.read_parquet(gnn_train_data.path)

    # Get unique users and merchants
    users = edges_df["src"].unique() if "src" in edges_df.columns else []
    merchants = edges_df["dst"].unique() if "dst" in edges_df.columns else []

    num_users = len(users)
    num_merchants = len(merchants)
    num_edges = len(edges_df)

    # Create sample subgraph for visualization
    # Take a random user and their 2-hop neighborhood
    if len(users) > 0:
        sample_user = np.random.choice(users)

        # Get merchants connected to sample user
        user_merchants = edges_df[edges_df["src"] == sample_user]["dst"].unique()[:10]

        # Get other users connected to those merchants
        other_users = edges_df[edges_df["dst"].isin(user_merchants)]["src"].unique()[:10]

        # Build subgraph
        G = nx.Graph()

        # Add user nodes
        for u in [sample_user] + list(other_users):
            G.add_node(f"U{u}", bipartite=0, node_type="user")

        # Add merchant nodes
        for m in user_merchants:
            G.add_node(f"M{m}", bipartite=1, node_type="merchant")

        # Add edges
        sub_edges = edges_df[
            (edges_df["src"].isin([sample_user] + list(other_users)))
            & (edges_df["dst"].isin(user_merchants))
        ]
        for _, row in sub_edges.iterrows():
            G.add_edge(f"U{int(row['src'])}", f"M{int(row['dst'])}")

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        # Layout
        user_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "user"]
        merchant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "merchant"]

        pos = nx.bipartite_layout(G, user_nodes)

        # Draw
        nx.draw_networkx_nodes(
            G, pos, nodelist=user_nodes,
            node_color="#3498db", node_size=500,
            node_shape="o", alpha=0.9, ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos, nodelist=merchant_nodes,
            node_color="#e74c3c", node_size=400,
            node_shape="s", alpha=0.9, ax=ax
        )

        # Highlight sample user
        nx.draw_networkx_nodes(
            G, pos, nodelist=[f"U{sample_user}"],
            node_color="#f1c40f", node_size=700,
            node_shape="o", ax=ax
        )

        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        ax.set_title("Sample User-Merchant Bipartite Subgraph", fontsize=14, fontweight="bold")
        ax.legend(
            handles=[
                plt.scatter([], [], c="#3498db", s=100, marker="o", label="Users"),
                plt.scatter([], [], c="#e74c3c", s=100, marker="s", label="Merchants"),
                plt.scatter([], [], c="#f1c40f", s=100, marker="o", label="Sample User"),
            ],
            loc="upper right"
        )
        ax.axis("off")

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No graph data available", ha="center", va="center", fontsize=16)
        ax.axis("off")

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Graph Structure Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-card.users {{ background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }}
            .stat-card.merchants {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
            .stat-value {{ font-size: 28px; font-weight: bold; }}
            .stat-label {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}
            .viz {{ text-align: center; margin: 30px 0; }}
            .viz img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .info {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Graph Structure Analysis</h1>
            <div class="stats-grid">
                <div class="stat-card users">
                    <div class="stat-value">{num_users:,}</div>
                    <div class="stat-label">Unique Users</div>
                </div>
                <div class="stat-card merchants">
                    <div class="stat-value">{num_merchants:,}</div>
                    <div class="stat-label">Unique Merchants</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{num_edges:,}</div>
                    <div class="stat-label">Total Edges (Transactions)</div>
                </div>
            </div>
            <div class="viz">
                <h3>Sample Subgraph Visualization</h3>
                <img src="data:image/png;base64,{img_base64}" alt="Graph Visualization">
            </div>
            <div class="info">
                <strong>Graph Structure:</strong> Bipartite graph with Users (circles) connected to Merchants (squares) via transaction edges.
                The yellow node represents a randomly sampled user showing their local neighborhood.
            </div>
        </div>
    </body>
    </html>
    """

    with open(graph_html.path, "w") as f:
        f.write(html_content)

    # Log metrics
    metrics.log_metric("num_users", num_users)
    metrics.log_metric("num_merchants", num_merchants)
    metrics.log_metric("num_edges", num_edges)
    metrics.log_metric("avg_edges_per_user", num_edges / max(num_users, 1))

    return {
        "num_users": num_users,
        "num_merchants": num_merchants,
        "num_edges": num_edges,
    }
