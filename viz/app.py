"""
Fraud Detection GNN — Live Transaction Monitor
=============================================
A Dash app that visualises how the GNN processes incoming transactions:

  LEFT   — scrollable transaction stream (click to select)
  CENTRE — 2-hop subgraph extraction for the selected transaction
  RIGHT  — fraud score, outcome badge, and Shapley explanation

Run:
    cd <project-root>
    uv run python viz/app.py

    Then open http://localhost:8050
"""

import json
import os
import time
from collections import defaultdict

import networkx as nx

import boto3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, dcc, html, ctx
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(PROJECT_DIR, "data", "TabFormer", "gnn")
TEST_DIR      = os.path.join(DATA_DIR, "test_gnn")

AWS_PROFILE   = "Admin-Account-Access-541765610078"
AWS_REGION    = "us-east-1"
ENDPOINT_NAME = "fraud-detection-endpoint-v2"

N_FRAUD       = 15
N_LEGIT       = 15
MAX_1HOP      = 10
MAX_2HOP      = 5
THRESHOLD     = 0.5
RANDOM_SEED   = 42

# RobustScaler params fit on training Amount (year < 2018); used to decode scaled amounts
AMOUNT_MEDIAN = 30.32
AMOUNT_IQR    = 56.33

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
print("Loading data…")
train_edges      = pd.read_csv(os.path.join(DATA_DIR, "edges", "user_to_merchant.csv"))
user_feats       = pd.read_csv(os.path.join(DATA_DIR, "nodes", "user.csv"))
merchant_feats   = pd.read_csv(os.path.join(DATA_DIR, "nodes", "merchant.csv"))

test_edges       = pd.read_csv(os.path.join(DATA_DIR, "edges", "user_to_merchant.csv"))
test_edge_attrs  = pd.read_csv(os.path.join(DATA_DIR, "edges", "user_to_merchant_attr.csv"))
test_labels      = pd.read_csv(os.path.join(DATA_DIR, "edges", "user_to_merchant_label.csv"))

user_mask     = pd.read_csv(os.path.join(TEST_DIR, "nodes", "user_feature_mask.csv"),
                             header=None).values.ravel().astype(np.int32)
merchant_mask = pd.read_csv(os.path.join(TEST_DIR, "nodes", "merchant_feature_mask.csv"),
                             header=None).values.ravel().astype(np.int32)
edge_mask     = pd.read_csv(os.path.join(TEST_DIR, "edges", "user_to_merchant_feature_mask.csv"),
                             header=None).values.ravel().astype(np.int32)

# ---------------------------------------------------------------------------
# Build neighbor index
# ---------------------------------------------------------------------------
print("Building neighbor index…")
neighbors_user     = defaultdict(set)
neighbors_merchant = defaultdict(set)
for u, grp in train_edges.groupby("src")["dst"]:
    neighbors_user[u] = set(grp.values)
for m, grp in train_edges.groupby("dst")["src"]:
    neighbors_merchant[m] = set(grp.values)

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def numpy_to_triton(data):
    dtype_map = {np.float32: "FP32", np.int32: "INT32", np.int64: "INT64", np.bool_: "BOOL"}
    return [
        {"name": k, "shape": list(v.shape),
         "datatype": dtype_map.get(v.dtype.type, "FP32"),
         "data": v.flatten().tolist()}
        for k, v in data.items()
    ]

def build_payload(edge_df, attr_df, compute_shap=False):
    srcs, dsts = edge_df["src"].values, edge_df["dst"].values
    u_ids = sorted(set(srcs));  m_ids = sorted(set(dsts))
    u_map = {u: i for i, u in enumerate(u_ids)}
    m_map = {m: i for i, m in enumerate(m_ids)}
    return {
        "x_user":                             user_feats.iloc[u_ids].values.astype(np.float32),
        "x_merchant":                         merchant_feats.iloc[m_ids].values.astype(np.float32),
        "edge_index_user_to_merchant":        np.vstack([[u_map[u] for u in srcs],
                                                         [m_map[m] for m in dsts]]).astype(np.int64),
        "edge_attr_user_to_merchant":         attr_df.values.astype(np.float32),
        "COMPUTE_SHAP":                       np.array([compute_shap], dtype=np.bool_),
        "feature_mask_user":                  user_mask,
        "feature_mask_merchant":              merchant_mask,
        "edge_feature_mask_user_to_merchant": edge_mask,
    }

# ---------------------------------------------------------------------------
# Batch inference on a large context window (N_INFERENCE transactions)
# then select N_FRAUD + N_LEGIT for display
# Results are cached in viz/inference_cache.npz to avoid re-running on restart
# ---------------------------------------------------------------------------
N_INFERENCE  = 50000
CACHE_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_cache.npz")
SHAP_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shap_cache.json")

rng = np.random.default_rng(RANDOM_SEED)
fraud_idx = test_labels[test_labels["Fraud"] == 1].index.tolist()
legit_idx  = test_labels[test_labels["Fraud"] == 0].index.tolist()
n_fraud_batch = min(len(fraud_idx), N_INFERENCE // 2)
n_legit_batch  = min(len(legit_idx), N_INFERENCE - n_fraud_batch)
batch_idx = np.sort(np.concatenate([
    rng.choice(fraud_idx, n_fraud_batch, replace=False),
    rng.choice(legit_idx, n_legit_batch, replace=False),
]))
batch_edges  = test_edges.iloc[batch_idx].reset_index(drop=True)
batch_attrs  = test_edge_attrs.iloc[batch_idx].reset_index(drop=True)
batch_labels = test_labels.iloc[batch_idx].reset_index(drop=True)

session   = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
runtime   = session.client("sagemaker-runtime")
sm_client = session.client("sagemaker")

if os.path.exists(CACHE_PATH):
    cache = np.load(CACHE_PATH)
    # Validate cache was built from the same batch
    if np.array_equal(cache["batch_idx"], batch_idx):
        batch_predictions = cache["batch_predictions"]
        print(f"Loaded predictions from cache  "
              f"({(batch_predictions > THRESHOLD).sum()}/{len(batch_idx)} flagged as fraud)")
    else:
        print("Cache batch mismatch — re-running inference…")
        os.remove(CACHE_PATH)
        batch_predictions = None
else:
    batch_predictions = None

if batch_predictions is None:
    print(f"Fetching batch predictions from endpoint ({len(batch_idx)} transactions)…")
    t0 = time.time()
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": numpy_to_triton(build_payload(batch_edges, batch_attrs)),
                         "outputs": [{"name": "PREDICTION"}]}),
    )
    batch_predictions = np.array(json.loads(resp["Body"].read())["outputs"][0]["data"]).flatten()
    np.savez(CACHE_PATH, batch_idx=batch_idx, batch_predictions=batch_predictions)
    print(f"Predictions ready in {time.time()-t0:.1f}s — cached to {CACHE_PATH}  "
          f"({(batch_predictions > THRESHOLD).sum()}/{len(batch_idx)} flagged as fraud)")

# Random balanced sample: N_FRAUD actual fraud + N_LEGIT actual legit
batch_is_fraud = batch_labels["Fraud"].values == 1
fraud_pos = np.where(batch_is_fraud)[0]
legit_pos  = np.where(~batch_is_fraud)[0]
rng_display = np.random.default_rng(RANDOM_SEED)
sel_fraud = rng_display.choice(fraud_pos, min(N_FRAUD, len(fraud_pos)), replace=False)
sel_legit = rng_display.choice(legit_pos, N_LEGIT, replace=False)
chosen = np.sort(np.concatenate([sel_fraud, sel_legit]))

sample_edges  = batch_edges.iloc[chosen].reset_index(drop=True)
sample_attrs  = batch_attrs.iloc[chosen].reset_index(drop=True)
sample_labels = batch_labels.iloc[chosen].reset_index(drop=True)
predictions   = batch_predictions[chosen]
N_TX          = len(chosen)

# ---------------------------------------------------------------------------
# SHAP cache — keyed by batch hash so it auto-invalidates when batch changes
# ---------------------------------------------------------------------------
def _batch_hash():
    return str(hash(batch_idx.tobytes()))

def _load_shap_cache():
    if not os.path.exists(SHAP_CACHE_PATH):
        return {}
    with open(SHAP_CACHE_PATH) as f:
        data = json.load(f)
    if data.get("batch_hash") != _batch_hash():
        return {}
    return data.get("entries", {})

def _save_shap_cache(cache):
    with open(SHAP_CACHE_PATH, "w") as f:
        json.dump({"batch_hash": _batch_hash(), "entries": cache}, f)

shap_cache = _load_shap_cache()

# ---------------------------------------------------------------------------
# Extract 2-hop subgraphs
# ---------------------------------------------------------------------------
print("Extracting subgraphs…")

def extract_subgraph(anchor_user, anchor_merchant):
    visited       = sorted(neighbors_user.get(anchor_user, set()))
    hop1_merchants = [m for m in visited if m != anchor_merchant][:MAX_1HOP]
    all_merchants  = sorted(set(hop1_merchants) | {anchor_merchant})

    hop2_users = set()
    for m in hop1_merchants:
        peers = sorted(neighbors_merchant.get(m, set()) - {anchor_user})
        hop2_users.update(peers[:MAX_2HOP])
    hop2_users = sorted(hop2_users)
    all_users  = sorted({anchor_user} | set(hop2_users))

    user_set, merchant_set = set(all_users), set(all_merchants)
    mask = train_edges["src"].isin(user_set) & train_edges["dst"].isin(merchant_set)

    return dict(
        anchor_user=anchor_user,
        anchor_merchant=anchor_merchant,
        all_users=all_users,
        all_merchants=all_merchants,
        hop1_merchants=hop1_merchants,
        hop2_users=hop2_users,
        context_edges=train_edges[mask].copy(),
    )

subgraphs = [
    extract_subgraph(int(row["src"]), int(row["dst"]))
    for _, row in sample_edges.iterrows()
]

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
COLORS = dict(
    anchor_user="#1D4ED8", hop2_user="#93C5FD",
    anchor_merchant="#15803D", hop1_merchant="#86EFAC",
    context_edge="#E5E7EB", fraud_edge="#EF4444", legit_edge="#22C55E",
    bg="#F9FAFB",
)

STEP_LABELS = [
    "Step 1 / 3 — New transaction arrives",
    "Step 2 / 3 — Expanding: user's transaction history (1-hop)",
    "Step 3 / 3 — Full context: shared users revealed (2-hop)",
]


def make_subgraph_figure(idx, step=2):
    """
    Build the bipartite subgraph figure.

    step=0: anchor user + anchor merchant + new transaction edge only
    step=1: add 1-hop merchants and anchor-user context edges
    step=2: add 2-hop users and all context edges (full view)
    """
    sg   = subgraphs[idx]
    pred = float(predictions[idx])
    gt   = int(sample_labels.iloc[idx]["Fraud"])

    all_u = sg["all_users"];    all_m = sg["all_merchants"]
    hop1  = set(sg["hop1_merchants"]);  hop2 = set(sg["hop2_users"])
    ctx   = sg["context_edges"]
    au, am = sg["anchor_user"], sg["anchor_merchant"]

    # Fraudulent transactions from the full test set between 2-hop users and
    # subgraph merchants — reveals the fraud ring connecting the neighbourhood
    all_m_set = set(all_m)
    ring_mask = (
        test_labels["Fraud"].values == 1
    ) & (
        test_edges["src"].isin(hop2).values
    ) & (
        test_edges["dst"].isin(all_m_set).values
    )
    ring_rows = test_edges[ring_mask].head(30)  # cap to avoid overdrawing

    # Build full graph for layout — always use complete subgraph so positions
    # are stable across animation steps
    G_layout = nx.Graph()
    for u in all_u:
        G_layout.add_node(f"u{u}")
    for m in all_m:
        G_layout.add_node(f"m{m}")
    for _, row in ctx.iterrows():
        u, m = int(row["src"]), int(row["dst"])
        if u in set(all_u) and m in set(all_m):
            G_layout.add_edge(f"u{u}", f"m{m}")
    G_layout.add_edge(f"u{au}", f"m{am}")

    raw_pos = nx.spring_layout(G_layout, seed=idx, k=1.5)
    u_pos = {u: raw_pos[f"u{u}"] for u in all_u if f"u{u}" in raw_pos}
    m_pos = {m: raw_pos[f"m{m}"] for m in all_m if f"m{m}" in raw_pos}

    fig = go.Figure()

    # ── Context edges ────────────────────────────────────────────────────
    if step >= 2:
        # All context edges
        cx, cy = [], []
        for _, row in ctx.iterrows():
            u, m = int(row["src"]), int(row["dst"])
            if u in u_pos and m in m_pos:
                cx += [u_pos[u][0], m_pos[m][0], None]
                cy += [u_pos[u][1], m_pos[m][1], None]
        if cx:
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                                      line=dict(color=COLORS["context_edge"], width=1),
                                      hoverinfo="none", showlegend=False))
    elif step == 1:
        # Only edges from the anchor user to 1-hop merchants
        cx, cy = [], []
        for _, row in ctx[ctx["src"] == au].iterrows():
            m = int(row["dst"])
            if m in m_pos:
                cx += [u_pos[au][0], m_pos[m][0], None]
                cy += [u_pos[au][1], m_pos[m][1], None]
        if cx:
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                                      line=dict(color=COLORS["context_edge"], width=1),
                                      hoverinfo="none", showlegend=False))

    # ── Fraud ring edges (step 2 only) ───────────────────────────────────
    # Known-fraudulent transactions from the full test set between 2-hop users
    # and subgraph merchants
    if step >= 2:
        for _, ring_row in ring_rows.iterrows():
            pu = int(ring_row["src"])
            pm = int(ring_row["dst"])
            if pu in u_pos and pm in m_pos:
                fig.add_trace(go.Scatter(
                    x=[u_pos[pu][0], m_pos[pm][0]],
                    y=[u_pos[pu][1], m_pos[pm][1]],
                    mode="lines",
                    line=dict(color=COLORS["fraud_edge"], width=2, dash="dot"),
                    hovertext=f"Fraud ring: U{pu}→M{pm}",
                    hoverinfo="text", showlegend=False,
                ))

    # ── New transaction edge (always shown) ──────────────────────────────
    edge_color = COLORS["fraud_edge"] if pred > THRESHOLD else COLORS["legit_edge"]
    ux, uy = u_pos[au];  mx, my = m_pos[am]
    fig.add_trace(go.Scatter(x=[ux, mx], y=[uy, my], mode="lines",
                              line=dict(color=edge_color, width=4),
                              hoverinfo="none", showlegend=False))

    # ── 2-hop users (step 2 only) ────────────────────────────────────────
    if step >= 2:
        h2 = [u for u in hop2 if u in u_pos]
        if h2:
            fig.add_trace(go.Scatter(
                x=[u_pos[u][0] for u in h2], y=[u_pos[u][1] for u in h2],
                mode="markers",
                marker=dict(size=10, color=COLORS["hop2_user"], symbol="circle",
                            line=dict(color=COLORS["anchor_user"], width=1)),
                hovertext=[f"User {u}  (2-hop)" for u in h2], hoverinfo="text",
                showlegend=False,
            ))

    # ── Anchor user (always shown) ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[u_pos[au][0]], y=[u_pos[au][1]], mode="markers+text",
        marker=dict(size=22, color=COLORS["anchor_user"], symbol="circle",
                    line=dict(color="white", width=2)),
        text=[f"U{au}"], textposition="middle left",
        hovertext=[f"User {au}  (anchor)<br>History: {len(neighbors_user.get(au, set()))} merchants"],
        hoverinfo="text", showlegend=False,
    ))

    # ── 1-hop merchants (step 1+) ────────────────────────────────────────
    if step >= 1:
        h1 = [m for m in hop1 if m in m_pos]
        if h1:
            fig.add_trace(go.Scatter(
                x=[m_pos[m][0] for m in h1], y=[m_pos[m][1] for m in h1],
                mode="markers",
                marker=dict(size=10, color=COLORS["hop1_merchant"], symbol="square",
                            line=dict(color=COLORS["anchor_merchant"], width=1)),
                hovertext=[f"Merchant {m}  (1-hop)" for m in h1], hoverinfo="text",
                showlegend=False,
            ))

    # ── Anchor merchant (always shown) ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[m_pos[am][0]], y=[m_pos[am][1]], mode="markers+text",
        marker=dict(size=22, color=COLORS["anchor_merchant"], symbol="square",
                    line=dict(color="white", width=2)),
        text=[f"M{am}"], textposition="middle right",
        hovertext=[f"Merchant {am}  (anchor)<br>Total customers: {len(neighbors_merchant.get(am, set()))}"],
        hoverinfo="text", showlegend=False,
    ))

    predicted = pred > THRESHOLD
    outcome   = ("TP" if predicted and gt else "TN" if not predicted and not gt
                 else "FP" if predicted and not gt else "FN")

    fig.update_layout(
        title=dict(
            text=(f"{STEP_LABELS[step]}  ·  "
                  f"Tx {idx+1}: U{au}→M{am}  ·  Score: {pred:.3f}  ·  {outcome}"),
            font=dict(size=12), x=0.02,
        ),
        # Fixed axes so nodes don't jump between animation steps
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        plot_bgcolor=COLORS["bg"],
        margin=dict(l=10, r=10, t=50, b=40),
        height=520,
        transition=dict(duration=400, easing="cubic-in-out"),
    )
    fig.add_annotation(
        text=("🔵 Anchor user   🟦 2-hop users   "
              "🟩 Anchor merchant   🟢 1-hop merchants   — History"),
        xref="paper", yref="paper", x=0.5, y=-0.06,
        showarrow=False, font=dict(size=10, color="#6B7280"),
    )
    return fig


def make_shap_figure(idx):
    """Fetch Shapley values for transaction idx and return a bar chart."""
    cache_key = str(idx)
    shap_keys = ["shap_values_user", "shap_values_merchant", "shap_values_user_to_merchant"]

    if cache_key in shap_cache:
        outputs = {k: np.array(v) for k, v in shap_cache[cache_key].items()}
        print(f"SHAP cache hit for tx {idx}")
    else:
        single_edge = sample_edges.iloc[[idx]]
        single_attr = sample_attrs.iloc[[idx]]
        payload = build_payload(single_edge, single_attr, compute_shap=True)
        body    = json.dumps({
            "inputs":  numpy_to_triton(payload),
            "outputs": [{"name": "PREDICTION"}] + [{"name": k} for k in shap_keys],
        })
        resp2   = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                          ContentType="application/json", Body=body)
        outputs = {o["name"]: np.array(o["data"]).reshape(o["shape"])
                   for o in json.loads(resp2["Body"].read())["outputs"]}
        shap_cache[cache_key] = {k: outputs[k].tolist() for k in shap_keys if k in outputs}
        _save_shap_cache(shap_cache)

    # Each SHAP output contains one value per mask group, in group-index order
    # Groups are defined by the feature masks set at preprocessing time:
    #   user  group 0 → Card profile
    #   merch group 1 → Merchant profile   group 2 → Merchant category (MCC)
    #   edge  group 3 → City   4 → ZIP   5 → Errors   6 → Tx type   7 → Amount
    group_labels = {
        "shap_values_user":              ["Card profile"],
        "shap_values_merchant":          ["Merchant profile", "Merchant category (MCC)"],
        "shap_values_user_to_merchant":  ["City", "ZIP code", "Errors",
                                          "Transaction type", "Amount"],
    }

    labels, values = [], []
    for key, names in group_labels.items():
        if key not in outputs:
            continue
        vals = outputs[key].flatten()
        for name, val in zip(names, vals):
            labels.append(name)
            values.append(float(val))

    order  = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = ["#EF4444" if v > 0 else "#3B82F6" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#9CA3AF")
    fig.update_layout(
        title=dict(text="Shapley  (red → fraud  ·  blue → legit)", font=dict(size=12), x=0.02),
        xaxis_title="Contribution",
        plot_bgcolor="white",
        margin=dict(l=10, r=30, t=45, b=30),
        height=max(220, len(values) * 30 + 60),
    )
    return fig


def make_results_figure():
    fraud_scores = predictions[sample_labels["Fraud"].values == 1]
    legit_scores = predictions[sample_labels["Fraud"].values == 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=fraud_scores, name="Fraud (GT)", marker_color="#EF4444",
                                opacity=0.7, nbinsx=12))
    fig.add_trace(go.Histogram(x=legit_scores, name="Legit (GT)", marker_color="#22C55E",
                                opacity=0.7, nbinsx=12))
    fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="#6B7280",
                  annotation_text=f"threshold={THRESHOLD}", annotation_position="top right")
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Fraud Score Distribution", font=dict(size=13), x=0.5),
        xaxis_title="Fraud Score",
        yaxis_title="Count",
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="white",
        margin=dict(l=30, r=30, t=45, b=50),
        height=220,
    )
    return fig



# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
STREAM_STYLE_BASE = dict(
    padding="8px 10px", marginBottom="4px", width="100%",
    textAlign="left", border="1px solid #E5E7EB",
    borderRadius="6px", cursor="pointer", fontSize="12px",
    background="white",
)

def stream_button(i):
    pred  = float(predictions[i])
    gt    = int(sample_labels.iloc[i]["Fraud"])
    score = f"{pred:.2f}"
    icon  = "🚨" if pred > THRESHOLD else "✓ "
    u     = int(sample_edges.iloc[i]["src"])
    m     = int(sample_edges.iloc[i]["dst"])
    color = "#FEF2F2" if pred > THRESHOLD else "#F0FDF4"
    border = "#FECACA" if pred > THRESHOLD else "#BBF7D0"
    return html.Button(
        [
            html.Span(f"Tx {i+1}", style={"fontWeight": "600", "marginRight": "6px"}),
            html.Span(f"U{u}→M{m}", style={"color": "#6B7280", "fontSize": "11px"}),
            html.Span(f"{icon} {score}", style={"float": "right", "fontWeight": "600"}),
        ],
        id={"type": "tx-btn", "index": i},
        n_clicks=0,
        style={**STREAM_STYLE_BASE, "background": color, "borderColor": border},
    )


def stat_box(value, label, color="#111827"):
    return html.Div([
        html.Div(value, style={"fontSize": "24px", "fontWeight": "700", "color": color}),
        html.Div(label, style={"fontSize": "11px", "color": "#6B7280"}),
    ], style={"textAlign": "center", "padding": "10px 20px",
              "background": "white", "borderRadius": "8px",
              "border": "1px solid #E5E7EB"})


app = Dash(__name__, title="GNN Fraud Monitor")

app.layout = html.Div([
    # ── Stores & interval ──────────────────────────────────────────────────
    dcc.Store(id="selected-tx", data=0),
    dcc.Store(id="anim-step", data=2),   # current animation step (0,1,2)
    dcc.Interval(id="anim-interval", interval=900, disabled=True, n_intervals=0),

    # ── Header ─────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H2("GNN Fraud Detection — Live Transaction Monitor",
                    style={"margin": "0", "fontSize": "18px", "fontWeight": "700"}),
            html.P("Architecture demo · 2-hop subgraph extraction · SageMaker + Triton · "
                   "TabFormer synthetic dataset · Threshold 0.05 · Training set",
                   style={"margin": "2px 0 0", "fontSize": "12px", "color": "#6B7280"}),
        ], style={"flex": "1"}),
        html.Div([
            stat_box(f"{N_TX}", "Transactions"),
            stat_box(f"{N_FRAUD}", "Fraud samples", "#EF4444"),
            stat_box(f"{N_LEGIT}", "Legit samples", "#22C55E"),
        ], style={"display": "flex", "gap": "8px"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "20px",
              "padding": "14px 20px", "background": "#1E3A5F",
              "color": "white", "borderBottom": "3px solid #2563EB"}),

    # ── Main row ───────────────────────────────────────────────────────────
    html.Div([

        # LEFT: transaction stream
        html.Div([
            html.Div("Transaction Stream",
                     style={"fontWeight": "600", "fontSize": "13px",
                            "marginBottom": "8px", "color": "#374151"}),
            html.Div(
                [stream_button(i) for i in range(N_TX)],
                style={"overflowY": "auto", "height": "500px"},
            ),
        ], style={"width": "220px", "flexShrink": "0", "padding": "12px",
                  "borderRight": "1px solid #E5E7EB", "background": "#FAFAFA"}),

        # CENTRE: subgraph
        html.Div([
            html.Div(id="step-indicator", style={"padding": "6px 10px 2px"}),
            dcc.Graph(id="subgraph-fig",
                      figure=make_subgraph_figure(0, step=2),
                      config={"displayModeBar": False}),
        ], style={"flex": "1", "padding": "0"}),

        # RIGHT: prediction panel + shapley
        html.Div([
            html.Div(id="prediction-panel"),
            html.Div([
                html.Button("Explain (SHAP)",
                            id="shap-btn", n_clicks=0,
                            style={"width": "100%", "padding": "8px",
                                   "background": "#2563EB", "color": "white",
                                   "border": "none", "borderRadius": "6px",
                                   "cursor": "pointer", "fontSize": "13px",
                                   "marginBottom": "8px"}),
                html.Div(id="shap-status",
                         style={"fontSize": "11px", "color": "#6B7280",
                                "marginBottom": "4px", "textAlign": "center"}),
                dcc.Graph(id="shap-fig",
                          config={"displayModeBar": False},
                          style={"display": "none"}),
            ], style={"padding": "0 8px"}),
        ], style={"width": "270px", "flexShrink": "0",
                  "borderLeft": "1px solid #E5E7EB",
                  "padding": "12px 4px", "overflowY": "auto", "height": "545px"}),

    ], style={"display": "flex", "borderBottom": "1px solid #E5E7EB"}),

    # ── Bottom: score distribution ─────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(figure=make_results_figure(),
                      config={"displayModeBar": False}),
        ], style={"flex": "1"}),
        html.Div([
            html.Div("Legend", style={"fontWeight": "600", "fontSize": "12px",
                                       "marginBottom": "8px", "color": "#374151"}),
            *[html.Div([
                html.Span(sym, style={"marginRight": "6px", "fontSize": "14px"}),
                html.Span(label, style={"fontSize": "12px"}),
            ], style={"marginBottom": "6px"})
              for sym, label in [
                  ("🔵", "Anchor user"),
                  ("🟦", "2-hop users (ring members)"),
                  ("🟩", "Anchor merchant"),
                  ("🟢", "1-hop merchants (user history)"),
                  ("—", "Historical context edges"),
                  ("🔴", "Fraud prediction"),
                  ("🟢", "Legit prediction"),
              ]],
        ], style={"width": "220px", "padding": "16px 20px",
                  "borderLeft": "1px solid #E5E7EB"}),
    ], style={"display": "flex", "background": "#FAFAFA", "padding": "8px 0"}),

], style={"fontFamily": "system-ui, -apple-system, sans-serif",
          "background": "#F3F4F6", "minHeight": "100vh"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("selected-tx", "data"),
    Input({"type": "tx-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_transaction(_n_clicks):
    triggered = ctx.triggered_id
    return triggered["index"] if triggered else 0


@app.callback(
    Output("anim-step",     "data"),
    Output("anim-interval", "disabled"),
    Input("selected-tx",        "data"),
    Input("anim-interval",      "n_intervals"),
    State("anim-step",          "data"),
    prevent_initial_call=True,
)
def manage_animation(selected_tx, n_intervals, current_step):
    """
    Drive the 3-step subgraph expansion animation.

    - When a new transaction is selected: reset to step 0 and enable interval.
    - Each interval tick: advance step by 1; stop the interval at step 2.
    """
    if ctx.triggered_id == "selected-tx":
        return 0, False          # restart animation
    next_step = current_step + 1
    if next_step >= 2:
        return 2, True           # final frame reached – stop interval
    return next_step, False


@app.callback(
    Output("subgraph-fig",   "figure"),
    Output("step-indicator", "children"),
    Input("anim-step",  "data"),
    State("selected-tx", "data"),
)
def update_subgraph(step, idx):
    step_dots = [
        html.Span(
            "●" if i <= step else "○",
            style={"color": "#2563EB" if i <= step else "#D1D5DB",
                   "marginRight": "6px", "fontSize": "14px"},
        )
        for i in range(3)
    ]
    labels = ["New transaction", "1-hop merchants", "2-hop users"]
    indicator = html.Div(
        step_dots + [
            html.Span(labels[step],
                      style={"fontSize": "11px", "color": "#6B7280",
                             "fontWeight": "600"})
        ],
        style={"display": "flex", "alignItems": "center"},
    )
    return make_subgraph_figure(idx, step=step), indicator


@app.callback(
    Output("prediction-panel", "children"),
    Input("selected-tx", "data"),
)
def update_prediction_panel(idx):
    pred = float(predictions[idx])
    gt   = int(sample_labels.iloc[idx]["Fraud"])
    sg   = subgraphs[idx]

    is_fraud  = pred > THRESHOLD
    predicted = "FRAUD" if is_fraud else "LEGIT"
    outcome   = ("TP" if is_fraud and gt else
                 "TN" if not is_fraud and not gt else
                 "FP" if is_fraud and not gt else "FN")
    outcome_colors = {"TP": "#EF4444", "TN": "#22C55E", "FP": "#F59E0B", "FN": "#8B5CF6"}
    pred_color = "#EF4444" if is_fraud else "#22C55E"

    row = sample_attrs.iloc[idx]

    # Column order: City(0-13), Errors(14-18), Zip(19-33), Chip(34-36), Amount(37)

    # City (cols 0-13) — one-hot, show active index
    city_idx = int(row.iloc[0:14].values.argmax())

    # Errors (cols 14-18) — Errors_4 = no error, others = error present
    err_vals = row.iloc[14:19].values

    # ZIP (cols 19-33) — one-hot, show active index
    zip_idx = int(row.iloc[19:34].values.argmax())

    # Transaction type (cols 34-36)
    tx_type_cols = ["Chip", "Online", "Swipe"]
    tx_type_vals = row.iloc[34:37].values
    tx_type = tx_type_cols[int(tx_type_vals.argmax())] if tx_type_vals.max() > 0 else "Unknown"

    # Amount (col 37) — stored as RobustScaler normalised value; decode to dollars
    amount_dollars = float(row.iloc[37]) * AMOUNT_IQR + AMOUNT_MEDIAN
    err_active = int(err_vals.argmax())
    error_label = "No error" if err_active == 4 else f"Error (code {err_active})"

    return html.Div([
        html.Div(f"Transaction {idx + 1}",
                 style={"fontWeight": "700", "fontSize": "14px", "marginBottom": "10px"}),

        html.Div([
            html.Div(predicted,
                     style={"fontSize": "28px", "fontWeight": "800", "color": pred_color}),
            html.Div(f"Score: {pred:.4f}",
                     style={"fontSize": "13px", "color": "#6B7280"}),
        ], style={"marginBottom": "12px"}),

        html.Div("Transaction Details",
                 style={"fontWeight": "600", "fontSize": "12px",
                        "color": "#374151", "marginBottom": "6px"}),
        *[html.Div([
            html.Span(label, style={"color": "#6B7280", "fontSize": "11px",
                                    "width": "90px", "display": "inline-block"}),
            html.Span(str(value), style={"fontWeight": "600", "fontSize": "12px"}),
          ], style={"marginBottom": "3px"})
          for label, value in [
              ("Type",    tx_type),
              ("Amount",  f"${amount_dollars:.2f}"),
              ("City",    f"City {city_idx}"),
              ("ZIP",     f"ZIP {zip_idx}"),
              ("Errors",  error_label),
          ]],

        html.Hr(style={"borderColor": "#E5E7EB", "margin": "10px 0"}),

        html.Div([
            html.Span("Outcome: ", style={"fontSize": "12px", "color": "#6B7280"}),
            html.Span(outcome, style={"fontWeight": "700", "color": outcome_colors[outcome],
                                       "fontSize": "14px"}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.Span("Ground truth: ", style={"fontSize": "12px", "color": "#6B7280"}),
            html.Span("FRAUD" if gt else "LEGIT",
                      style={"fontSize": "12px",
                             "color": "#EF4444" if gt else "#22C55E",
                             "fontWeight": "600"}),
        ], style={"marginBottom": "14px"}),

        html.Hr(style={"borderColor": "#E5E7EB", "margin": "10px 0"}),

        html.Div("Subgraph", style={"fontWeight": "600", "fontSize": "12px",
                                     "color": "#374151", "marginBottom": "6px"}),
        *[html.Div([
            html.Span(label, style={"color": "#6B7280", "fontSize": "11px", "width": "110px",
                                     "display": "inline-block"}),
            html.Span(str(value), style={"fontWeight": "600", "fontSize": "12px"}),
          ], style={"marginBottom": "3px"})
          for label, value in [
              ("Users in subgraph", len(sg["all_users"])),
              ("Merchants", len(sg["all_merchants"])),
              ("Context edges", len(sg["context_edges"])),
              ("1-hop merchants", len(sg["hop1_merchants"])),
              ("2-hop users", len(sg["hop2_users"])),
          ]],
    ], style={"padding": "10px 12px"})


@app.callback(
    Output("shap-fig",    "figure"),
    Output("shap-fig",    "style"),
    Output("shap-status", "children"),
    Input("shap-btn",     "n_clicks"),
    State("selected-tx",  "data"),
    prevent_initial_call=True,
)
def fetch_shap(n_clicks, idx):
    status = "Computing Shapley values…"
    try:
        fig = make_shap_figure(idx)
        return fig, {"display": "block"}, f"Tx {idx+1} explained."
    except Exception as e:
        empty = go.Figure()
        empty.update_layout(height=220, plot_bgcolor="white")
        return empty, {"display": "none"}, f"Error: {e}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=8050)
