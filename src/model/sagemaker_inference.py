import os

import numpy as np
import pandas as pd
import tritonclient.http as httpclient
from tritonclient.http import InferInput, InferRequestedOutput


def make_example_request():
    # -- example sizes --
    num_merchants = 5
    num_users   = 7
    num_edges   = 3
    merchant_feature_dim = 24
    user_feature_dim = 13
    user_to_merchant_feature_dim = 38

    # -- 1) features --
    x_merchant = np.random.randn(num_merchants, merchant_feature_dim).astype(np.float32)
    x_user   = np.random.randn(num_users, user_feature_dim).astype(np.float32)

    # -- 2) shap flag and masks --
    compute_shap          = np.array([True], dtype=np.bool_)
    feature_mask_merchant   = np.random.randint(0,2, size=(merchant_feature_dim,), dtype=np.int32)
    feature_mask_user     = np.random.randint(0,2, size=(user_feature_dim,), dtype=np.int32)

    # -- 3) edges: index [2, num_edges] and attributes [num_edges,user_to_merchant_feature_dim] --
    edge_index_user_to_merchant = np.vstack([
        np.random.randint(0, num_users,   size=(num_edges,)),
        np.random.randint(0, num_merchants, size=(num_edges,))
    ]).astype(np.int64)
    
    edge_attr_user_to_merchant = np.random.randn(num_edges, user_to_merchant_feature_dim).astype(np.float32)

    feature_mask_user_to_merchant =  np.random.randint(0,2, size=(user_to_merchant_feature_dim,), dtype=np.int32)

    return {
        "x_merchant": x_merchant,
        "x_user": x_user,
        "COMPUTE_SHAP": compute_shap,
        "feature_mask_merchant": feature_mask_merchant,
        "feature_mask_user": feature_mask_user,
        "edge_index_user_to_merchant": edge_index_user_to_merchant,
        "edge_attr_user_to_merchant": edge_attr_user_to_merchant,
        "edge_feature_mask_user_to_merchant": feature_mask_user_to_merchant
    }

def prepare_and_send_inference_request(data, host="localhost", http_port=8000):

    # Connect to Triton
    client = httpclient.InferenceServerClient(url=f'{host}:{http_port}')

    # Prepare Inputs

    inputs = []
    def _add_input(name, arr, dtype):
        inp = InferInput(name, arr.shape, datatype=dtype)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    for key, value in data.items():
        if key.startswith("x_"):
            dtype = "FP32"
        elif key.startswith("feature_mask_"):
            dtype = "INT32"
        elif key.startswith("edge_feature_mask_"):
            dtype = "INT32"            
        elif key.startswith("edge_index_"):
            dtype = "INT64"
        elif key.startswith("edge_attr_"):
            dtype = "FP32"
        elif key == "COMPUTE_SHAP":
            dtype = "BOOL"
        else:
            continue  # skip things we don't care about

        _add_input(key, value, dtype)


    # Outputs

    outputs = [InferRequestedOutput("PREDICTION")]

    for key in data:
        if key.startswith("x_"):
            node = key[len("x_"):]  # extract node name
            outputs.append(InferRequestedOutput(f"shap_values_{node}"))
        elif key.startswith("edge_attr_"):
            edge_name = key[len("edge_attr_"):]  # extract edge name
            outputs.append(InferRequestedOutput(f"shap_values_{edge_name}"))
    
    # Send request

    model_name="prediction_and_shapley"
    response = client.infer(
        model_name,
        inputs=inputs,
        request_id=str(1),
        outputs=outputs,
        timeout= 3000
    )

    result = {}

    # always include prediction
    result["PREDICTION"] = response.as_numpy("PREDICTION")

    # add shap values
    for key in data:
        if key.startswith("x_"):
            node = key[len("x_"):]  # e.g. "merchant", "user"
            result[f"shap_values_{node}"] = response.as_numpy(f"shap_values_{node}")
        if key.startswith("edge_attr_"):
            edge_name = key[len("edge_attr_"):]  # e.g. ("user" "to"  "merchant")
            result[f"shap_values_{edge_name}"] = response.as_numpy(f"shap_values_{edge_name}")
    
    return result

def load_hetero_graph(gnn_data_dir, partition):
    """
    Load a heterogeneous graph from preprocessed CSV files.

    Args:
        gnn_data_dir: Base directory containing the GNN data.
        partition: Data partition to load (e.g., 'train', 'test').

    Returns:
        dict: A dictionary containing:
            - x_<node>: Node features as np.float32 arrays
            - feature_mask_<node>: Feature masks as np.int32 arrays
            - edge_index_<edge>: Edge indices as np.int64 arrays
            - edge_attr_<edge>: Edge attributes as np.float32 arrays
            - edge_feature_mask_<edge>: Edge feature masks as np.int32 arrays
            - edge_label_<edge>: Edge labels as DataFrames (if present)
    Reads:
      - All node CSVs from nodes/, plus their matching feature masks (<node>_feature_mask.csv)
        If missing, a mask of all ones is created (np.int32).
      - All edge CSVs from edges/:
          base        -> edge_index_<edge> (np.int64)
          *_attr.csv  -> edge_attr_<edge>  (np.float32)
          *_label.csv -> exactly one -> edge_label_<edge> (DataFrame)
    """
    base = os.path.join(gnn_data_dir, f"{partition}_gnn")
    # Fall back to gnn_data_dir root if partition subdirectory doesn't exist
    if not os.path.isdir(base):
        base = gnn_data_dir
    nodes_dir = os.path.join(base, "nodes")
    edges_dir = os.path.join(base, "edges")

    out = {}
    node_feature_mask = {}

    # --- Nodes: every CSV becomes x_<node>; also read/create feature_mask_<node> ---
    if os.path.isdir(nodes_dir):
        for fname in os.listdir(nodes_dir):
            if fname.lower().endswith(".csv") and not fname.lower().endswith("_feature_mask.csv"):
                node_name = fname[:-len(".csv")]
                node_path = os.path.join(nodes_dir, fname)
                node_df = pd.read_csv(node_path)
                out[f"x_{node_name}"] = node_df.to_numpy(dtype=np.float32)

                # feature mask file (optional)
                mask_fname = f"{node_name}_feature_mask.csv"
                mask_path = os.path.join(nodes_dir, mask_fname)
                if os.path.exists(mask_path):
                    mask_df = pd.read_csv(mask_path, header=None)
                    node_feature_mask[node_name] = mask_df
                    feature_mask = mask_df.to_numpy(dtype=np.int32).ravel()
                else:
                    # create a must with all zeros
                    feature_mask = np.zeros(node_df.shape[1], dtype=np.int32)
                out[f"feature_mask_{node_name}"] = feature_mask

    # --- Edges: group into base, attr, label by filename suffix ---
    base_edges = {}
    edge_attrs = {}
    edge_labels = {}
    edge_feature_mask = {}

    if os.path.isdir(edges_dir):
        for fname in os.listdir(edges_dir):
            if not fname.lower().endswith(".csv"):
                continue
            path = os.path.join(edges_dir, fname)
            lower = fname.lower()
            if lower.endswith("_attr.csv"):
                edge_name = fname[:-len("_attr.csv")]
                edge_attrs[edge_name] = pd.read_csv(path) #, header=None)
            elif lower.endswith("_label.csv"):
                edge_name = fname[:-len("_label.csv")]
                edge_labels[edge_name] = pd.read_csv(path)
            elif lower.endswith("_feature_mask.csv"):
                edge_name = fname[:-len("_feature_mask.csv")]
                edge_feature_mask[edge_name] = pd.read_csv(path, header=None)
            else:
                edge_name = fname[:-len(".csv")]
                base_edges[edge_name] = pd.read_csv(path) #, header=None)



    # Enforce: only one label file total
    if len(edge_labels) == 0:
        raise FileNotFoundError("No '*_label.csv' found in edges/. Exactly one label file is required.")
    if len(edge_labels) > 1:
        raise ValueError(f"Found multiple label files: {list(edge_labels.keys())}. Exactly one is allowed.")

    # Build output keys for edges
    for edge_name, df in base_edges.items():
        out[f"edge_index_{edge_name}"] = df.to_numpy(dtype=np.int64).T
        if edge_name in edge_attrs:
            out[f"edge_attr_{edge_name}"] = edge_attrs[edge_name].to_numpy(dtype=np.float32)
        if edge_name in edge_feature_mask:
            out[f"edge_feature_mask_{edge_name}"] = edge_feature_mask[edge_name].to_numpy(dtype=np.int32).ravel()
        else:
            # create a must with all zeros
            out[f"edge_feature_mask_{edge_name}"] = np.zeros(edge_attrs[edge_name].shape[1], dtype=np.int32)

        

    # Add the single label file (kept as DataFrame)
    (label_edge_name, label_df), = edge_labels.items()
    out[f"edge_label_{label_edge_name}"] = label_df

    return out
