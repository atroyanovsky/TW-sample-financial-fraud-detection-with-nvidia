"""Configuration for the financial fraud detection pipeline."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class GNNHyperparameters:
    """Hyperparameters for GNN model."""
    hidden_channels: int = 32
    n_hops: int = 2
    layer: str = "SAGEConv"
    dropout_prob: float = 0.1
    batch_size: int = 4096
    fan_out: int = 10
    num_epochs: int = 20


@dataclass
class XGBHyperparameters:
    """Hyperparameters for XGBoost model."""
    max_depth: int = 6
    learning_rate: float = 0.2
    num_parallel_tree: int = 3
    num_boost_round: int = 512
    gamma: float = 0.0


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    kind: str = "GNN_XGBoost"
    gpu: str = "single"
    gnn: GNNHyperparameters = field(default_factory=GNNHyperparameters)
    xgb: XGBHyperparameters = field(default_factory=XGBHyperparameters)


@dataclass
class TrainingConfig:
    """Configuration for training paths and models."""
    # Paths (SageMaker defaults)
    data_dir: str = "/opt/ml/input/data/data"
    output_dir: str = "/opt/ml/model"
    
    # Models
    models: List[ModelConfig] = field(default_factory=lambda: [ModelConfig()])


@dataclass
class DataConfig:
    """Configuration for data processing."""
    # Input/output
    input_path: str
    output_dir: str = "processed_data"
    
    # Data splitting
    test_split_year: int = 2019
    
    # Undersampling
    under_sample: bool = True
    fraud_ratio: float = 0.1
    random_seed: int = 42
    
    # Feature columns
    id_columns: List[str] = None
    categorical_features: List[str] = None
    numerical_features: List[str] = None
    
    def __post_init__(self):
        # Column name constants
        self.COL_USER = "User"
        self.COL_CARD = "Card"
        self.COL_AMOUNT = "Amount"
        self.COL_MCC = "MCC"
        self.COL_TIME = "Time"
        self.COL_DAY = "Day"
        self.COL_MONTH = "Month"
        self.COL_YEAR = "Year"
        self.COL_MERCHANT = "Merchant"
        self.COL_STATE = "State"
        self.COL_CITY = "City"
        self.COL_ZIP = "Zip"
        self.COL_ERROR = "Errors"
        self.COL_CHIP = "Chip"
        self.COL_FRAUD = "Fraud"
        self.COL_TRANSACTION_ID = "Tx_ID"
        self.COL_MERCHANT_ID = "Merchant_ID"
        self.COL_USER_ID = "User_ID"
        
        # Other constants
        self.UNKNOWN_STRING_MARKER = "XX"
        self.UNKNOWN_ZIP_CODE = 0
        self.COL_GRAPH_SRC = "src"
        self.COL_GRAPH_DST = "dst"
        self.COL_GRAPH_WEIGHT = "wgt"
        
        # Column groups
        if self.id_columns is None:
            self.id_columns = [self.COL_MERCHANT, self.COL_CARD, self.COL_MCC]
        if self.categorical_features is None:
            self.categorical_features = [self.COL_ERROR, self.COL_CHIP, 
                                      self.COL_CITY, self.COL_ZIP]
        if self.numerical_features is None:
            self.numerical_features = [self.COL_AMOUNT]
            
        # Derived constants
        self.MERCHANT_AND_USER_COLS = [self.COL_MERCHANT, self.COL_CARD, self.COL_MCC]
        
        # Column name mappings for raw data
        self.RAW_COLUMN_MAPPING = {
            "Merchant Name": self.COL_MERCHANT,
            "Merchant State": self.COL_STATE,
            "Merchant City": self.COL_CITY,
            "Errors?": self.COL_ERROR,
            "Use Chip": self.COL_CHIP,
            "Is Fraud?": self.COL_FRAUD
        }

# Create default configuration
# Note: test_split_year=2018 means:
#   - Training: data before 2018
#   - Validation: data during 2018
#   - Test: data after 2018
default_config = DataConfig(
    input_path="",  # Will be set at runtime
    output_dir="processed_data",
    test_split_year=2018,
    under_sample=True,
    fraud_ratio=0.1,
    random_seed=42,
    id_columns=None,  # Will be set in __post_init__
    categorical_features=None,  # Will be set in __post_init__
    numerical_features=None  # Will be set in __post_init__
)

# Initialize the config to set up all constants
default_config.__post_init__()

# Default training configuration (from config.yaml)
default_training_config = TrainingConfig(
    data_dir="/opt/ml/input/data/data",
    output_dir="/opt/ml/model",
    models=[
        ModelConfig(
            kind="GNN_XGBoost",
            gpu="single",
            gnn=GNNHyperparameters(
                hidden_channels=32,
                n_hops=2,
                layer="SAGEConv",
                dropout_prob=0.1,
                batch_size=4096,
                fan_out=10,
                num_epochs=20
            ),
            xgb=XGBHyperparameters(
                max_depth=6,
                learning_rate=0.2,
                num_parallel_tree=3,
                num_boost_round=512,
                gamma=0.0
            )
        )
    ]
)
