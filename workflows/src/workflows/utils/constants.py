# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Column name constants and markers for TabFormer preprocessing."""

# Original column names (renamed for cleaner access)
COL_USER = "User"
COL_CARD = "Card"
COL_AMOUNT = "Amount"
COL_MCC = "MCC"
COL_TIME = "Time"
COL_DAY = "Day"
COL_MONTH = "Month"
COL_YEAR = "Year"

COL_MERCHANT = "Merchant"
COL_STATE = "State"
COL_CITY = "City"
COL_ZIP = "Zip"
COL_ERROR = "Errors"
COL_CHIP = "Chip"
COL_FRAUD = "Fraud"

# Generated ID columns
COL_TRANSACTION_ID = "Tx_ID"
COL_MERCHANT_ID = "Merchant_ID"
COL_USER_ID = "User_ID"

# Missing value markers
UNKNOWN_STRING_MARKER = "XX"
UNKNOWN_ZIP_CODE = 0

# Graph column names
COL_GRAPH_SRC = "src"
COL_GRAPH_DST = "dst"
COL_GRAPH_WEIGHT = "wgt"

# Columns for binary encoding (merchant/user identifiers)
MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_CARD, COL_MCC]

# Column rename mapping from raw TabFormer CSV
COLUMN_RENAME_MAP = {
    "Merchant Name": COL_MERCHANT,
    "Merchant State": COL_STATE,
    "Merchant City": COL_CITY,
    "Errors?": COL_ERROR,
    "Use Chip": COL_CHIP,
    "Is Fraud?": COL_FRAUD,
}

# Fraud label encoding
FRAUD_TO_BINARY = {"No": 0, "Yes": 1}

# Default split years
TRAIN_YEAR_CUTOFF = 2018  # < 2018 for training
VALIDATION_YEAR = 2018  # == 2018 for validation
# > 2018 for test
