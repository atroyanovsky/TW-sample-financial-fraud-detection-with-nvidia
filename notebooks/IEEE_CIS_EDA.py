"""
IEEE_CIS_EDA.py
---------------
Exploratory Data Analysis script for the IEEE-CIS Fraud Detection dataset.

This script provides comprehensive EDA including:
- Data overview and summary statistics
- Missing value analysis
- Distribution analysis (numerical and categorical)
- Fraud class balance analysis
- Correlation analysis
- Temporal patterns
- Feature importance insights
- Interactive visualizations

Usage:
    python IEEE_CIS_EDA.py --data-path data/IEEE_CIS [--sample 0.1] [--output-dir eda_output]
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Try to import plotly for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Interactive visualizations will be disabled.")

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ieee_cis_data(
    base_path: str,
    transaction_csv: str = "train_transaction.csv",
    identity_csv: str = "train_identity.csv",
    sample_fraction: Optional[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction and identity data."""
    
    print("=" * 60)
    print("LOADING IEEE-CIS FRAUD DETECTION DATASET")
    print("=" * 60)
    
    # Load transaction data
    transaction_path = os.path.join(base_path, "raw", transaction_csv)
    print(f"\nLoading transactions from: {transaction_path}")
    
    dtype_dict = {f'V{i}': 'float32' for i in range(1, 340)}
    dtype_dict.update({f'C{i}': 'float32' for i in range(1, 15)})
    dtype_dict.update({f'D{i}': 'float32' for i in range(1, 16)})
    dtype_dict.update({f'M{i}': 'object' for i in range(1, 10)})
    dtype_dict['TransactionAmt'] = 'float32'
    dtype_dict['TransactionDT'] = 'int64'
    dtype_dict['isFraud'] = 'int8'
    
    transactions = pd.read_csv(transaction_path, dtype=dtype_dict)
    print(f"  Loaded {len(transactions):,} transactions")
    
    # Load identity data
    identity_path = os.path.join(base_path, "raw", identity_csv)
    print(f"Loading identity from: {identity_path}")
    identity = pd.read_csv(identity_path)
    print(f"  Loaded {len(identity):,} identity records")
    
    # Merge
    print("\nMerging datasets...")
    data = transactions.merge(identity, on='TransactionID', how='left')
    print(f"  Merged data: {len(data):,} rows, {len(data.columns)} columns")
    
    del transactions, identity
    
    # Optional sampling
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"\nSampling {sample_fraction*100:.1f}% of data...")
        data = data.sample(frac=sample_fraction, random_state=random_state)
        data = data.reset_index(drop=True)
        print(f"  Sampled data: {len(data):,} rows")
    
    # Add datetime column
    reference_date = datetime(2017, 11, 30)
    data['datetime'] = pd.to_datetime(
        data['TransactionDT'].apply(lambda x: reference_date + timedelta(seconds=int(x)))
    )
    
    return data


# =============================================================================
# DATA OVERVIEW
# =============================================================================

def print_data_overview(data: pd.DataFrame) -> None:
    """Print comprehensive data overview."""
    
    print("\n" + "=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    
    print(f"\nShape: {data.shape[0]:,} rows × {data.shape[1]} columns")
    print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Column types
    print("\nColumn Types:")
    type_counts = data.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count} columns")
    
    # Categorize columns
    v_cols = [c for c in data.columns if c.startswith('V')]
    c_cols = [c for c in data.columns if c.startswith('C') and c[1:].isdigit()]
    d_cols = [c for c in data.columns if c.startswith('D') and c[1:].isdigit()]
    m_cols = [c for c in data.columns if c.startswith('M') and c[1:].isdigit()]
    id_cols = [c for c in data.columns if c.startswith('id_')]
    
    print("\nColumn Groups:")
    print(f"  V features (Vesta): {len(v_cols)} columns (V1-V339)")
    print(f"  C features (counting): {len(c_cols)} columns")
    print(f"  D features (timedelta): {len(d_cols)} columns")
    print(f"  M features (match): {len(m_cols)} columns")
    print(f"  Identity features: {len(id_cols)} columns")
    
    # Date range
    if 'datetime' in data.columns:
        print(f"\nDate Range:")
        print(f"  From: {data['datetime'].min()}")
        print(f"  To:   {data['datetime'].max()}")
        print(f"  Duration: {(data['datetime'].max() - data['datetime'].min()).days} days")


def get_summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for numerical columns."""
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_df = data[numerical_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    stats_df['missing'] = data[numerical_cols].isnull().sum()
    stats_df['missing_pct'] = (stats_df['missing'] / len(data) * 100).round(2)
    stats_df['unique'] = data[numerical_cols].nunique()
    stats_df['skew'] = data[numerical_cols].skew()
    stats_df['kurtosis'] = data[numerical_cols].kurtosis()
    
    return stats_df


# =============================================================================
# MISSING VALUE ANALYSIS
# =============================================================================

def analyze_missing_values(data: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """Analyze and visualize missing values."""
    
    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    
    # Calculate missing values
    missing = pd.DataFrame({
        'column': data.columns,
        'missing_count': data.isnull().sum().values,
        'missing_pct': (data.isnull().sum().values / len(data) * 100)
    })
    missing = missing.sort_values('missing_pct', ascending=False)
    missing = missing[missing['missing_count'] > 0]
    
    print(f"\nColumns with missing values: {len(missing)} / {len(data.columns)}")
    print(f"Total missing cells: {data.isnull().sum().sum():,}")
    
    # Top missing columns
    print("\nTop 20 columns by missing %:")
    print(missing.head(20).to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart of top 30 missing columns
    top_missing = missing.head(30)
    ax1 = axes[0]
    bars = ax1.barh(range(len(top_missing)), top_missing['missing_pct'].values, color='coral')
    ax1.set_yticks(range(len(top_missing)))
    ax1.set_yticklabels(top_missing['column'].values, fontsize=8)
    ax1.set_xlabel('Missing %')
    ax1.set_title('Top 30 Columns by Missing Values', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Missing value distribution histogram
    ax2 = axes[1]
    ax2.hist(missing['missing_pct'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Missing %')
    ax2.set_ylabel('Number of Columns')
    ax2.set_title('Distribution of Missing Value Percentages', fontsize=12, fontweight='bold')
    ax2.axvline(missing['missing_pct'].median(), color='red', linestyle='--', 
                label=f'Median: {missing["missing_pct"].median():.1f}%')
    ax2.legend()
    
    plt.tight_layout()
    
    return missing, fig


def plot_missing_heatmap(data: pd.DataFrame, sample_size: int = 1000) -> plt.Figure:
    """Create a heatmap showing missing value patterns."""
    
    # Sample for visualization
    sample_data = data.sample(min(sample_size, len(data)), random_state=42)
    
    # Select columns with missing values
    cols_with_missing = data.columns[data.isnull().any()].tolist()
    
    if len(cols_with_missing) > 50:
        # Take top 50 by missing percentage
        missing_pct = data[cols_with_missing].isnull().sum() / len(data)
        cols_with_missing = missing_pct.nlargest(50).index.tolist()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create binary missing matrix
    missing_matrix = sample_data[cols_with_missing].isnull().astype(int)
    
    sns.heatmap(missing_matrix.T, cmap='YlOrRd', cbar_kws={'label': 'Missing (1) / Present (0)'},
                ax=ax, yticklabels=True)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Column')
    ax.set_title('Missing Value Pattern Heatmap (Sample)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# TARGET VARIABLE ANALYSIS
# =============================================================================

def analyze_fraud_distribution(data: pd.DataFrame) -> Tuple[dict, plt.Figure]:
    """Analyze the fraud target variable distribution."""
    
    print("\n" + "=" * 60)
    print("FRAUD TARGET ANALYSIS")
    print("=" * 60)
    
    fraud_counts = data['isFraud'].value_counts()
    fraud_pct = data['isFraud'].value_counts(normalize=True) * 100
    
    stats = {
        'total_transactions': len(data),
        'fraud_count': int(fraud_counts.get(1, 0)),
        'non_fraud_count': int(fraud_counts.get(0, 0)),
        'fraud_rate': float(fraud_pct.get(1, 0)),
        'imbalance_ratio': float(fraud_counts.get(0, 1) / max(fraud_counts.get(1, 1), 1))
    }
    
    print(f"\nFraud Distribution:")
    print(f"  Non-Fraud (0): {stats['non_fraud_count']:,} ({100 - stats['fraud_rate']:.2f}%)")
    print(f"  Fraud (1):     {stats['fraud_count']:,} ({stats['fraud_rate']:.2f}%)")
    print(f"  Imbalance Ratio: {stats['imbalance_ratio']:.1f}:1")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Pie chart
    ax1 = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    ax1.pie([stats['non_fraud_count'], stats['fraud_count']], 
            explode=explode, labels=['Non-Fraud', 'Fraud'],
            colors=colors, autopct='%1.2f%%', startangle=90)
    ax1.set_title('Fraud Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2 = axes[1]
    bars = ax2.bar(['Non-Fraud', 'Fraud'], [stats['non_fraud_count'], stats['fraud_count']], 
                   color=colors, edgecolor='black')
    ax2.set_ylabel('Count')
    ax2.set_title('Transaction Counts', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, [stats['non_fraud_count'], stats['fraud_count']]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                f'{val:,}', ha='center', fontsize=10)
    
    # Log scale bar chart
    ax3 = axes[2]
    ax3.bar(['Non-Fraud', 'Fraud'], [stats['non_fraud_count'], stats['fraud_count']], 
            color=colors, edgecolor='black')
    ax3.set_ylabel('Count (log scale)')
    ax3.set_yscale('log')
    ax3.set_title('Transaction Counts (Log Scale)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return stats, fig


# =============================================================================
# NUMERICAL FEATURE ANALYSIS
# =============================================================================

def analyze_transaction_amount(data: pd.DataFrame) -> plt.Figure:
    """Detailed analysis of transaction amount."""
    
    print("\n" + "=" * 60)
    print("TRANSACTION AMOUNT ANALYSIS")
    print("=" * 60)
    
    amount = data['TransactionAmt']
    fraud = data['isFraud']
    
    print(f"\nOverall Statistics:")
    print(f"  Mean:   ${amount.mean():.2f}")
    print(f"  Median: ${amount.median():.2f}")
    print(f"  Std:    ${amount.std():.2f}")
    print(f"  Min:    ${amount.min():.2f}")
    print(f"  Max:    ${amount.max():.2f}")
    
    print(f"\nBy Fraud Status:")
    for label, name in [(0, 'Non-Fraud'), (1, 'Fraud')]:
        subset = amount[fraud == label]
        print(f"  {name}: Mean=${subset.mean():.2f}, Median=${subset.median():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Distribution
    ax = axes[0, 0]
    ax.hist(amount.clip(upper=amount.quantile(0.99)), bins=100, 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Transaction Amount ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Transaction Amount Distribution\n(clipped at 99th percentile)', fontweight='bold')
    
    # Log distribution
    ax = axes[0, 1]
    ax.hist(np.log1p(amount), bins=100, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Log(Amount + 1)')
    ax.set_ylabel('Frequency')
    ax.set_title('Log-Transformed Amount Distribution', fontweight='bold')
    
    # Box plot by fraud
    ax = axes[0, 2]
    fraud_amounts = [amount[fraud == 0], amount[fraud == 1]]
    bp = ax.boxplot(fraud_amounts, labels=['Non-Fraud', 'Fraud'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_ylabel('Transaction Amount ($)')
    ax.set_title('Amount by Fraud Status', fontweight='bold')
    ax.set_yscale('log')
    
    # Density by fraud status
    ax = axes[1, 0]
    for label, name, color in [(0, 'Non-Fraud', '#2ecc71'), (1, 'Fraud', '#e74c3c')]:
        subset = np.log1p(amount[fraud == label])
        sns.kdeplot(subset, ax=ax, label=name, color=color, fill=True, alpha=0.3)
    ax.set_xlabel('Log(Amount + 1)')
    ax.set_ylabel('Density')
    ax.set_title('Amount Density by Fraud Status', fontweight='bold')
    ax.legend()
    
    # Amount percentiles
    ax = axes[1, 1]
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = [amount.quantile(p/100) for p in percentiles]
    ax.bar(range(len(percentiles)), pct_values, color='teal', edgecolor='black')
    ax.set_xticks(range(len(percentiles)))
    ax.set_xticklabels([f'{p}%' for p in percentiles])
    ax.set_ylabel('Amount ($)')
    ax.set_title('Amount Percentiles', fontweight='bold')
    ax.set_yscale('log')
    
    # Fraud rate by amount bin
    ax = axes[1, 2]
    data_temp = data[['TransactionAmt', 'isFraud']].copy()
    data_temp['amount_bin'] = pd.qcut(data_temp['TransactionAmt'], q=10, duplicates='drop')
    fraud_rate_by_bin = data_temp.groupby('amount_bin')['isFraud'].mean() * 100
    fraud_rate_by_bin.plot(kind='bar', ax=ax, color='purple', edgecolor='black')
    ax.set_xlabel('Amount Bin')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('Fraud Rate by Amount Bin', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def analyze_v_features(data: pd.DataFrame) -> plt.Figure:
    """Analyze V features (Vesta engineered features)."""
    
    print("\n" + "=" * 60)
    print("V FEATURES ANALYSIS (Vesta Engineered)")
    print("=" * 60)
    
    v_cols = [c for c in data.columns if c.startswith('V')]
    print(f"\nNumber of V features: {len(v_cols)}")
    
    # Missing value analysis for V features
    v_missing = data[v_cols].isnull().sum() / len(data) * 100
    print(f"\nV Features Missing Value Summary:")
    print(f"  Mean missing %: {v_missing.mean():.1f}%")
    print(f"  Max missing %:  {v_missing.max():.1f}%")
    print(f"  Features with >50% missing: {(v_missing > 50).sum()}")
    
    # Correlation with fraud for V features
    fraud = data['isFraud']
    v_fraud_corr = {}
    for col in v_cols:
        if data[col].notna().sum() > 100:
            corr = data[col].corr(fraud)
            if not np.isnan(corr):
                v_fraud_corr[col] = corr
    
    # Top correlated V features
    sorted_corr = sorted(v_fraud_corr.items(), key=lambda x: abs(x[1]), reverse=True)
    print(f"\nTop 10 V features by absolute correlation with fraud:")
    for col, corr in sorted_corr[:10]:
        print(f"  {col}: {corr:+.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Missing values
    ax = axes[0, 0]
    ax.hist(v_missing, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Missing %')
    ax.set_ylabel('Number of V Features')
    ax.set_title('Missing Value Distribution in V Features', fontweight='bold')
    
    # Correlation distribution
    ax = axes[0, 1]
    corr_values = list(v_fraud_corr.values())
    ax.hist(corr_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Correlation with Fraud')
    ax.set_ylabel('Number of V Features')
    ax.set_title('V Features Correlation with Fraud', fontweight='bold')
    
    # Top positive correlations
    ax = axes[1, 0]
    top_pos = sorted_corr[:15]
    cols = [x[0] for x in top_pos]
    vals = [x[1] for x in top_pos]
    colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]
    ax.barh(range(len(cols)), vals, color=colors)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel('Correlation with Fraud')
    ax.set_title('Top 15 V Features by Correlation', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    
    # Sample V feature distributions by fraud
    ax = axes[1, 1]
    if len(sorted_corr) > 0:
        top_v = sorted_corr[0][0]
        for label, name, color in [(0, 'Non-Fraud', '#2ecc71'), (1, 'Fraud', '#e74c3c')]:
            subset = data[data['isFraud'] == label][top_v].dropna()
            if len(subset) > 0:
                sns.kdeplot(subset.clip(subset.quantile(0.01), subset.quantile(0.99)), 
                           ax=ax, label=name, color=color, fill=True, alpha=0.3)
        ax.set_xlabel(top_v)
        ax.set_ylabel('Density')
        ax.set_title(f'{top_v} Distribution by Fraud Status\n(highest correlation)', fontweight='bold')
        ax.legend()
    
    plt.tight_layout()
    return fig


# =============================================================================
# CATEGORICAL FEATURE ANALYSIS
# =============================================================================

def analyze_categorical_features(data: pd.DataFrame) -> plt.Figure:
    """Analyze key categorical features."""
    
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURE ANALYSIS")
    print("=" * 60)
    
    cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 
                'DeviceType', 'DeviceInfo']
    cat_cols = [c for c in cat_cols if c in data.columns]
    
    for col in cat_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {data[col].nunique()}")
        print(f"  Missing: {data[col].isnull().sum()} ({data[col].isnull().mean()*100:.1f}%)")
        print(f"  Top 3 values: {data[col].value_counts().head(3).to_dict()}")
    
    # Visualization
    n_cols = min(len(cat_cols), 6)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(cat_cols[:6]):
        ax = axes[idx]
        
        # Calculate fraud rate by category
        fraud_rate = data.groupby(col)['isFraud'].agg(['mean', 'count'])
        fraud_rate = fraud_rate.sort_values('count', ascending=False).head(10)
        
        # Plot
        x = range(len(fraud_rate))
        bars = ax.bar(x, fraud_rate['mean'] * 100, color='steelblue', edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fraud_rate.index, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_title(f'Fraud Rate by {col}', fontweight='bold')
        
        # Add count labels
        for i, (_, row) in enumerate(fraud_rate.iterrows()):
            ax.text(i, row['mean']*100 + 0.5, f'n={int(row["count"]):,}', 
                   ha='center', fontsize=7, rotation=90)
    
    # Hide unused axes
    for idx in range(len(cat_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def analyze_product_and_card(data: pd.DataFrame) -> plt.Figure:
    """Detailed analysis of ProductCD and card features."""
    
    print("\n" + "=" * 60)
    print("PRODUCT AND CARD ANALYSIS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # ProductCD distribution
    ax = axes[0, 0]
    product_counts = data['ProductCD'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(product_counts)))
    ax.pie(product_counts, labels=product_counts.index, colors=colors, autopct='%1.1f%%')
    ax.set_title('ProductCD Distribution', fontweight='bold')
    
    # ProductCD fraud rate
    ax = axes[0, 1]
    product_fraud = data.groupby('ProductCD')['isFraud'].mean() * 100
    bars = ax.bar(product_fraud.index, product_fraud.values, color='coral', edgecolor='black')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('Fraud Rate by ProductCD', fontweight='bold')
    for bar, val in zip(bars, product_fraud.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
               f'{val:.1f}%', ha='center', fontsize=9)
    
    # Card4 (card brand)
    ax = axes[0, 2]
    if 'card4' in data.columns:
        card4_fraud = data.groupby('card4')['isFraud'].mean() * 100
        card4_fraud = card4_fraud.sort_values(ascending=False)
        ax.barh(range(len(card4_fraud)), card4_fraud.values, color='teal', edgecolor='black')
        ax.set_yticks(range(len(card4_fraud)))
        ax.set_yticklabels(card4_fraud.index)
        ax.set_xlabel('Fraud Rate (%)')
        ax.set_title('Fraud Rate by Card Brand (card4)', fontweight='bold')
    
    # Card6 (card type)
    ax = axes[1, 0]
    if 'card6' in data.columns:
        card6_counts = data['card6'].value_counts()
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f'][:len(card6_counts)]
        ax.pie(card6_counts, labels=card6_counts.index, colors=colors, autopct='%1.1f%%')
        ax.set_title('Card Type (card6) Distribution', fontweight='bold')
    
    # Card6 fraud rate
    ax = axes[1, 1]
    if 'card6' in data.columns:
        card6_fraud = data.groupby('card6')['isFraud'].mean() * 100
        bars = ax.bar(card6_fraud.index, card6_fraud.values, color='purple', edgecolor='black')
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_title('Fraud Rate by Card Type', fontweight='bold')
    
    # Cross-tabulation heatmap: ProductCD vs card6
    ax = axes[1, 2]
    if 'card6' in data.columns:
        cross_fraud = data.pivot_table(values='isFraud', index='ProductCD', 
                                        columns='card6', aggfunc='mean') * 100
        sns.heatmap(cross_fraud, annot=True, fmt='.1f', cmap='Reds', ax=ax, 
                   cbar_kws={'label': 'Fraud Rate (%)'})
        ax.set_title('Fraud Rate: ProductCD × Card Type', fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

def analyze_temporal_patterns(data: pd.DataFrame) -> plt.Figure:
    """Analyze temporal patterns in transactions and fraud."""
    
    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS")
    print("=" * 60)
    
    # Extract time features
    data_temp = data[['datetime', 'TransactionAmt', 'isFraud']].copy()
    data_temp['hour'] = data_temp['datetime'].dt.hour
    data_temp['day_of_week'] = data_temp['datetime'].dt.dayofweek
    data_temp['day_of_month'] = data_temp['datetime'].dt.day
    data_temp['date'] = data_temp['datetime'].dt.date
    
    print(f"\nHourly Pattern:")
    hourly_fraud = data_temp.groupby('hour')['isFraud'].mean() * 100
    print(f"  Peak fraud hour: {hourly_fraud.idxmax()} ({hourly_fraud.max():.2f}%)")
    print(f"  Lowest fraud hour: {hourly_fraud.idxmin()} ({hourly_fraud.min():.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Transactions over time
    ax = axes[0, 0]
    daily_txn = data_temp.groupby('date').size()
    ax.plot(daily_txn.index, daily_txn.values, color='steelblue', alpha=0.7)
    ax.fill_between(daily_txn.index, daily_txn.values, alpha=0.3)
    ax.set_xlabel('Date')
    ax.set_ylabel('Transaction Count')
    ax.set_title('Daily Transaction Volume', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Fraud rate over time
    ax = axes[0, 1]
    daily_fraud = data_temp.groupby('date')['isFraud'].mean() * 100
    ax.plot(daily_fraud.index, daily_fraud.values, color='#e74c3c', alpha=0.7)
    ax.fill_between(daily_fraud.index, daily_fraud.values, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('Daily Fraud Rate', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    
    # Hourly pattern
    ax = axes[0, 2]
    hourly_stats = data_temp.groupby('hour').agg({
        'isFraud': 'mean',
        'TransactionAmt': 'count'
    })
    ax2 = ax.twinx()
    ax.bar(hourly_stats.index, hourly_stats['isFraud'] * 100, alpha=0.7, 
           color='coral', label='Fraud Rate')
    ax2.plot(hourly_stats.index, hourly_stats['TransactionAmt'], 
             color='steelblue', linewidth=2, marker='o', label='Volume')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Fraud Rate (%)', color='coral')
    ax2.set_ylabel('Transaction Count', color='steelblue')
    ax.set_title('Hourly Patterns', fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    
    # Day of week pattern
    ax = axes[1, 0]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_fraud = data_temp.groupby('day_of_week')['isFraud'].mean() * 100
    bars = ax.bar(dow_names, dow_fraud.values, color='teal', edgecolor='black')
    ax.set_ylabel('Fraud Rate (%)')
    ax.set_title('Fraud Rate by Day of Week', fontweight='bold')
    
    # Transaction amount by hour
    ax = axes[1, 1]
    hourly_amt = data_temp.groupby('hour')['TransactionAmt'].agg(['mean', 'median'])
    ax.plot(hourly_amt.index, hourly_amt['mean'], marker='o', label='Mean', color='steelblue')
    ax.plot(hourly_amt.index, hourly_amt['median'], marker='s', label='Median', color='coral')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Transaction Amount ($)')
    ax.set_title('Transaction Amount by Hour', fontweight='bold')
    ax.legend()
    ax.set_xticks(range(0, 24, 2))
    
    # Fraud rate heatmap: hour vs day of week
    ax = axes[1, 2]
    heatmap_data = data_temp.pivot_table(values='isFraud', index='day_of_week', 
                                          columns='hour', aggfunc='mean') * 100
    heatmap_data.index = dow_names
    sns.heatmap(heatmap_data, cmap='Reds', ax=ax, cbar_kws={'label': 'Fraud Rate (%)'})
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_title('Fraud Rate: Day × Hour', fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(data: pd.DataFrame, top_n: int = 30) -> Tuple[pd.DataFrame, plt.Figure]:
    """Analyze feature correlations with fraud target."""
    
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Select numerical columns
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'isFraud' and c != 'TransactionID']
    
    # Calculate correlations with fraud
    fraud_corr = {}
    for col in num_cols:
        if data[col].notna().sum() > 100:
            corr = data[col].corr(data['isFraud'])
            if not np.isnan(corr):
                fraud_corr[col] = corr
    
    # Sort by absolute correlation
    sorted_corr = sorted(fraud_corr.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {top_n} features by absolute correlation with fraud:")
    for col, corr in sorted_corr[:top_n]:
        print(f"  {col:30s}: {corr:+.4f}")
    
    # Create DataFrame
    corr_df = pd.DataFrame(sorted_corr, columns=['feature', 'correlation'])
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Top correlations bar chart
    ax = axes[0]
    top_features = corr_df.head(top_n)
    colors = ['#e74c3c' if c > 0 else '#3498db' for c in top_features['correlation']]
    bars = ax.barh(range(len(top_features)), top_features['correlation'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=8)
    ax.set_xlabel('Correlation with Fraud')
    ax.set_title(f'Top {top_n} Features by Correlation', fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    
    # Correlation distribution
    ax = axes[1]
    ax.hist(corr_df['correlation'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Correlation with Fraud')
    ax.set_ylabel('Number of Features')
    ax.set_title('Distribution of Feature Correlations', fontweight='bold')
    
    plt.tight_layout()
    
    return corr_df, fig


def plot_correlation_matrix(data: pd.DataFrame, features: List[str] = None) -> plt.Figure:
    """Plot correlation matrix for selected features."""
    
    if features is None:
        # Select key features
        features = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5',
                   'addr1', 'addr2', 'dist1', 'dist2', 'isFraud']
        features = [f for f in features if f in data.columns]
    
    # Calculate correlation matrix
    corr_matrix = data[features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# =============================================================================
# INTERACTIVE VISUALIZATIONS (Plotly)
# =============================================================================

def create_interactive_dashboard(data: pd.DataFrame, output_dir: str) -> None:
    """Create interactive Plotly visualizations."""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Skipping interactive visualizations.")
        return
    
    print("\n" + "=" * 60)
    print("CREATING INTERACTIVE VISUALIZATIONS")
    print("=" * 60)
    
    # Sample for performance
    sample_size = min(50000, len(data))
    sample_data = data.sample(sample_size, random_state=42)
    
    # 1. Transaction Amount by Fraud Status (Box plot)
    fig1 = px.box(sample_data, x='isFraud', y='TransactionAmt', 
                  color='isFraud', log_y=True,
                  title='Transaction Amount by Fraud Status',
                  labels={'isFraud': 'Is Fraud', 'TransactionAmt': 'Amount ($)'},
                  color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    fig1.write_html(os.path.join(output_dir, 'interactive_amount_boxplot.html'))
    print("  Saved: interactive_amount_boxplot.html")
    
    # 2. ProductCD and Card4 Sunburst
    if 'card4' in sample_data.columns:
        sunburst_data = sample_data.groupby(['ProductCD', 'card4', 'isFraud']).size().reset_index(name='count')
        fig2 = px.sunburst(sunburst_data, path=['ProductCD', 'card4', 'isFraud'], 
                           values='count', title='Transaction Hierarchy: Product → Card Brand → Fraud')
        fig2.write_html(os.path.join(output_dir, 'interactive_sunburst.html'))
        print("  Saved: interactive_sunburst.html")
    
    # 3. Time series with fraud
    sample_data['date'] = sample_data['datetime'].dt.date
    daily_stats = sample_data.groupby('date').agg({
        'isFraud': ['sum', 'mean', 'count']
    }).reset_index()
    daily_stats.columns = ['date', 'fraud_count', 'fraud_rate', 'total_count']
    daily_stats['fraud_rate'] *= 100
    
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=('Daily Transaction Volume', 'Daily Fraud Rate (%)'))
    fig3.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['total_count'],
                              fill='tozeroy', name='Transactions'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=daily_stats['date'], y=daily_stats['fraud_rate'],
                              fill='tozeroy', name='Fraud Rate', line=dict(color='red')), row=2, col=1)
    fig3.update_layout(title='Time Series Analysis', height=600)
    fig3.write_html(os.path.join(output_dir, 'interactive_timeseries.html'))
    print("  Saved: interactive_timeseries.html")
    
    # 4. 3D Scatter (Amount, Time, Card)
    sample_data['hour'] = sample_data['datetime'].dt.hour
    fig4 = px.scatter_3d(sample_data.sample(min(10000, len(sample_data))), 
                         x='TransactionAmt', y='hour', z='card1',
                         color='isFraud', opacity=0.5,
                         title='3D View: Amount × Hour × Card1',
                         color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
    fig4.update_traces(marker=dict(size=3))
    fig4.write_html(os.path.join(output_dir, 'interactive_3d_scatter.html'))
    print("  Saved: interactive_3d_scatter.html")
    
    # 5. Parallel coordinates for top features
    top_features = ['TransactionAmt', 'card1', 'addr1', 'dist1', 'isFraud']
    top_features = [f for f in top_features if f in sample_data.columns]
    sample_subset = sample_data[top_features].dropna().sample(min(5000, len(sample_data)))
    
    fig5 = px.parallel_coordinates(sample_subset, color='isFraud',
                                   title='Parallel Coordinates: Key Features',
                                   color_continuous_scale=[[0, '#2ecc71'], [1, '#e74c3c']])
    fig5.write_html(os.path.join(output_dir, 'interactive_parallel_coords.html'))
    print("  Saved: interactive_parallel_coords.html")


# =============================================================================
# MAIN REPORT GENERATOR
# =============================================================================

def generate_eda_report(
    data: pd.DataFrame,
    output_dir: str = "eda_output"
) -> None:
    """Generate complete EDA report with all visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  IEEE-CIS FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # 1. Data Overview
    print_data_overview(data)
    
    # 2. Summary Statistics
    stats_df = get_summary_statistics(data)
    stats_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    print(f"\nSaved: summary_statistics.csv")
    
    # 3. Missing Value Analysis
    missing_df, fig = analyze_missing_values(data)
    missing_df.to_csv(os.path.join(output_dir, 'missing_values.csv'), index=False)
    fig.savefig(os.path.join(output_dir, 'missing_values.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: missing_values.csv, missing_values.png")
    
    fig = plot_missing_heatmap(data)
    fig.savefig(os.path.join(output_dir, 'missing_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: missing_heatmap.png")
    
    # 4. Fraud Distribution
    fraud_stats, fig = analyze_fraud_distribution(data)
    fig.savefig(os.path.join(output_dir, 'fraud_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: fraud_distribution.png")
    
    # 5. Transaction Amount Analysis
    fig = analyze_transaction_amount(data)
    fig.savefig(os.path.join(output_dir, 'transaction_amount.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: transaction_amount.png")
    
    # 6. V Features Analysis
    fig = analyze_v_features(data)
    fig.savefig(os.path.join(output_dir, 'v_features.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: v_features.png")
    
    # 7. Categorical Features
    fig = analyze_categorical_features(data)
    fig.savefig(os.path.join(output_dir, 'categorical_features.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: categorical_features.png")
    
    fig = analyze_product_and_card(data)
    fig.savefig(os.path.join(output_dir, 'product_card_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: product_card_analysis.png")
    
    # 8. Temporal Analysis
    fig = analyze_temporal_patterns(data)
    fig.savefig(os.path.join(output_dir, 'temporal_patterns.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: temporal_patterns.png")
    
    # 9. Correlation Analysis
    corr_df, fig = analyze_correlations(data)
    corr_df.to_csv(os.path.join(output_dir, 'feature_correlations.csv'), index=False)
    fig.savefig(os.path.join(output_dir, 'correlations.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: feature_correlations.csv, correlations.png")
    
    fig = plot_correlation_matrix(data)
    fig.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: correlation_matrix.png")
    
    # 10. Interactive Visualizations
    create_interactive_dashboard(data, output_dir)
    
    print("\n" + "=" * 70)
    print("  EDA COMPLETE!")
    print(f"  All outputs saved to: {output_dir}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for IEEE-CIS Fraud Detection Dataset"
    )
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
        "--output-dir",
        type=str,
        default="eda_output",
        help="Directory to save EDA outputs"
    )
    
    args = parser.parse_args()
    
    # Load data
    data = load_ieee_cis_data(
        base_path=args.data_path,
        sample_fraction=args.sample
    )
    
    # Generate EDA report
    generate_eda_report(data, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
