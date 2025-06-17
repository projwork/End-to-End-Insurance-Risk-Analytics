"""
Utilities module for Insurance Risk Analytics
Contains helper functions and common utilities used across the project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')

def setup_plotting_environment():
    """Set up the plotting environment with consistent styling."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 10
    
    print("✅ Plotting environment configured successfully!")

def format_currency(amount, currency_symbol='$'):
    """
    Format currency values for display.
    
    Args:
        amount (float): Amount to format
        currency_symbol (str): Currency symbol to use
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(amount):
        return 'N/A'
    
    if abs(amount) >= 1e9:
        return f"{currency_symbol}{amount/1e9:.2f}B"
    elif abs(amount) >= 1e6:
        return f"{currency_symbol}{amount/1e6:.2f}M"
    elif abs(amount) >= 1e3:
        return f"{currency_symbol}{amount/1e3:.2f}K"
    else:
        return f"{currency_symbol}{amount:.2f}"

def format_percentage(value, decimal_places=2):
    """
    Format percentage values for display.
    
    Args:
        value (float): Value to format as percentage
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    if pd.isna(value):
        return 'N/A'
    
    return f"{value * 100:.{decimal_places}f}%"

def format_number(number, decimal_places=0):
    """
    Format large numbers with appropriate suffixes.
    
    Args:
        number (float): Number to format
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted number string
    """
    if pd.isna(number):
        return 'N/A'
    
    if abs(number) >= 1e9:
        return f"{number/1e9:.{decimal_places}f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.{decimal_places}f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.{decimal_places}f}K"
    else:
        return f"{number:,.{decimal_places}f}"

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a dataset.
    
    Args:
        data (array-like): Data values
        confidence (float): Confidence level (default 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    from scipy import stats
    
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    
    if len(data) == 0:
        return (np.nan, np.nan)
    
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    
    # t-distribution critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_of_error = t_critical * se
    
    return (mean - margin_of_error, mean + margin_of_error)

def detect_data_drift(old_data, new_data, threshold=0.1):
    """
    Detect data drift between two datasets.
    
    Args:
        old_data (pd.Series): Historical data
        new_data (pd.Series): New data
        threshold (float): Threshold for significant drift
        
    Returns:
        dict: Drift analysis results
    """
    from scipy import stats
    
    # Basic statistics comparison
    old_mean = old_data.mean()
    new_mean = new_data.mean()
    mean_change = abs((new_mean - old_mean) / old_mean) if old_mean != 0 else float('inf')
    
    old_std = old_data.std()
    new_std = new_data.std()
    std_change = abs((new_std - old_std) / old_std) if old_std != 0 else float('inf')
    
    # Statistical test for distribution difference
    try:
        ks_statistic, ks_p_value = stats.ks_2samp(old_data.dropna(), new_data.dropna())
    except:
        ks_statistic, ks_p_value = np.nan, np.nan
    
    drift_detected = (mean_change > threshold) or (std_change > threshold) or (ks_p_value < 0.05)
    
    return {
        'drift_detected': drift_detected,
        'mean_change': mean_change,
        'std_change': std_change,
        'ks_statistic': ks_statistic,
        'ks_p_value': ks_p_value,
        'old_mean': old_mean,
        'new_mean': new_mean,
        'old_std': old_std,
        'new_std': new_std
    }

def create_summary_table(df, groupby_col, metrics_cols, agg_funcs=['sum', 'mean', 'count']):
    """
    Create a summary table with multiple aggregations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        groupby_col (str): Column to group by
        metrics_cols (list): Columns to aggregate
        agg_funcs (list): Aggregation functions to apply
        
    Returns:
        pd.DataFrame: Summary table
    """
    agg_dict = {}
    for col in metrics_cols:
        if col in df.columns:
            agg_dict[col] = agg_funcs
    
    if not agg_dict:
        return pd.DataFrame()
    
    summary = df.groupby(groupby_col).agg(agg_dict).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary

def print_dataframe_info(df, name="Dataset"):
    """
    Print comprehensive information about a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to analyze
        name (str): Name of the dataset
    """
    print(f"\n{'='*60}")
    print(f"{name.upper()} INFORMATION")
    print(f"{'='*60}")
    
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Duplicated rows: {df.duplicated().sum():,}")
    
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    
    missing_cols = missing_info[missing_info['Count'] > 0]
    if len(missing_cols) > 0:
        print(f"Columns with missing values: {len(missing_cols)}")
        print(missing_cols.head(10))
    else:
        print("No missing values found!")

def validate_data_types(df, expected_types):
    """
    Validate data types in a dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        expected_types (dict): Expected data types {column: dtype}
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'missing_columns': [],
        'type_mismatches': []
    }
    
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            validation_results['missing_columns'].append(col)
            validation_results['valid'] = False
        elif not df[col].dtype == expected_type:
            validation_results['type_mismatches'].append({
                'column': col,
                'expected': expected_type,
                'actual': df[col].dtype
            })
            validation_results['valid'] = False
    
    if validation_results['missing_columns']:
        validation_results['issues'].append(f"Missing columns: {validation_results['missing_columns']}")
    
    if validation_results['type_mismatches']:
        for mismatch in validation_results['type_mismatches']:
            validation_results['issues'].append(
                f"Type mismatch in {mismatch['column']}: expected {mismatch['expected']}, got {mismatch['actual']}"
            )
    
    return validation_results

def create_directory_structure(base_path, directories):
    """
    Create directory structure for the project.
    
    Args:
        base_path (str): Base path for the project
        directories (list): List of directory names to create
    """
    base_path = Path(base_path)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def save_analysis_results(results, filename, output_dir="output"):
    """
    Save analysis results to a file.
    
    Args:
        results (dict): Results to save
        filename (str): Output filename
        output_dir (str): Output directory
    """
    import json
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to results
    results['timestamp'] = datetime.now().isoformat()
    results['version'] = '1.0'
    
    # Save to JSON file
    output_path = Path(output_dir) / filename
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return obj
    
    # Recursively convert numpy types
    def deep_convert(obj):
        if isinstance(obj, dict):
            return {key: deep_convert(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [deep_convert(item) for item in obj]
        else:
            return convert_numpy_types(obj)
    
    converted_results = deep_convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(converted_results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")

def load_analysis_results(filename, input_dir="output"):
    """
    Load analysis results from a file.
    
    Args:
        filename (str): Input filename
        input_dir (str): Input directory
        
    Returns:
        dict: Loaded results
    """
    import json
    
    input_path = Path(input_dir) / filename
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Results loaded from: {input_path}")
    return results

def compare_dataframes(df1, df2, name1="DataFrame 1", name2="DataFrame 2"):
    """
    Compare two dataframes and highlight differences.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        name1 (str): Name for first dataframe
        name2 (str): Name for second dataframe
        
    Returns:
        dict: Comparison results
    """
    comparison = {
        'shape_match': df1.shape == df2.shape,
        'columns_match': list(df1.columns) == list(df2.columns),
        'dtypes_match': df1.dtypes.equals(df2.dtypes),
        'content_match': df1.equals(df2) if df1.shape == df2.shape and list(df1.columns) == list(df2.columns) else False
    }
    
    comparison['summary'] = f"""
    {name1}: {df1.shape}
    {name2}: {df2.shape}
    Shape match: {comparison['shape_match']}
    Columns match: {comparison['columns_match']}
    Data types match: {comparison['dtypes_match']}
    Content match: {comparison['content_match']}
    """
    
    if not comparison['columns_match']:
        comparison['column_differences'] = {
            'only_in_df1': set(df1.columns) - set(df2.columns),
            'only_in_df2': set(df2.columns) - set(df1.columns)
        }
    
    return comparison

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        print(f"🚀 Starting {description}...")
    
    def update(self, step_description=""):
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        bar_length = 30
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{self.description}: |{bar}| {progress:.1f}% {step_description}', end='', flush=True)
        
        if self.current_step == self.total_steps:
            print(f'\n✅ {self.description} completed!')
    
    def finish(self):
        self.current_step = self.total_steps
        self.update("Done!")

# Constants for common use
BUSINESS_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

def print_section_header(title, char='=', width=80):
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"{title.center(width)}")
    print(f"{char * width}")

def print_subsection_header(title, char='-', width=60):
    """Print a formatted subsection header."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")

def safe_divide(numerator, denominator, default=np.nan):
    """Safely divide two numbers, returning default for division by zero."""
    return numerator / denominator if denominator != 0 else default

def chunks(lst, chunk_size):
    """Yield successive chunks from a list."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
