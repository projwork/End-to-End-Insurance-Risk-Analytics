"""
Data loading and preprocessing module for Insurance Risk Analytics
Handles data import, cleaning, and basic transformations.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import warnings

from .config import DATA_PATH, SAMPLE_SIZE, RANDOM_STATE, PANDAS_DISPLAY_OPTIONS, COLUMN_CATEGORIES

warnings.filterwarnings('ignore')

class InsuranceDataLoader:
    """
    Data loader class for insurance risk analytics.
    Handles data loading, preprocessing, and feature engineering.
    """
    
    def __init__(self, data_path=None, sample_size=None):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the data file
            sample_size (int): Number of rows to sample
        """
        self.data_path = data_path or DATA_PATH
        self.sample_size = sample_size or SAMPLE_SIZE
        self.df = None
        self._setup_pandas_options()
    
    def _setup_pandas_options(self):
        """Configure pandas display options."""
        for option, value in PANDAS_DISPLAY_OPTIONS.items():
            pd.set_option(option, value)
    
    def get_file_info(self):
        """
        Get information about the data file.
        
        Returns:
            dict: File information including size and estimated records
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        file_size_mb = os.path.getsize(self.data_path) / (1024 * 1024)
        
        return {
            'file_size_mb': file_size_mb,
            'file_path': self.data_path,
            'sample_size': self.sample_size,
            'sample_percentage': (self.sample_size / (file_size_mb * 1024 * 20)) * 100
        }
    
    def load_data(self, delimiter='|', encoding='utf-8'):
        """
        Load the insurance data from file.
        
        Args:
            delimiter (str): Column delimiter
            encoding (str): File encoding
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading {self.sample_size:,} rows from {self.data_path}...")
        
        try:
            self.df = pd.read_csv(
                self.data_path, 
                delimiter=delimiter, 
                nrows=self.sample_size,
                encoding=encoding
            )
            print(f"✅ Data loaded successfully! Shape: {self.df.shape}")
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self):
        """
        Preprocess the loaded data with feature engineering.
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        if self.df is None:
            raise ValueError("Data must be loaded first. Call load_data() method.")
        
        print("🔄 Preprocessing data...")
        
        # Convert datetime columns
        self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'])
        
        # Create date features
        self.df['Year'] = self.df['TransactionMonth'].dt.year
        self.df['Month'] = self.df['TransactionMonth'].dt.month
        self.df['Quarter'] = self.df['TransactionMonth'].dt.quarter
        self.df['Month_Name'] = self.df['TransactionMonth'].dt.month_name()
        
        # Calculate Loss Ratio (Key KPI)
        self.df['LossRatio'] = np.where(
            self.df['TotalPremium'] > 0, 
            self.df['TotalClaims'] / self.df['TotalPremium'], 
            np.nan
        )
        
        # Add risk categories
        self.df['RiskCategory'] = pd.cut(
            self.df['LossRatio'],
            bins=[-np.inf, 0.8, 1.0, np.inf],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Add claims indicator
        self.df['HasClaims'] = (self.df['TotalClaims'] > 0).astype(int)
        
        print("✅ Data preprocessing completed!")
        return self.df
    
    def get_data_overview(self):
        """
        Get comprehensive overview of the loaded data.
        
        Returns:
            dict: Data overview statistics
        """
        if self.df is None:
            raise ValueError("Data must be loaded first.")
        
        overview = {
            'shape': self.df.shape,
            'columns_count': self.df.shape[1],
            'rows_count': self.df.shape[0],
            'period_start': self.df['TransactionMonth'].min(),
            'period_end': self.df['TransactionMonth'].max(),
            'months_covered': self.df['TransactionMonth'].nunique(),
            'unique_policies': self.df['PolicyID'].nunique(),
            'average_loss_ratio': self.df['LossRatio'].mean(),
            'loss_ratio_records': self.df['LossRatio'].notna().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        return overview
    
    def get_column_info(self):
        """
        Get detailed information about columns and their categories.
        
        Returns:
            dict: Column information by category
        """
        if self.df is None:
            raise ValueError("Data must be loaded first.")
        
        column_info = {}
        
        for category, columns in COLUMN_CATEGORIES.items():
            available_columns = [col for col in columns if col in self.df.columns]
            column_info[category] = {
                'columns': available_columns,
                'count': len(available_columns),
                'missing_columns': [col for col in columns if col not in self.df.columns]
            }
        
        # Add data types information
        column_info['data_types'] = {
            'numerical': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical': list(self.df.select_dtypes(include=['object']).columns),
            'datetime': list(self.df.select_dtypes(include=['datetime64']).columns)
        }
        
        return column_info
    
    def print_data_summary(self):
        """Print a comprehensive summary of the loaded data."""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return
        
        file_info = self.get_file_info()
        overview = self.get_data_overview()
        
        print("="*80)
        print("📊 INSURANCE DATA SUMMARY")
        print("="*80)
        
        print(f"\n📁 FILE INFORMATION:")
        print(f"   File Size: {file_info['file_size_mb']:.2f} MB")
        print(f"   Sample Size: {file_info['sample_size']:,} rows")
        print(f"   Sample %: {file_info['sample_percentage']:.2f}%")
        
        print(f"\n📈 DATA OVERVIEW:")
        print(f"   Shape: {overview['shape']}")
        print(f"   Period: {overview['period_start'].strftime('%B %Y')} to {overview['period_end'].strftime('%B %Y')}")
        print(f"   Months: {overview['months_covered']}")
        print(f"   Unique Policies: {overview['unique_policies']:,}")
        print(f"   Average Loss Ratio: {overview['average_loss_ratio']:.4f}")
        print(f"   Memory Usage: {overview['memory_usage_mb']:.2f} MB")
        
        print(f"\n📋 COLUMN CATEGORIES:")
        column_info = self.get_column_info()
        for category, info in column_info.items():
            if category != 'data_types':
                print(f"   {category.title()}: {info['count']} columns")

def load_insurance_data(data_path=None, sample_size=None, preprocess=True):
    """
    Convenience function to load and preprocess insurance data.
    
    Args:
        data_path (str): Path to data file
        sample_size (int): Number of rows to sample
        preprocess (bool): Whether to preprocess the data
        
    Returns:
        tuple: (dataframe, loader_instance)
    """
    loader = InsuranceDataLoader(data_path, sample_size)
    df = loader.load_data()
    
    if preprocess:
        df = loader.preprocess_data()
    
    return df, loader
