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
    A comprehensive data loader for insurance risk analytics with DVC integration.
    Handles data loading, preprocessing, feature engineering, and version management.
    """
    
    def __init__(self, file_path=None, sample_size=None, random_state=42):
        """
        Initialize the data loader.
        
        Args:
            file_path (str): Path to the data file
            sample_size (int): Number of rows to sample
            random_state (int): Random state for reproducibility
        """
        self.file_path = file_path or DATA_PATH
        self.sample_size = sample_size or SAMPLE_SIZE
        self.random_state = random_state or RANDOM_STATE
        self.df = None
        self._setup_pandas_options()
        
        # DVC integration
        self.dvc_available = self._check_dvc_availability()
    
    def _setup_pandas_options(self):
        """Configure pandas display options."""
        for option, value in PANDAS_DISPLAY_OPTIONS.items():
            pd.set_option(option, value)
    
    def _check_dvc_availability(self):
        """Check if DVC is available and configured"""
        try:
            import subprocess
            result = subprocess.run(['dvc', 'version'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def get_available_data_versions(self):
        """Get list of available data versions tracked by DVC"""
        if not self.dvc_available:
            print("⚠️ DVC not available. Cannot list data versions.")
            return []
        
        try:
            import subprocess
            result = subprocess.run(['dvc', 'list', '.', 'data'], capture_output=True, text=True)
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                dvc_files = [f for f in files if f.endswith('.dvc')]
                return dvc_files
            else:
                print("⚠️ Could not list DVC files")
                return []
        except Exception as e:
            print(f"⚠️ Error listing DVC files: {e}")
            return []
    
    def load_data_version(self, version_file):
        """Load a specific data version using DVC"""
        if not self.dvc_available:
            print("⚠️ DVC not available. Loading data directly.")
            return self.load_data()
        
        try:
            # Extract the actual data file name from .dvc file
            data_file = version_file.replace('.dvc', '')
            data_path = os.path.join('data', data_file)
            
            # Ensure data is pulled from DVC
            import subprocess
            subprocess.run(['dvc', 'pull', f'data/{version_file}'], check=True)
            
            print(f"📊 Loading data version: {data_file}")
            
            # Load the data
            df = pd.read_csv(data_path, sep='|', 
                           parse_dates=['TransactionMonth'])
            
            print(f"✅ Successfully loaded {len(df):,} records from {data_file}")
            return df
            
        except Exception as e:
            print(f"⚠️ Error loading DVC version {version_file}: {e}")
            print("Falling back to default data loading...")
            return self.load_data()
    
    def create_data_checkpoint(self, df, version_name, description=""):
        """Create a new data version checkpoint with DVC"""
        if not self.dvc_available:
            print("⚠️ DVC not available. Cannot create checkpoint.")
            return False
        
        try:
            # Create the data file
            checkpoint_path = f'data/checkpoint_{version_name}.txt'
            df.to_csv(checkpoint_path, sep='|', index=False)
            
            # Add to DVC
            import subprocess
            subprocess.run(['dvc', 'add', checkpoint_path], check=True)
            
            print(f"✅ Created data checkpoint: {checkpoint_path}")
            print(f"   Description: {description}")
            print(f"   Records: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print("📝 Remember to commit the .dvc file to Git!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating checkpoint: {e}")
            return False
    
    def get_data_info(self, version_file=None):
        """Get information about a specific data version"""
        if version_file and self.dvc_available:
            try:
                import subprocess
                result = subprocess.run(['dvc', 'get-url', f'data/{version_file}'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"📊 DVC Version Info: {version_file}")
                    # Additional DVC-specific info could be added here
            except:
                pass
        
        # Fall back to standard data info
        return self.get_data_overview()
    
    def get_file_info(self):
        """
        Get information about the data file.
        
        Returns:
            dict: File information including size and estimated records
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file not found at {self.file_path}")
        
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)
        
        return {
            'file_size_mb': file_size_mb,
            'file_path': self.file_path,
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
        print(f"Loading {self.sample_size:,} rows from {self.file_path}...")
        
        try:
            self.df = pd.read_csv(
                self.file_path, 
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
    
    def load_and_prepare_data(self, delimiter='|', encoding='utf-8'):
        """
        Convenience method to load and preprocess data in one step.
        
        Args:
            delimiter (str): Column delimiter
            encoding (str): File encoding
            
        Returns:
            pd.DataFrame: Loaded and preprocessed dataset
        """
        self.load_data(delimiter=delimiter, encoding=encoding)
        return self.preprocess_data()

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
