"""
Data Quality Assessment module for Insurance Risk Analytics
Handles missing value analysis, data consistency checks, and outlier detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import jarque_bera, skew, kurtosis
import warnings

from .config import KEY_FINANCIAL_VARS, OUTLIER_IQR_MULTIPLIER

warnings.filterwarnings('ignore')

class DataQualityAssessor:
    """
    Data quality assessment class for insurance risk analytics.
    Provides comprehensive data quality analysis and reporting.
    """
    
    def __init__(self, df):
        """
        Initialize the data quality assessor.
        
        Args:
            df (pd.DataFrame): The dataset to assess
        """
        self.df = df
        self.quality_report = {}
    
    def assess_missing_values(self):
        """
        Assess missing values in the dataset.
        
        Returns:
            pd.DataFrame: Missing value report
        """
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_report = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        self.quality_report['missing_values'] = {
            'total_columns': len(self.df.columns),
            'columns_with_missing': (missing_data > 0).sum(),
            'total_missing': missing_data.sum(),
            'report': missing_report
        }
        
        return missing_report
    
    def assess_financial_variables(self):
        """
        Assess quality of key financial variables.
        
        Returns:
            dict: Financial variables quality assessment
        """
        financial_quality = {}
        
        for var in KEY_FINANCIAL_VARS:
            if var in self.df.columns:
                data = self.df[var]
                
                financial_quality[var] = {
                    'missing_count': data.isnull().sum(),
                    'missing_percentage': (data.isnull().sum() / len(self.df)) * 100,
                    'zero_count': (data == 0).sum(),
                    'zero_percentage': ((data == 0).sum() / len(self.df)) * 100,
                    'negative_count': (data < 0).sum(),
                    'negative_percentage': ((data < 0).sum() / len(self.df)) * 100,
                    'min_value': data.min() if not data.empty else None,
                    'max_value': data.max() if not data.empty else None,
                    'mean_value': data.mean() if not data.empty else None,
                    'median_value': data.median() if not data.empty else None
                }
        
        self.quality_report['financial_variables'] = financial_quality
        return financial_quality
    
    def detect_outliers(self, variables=None):
        """
        Detect outliers using IQR method.
        
        Args:
            variables (list): Variables to check for outliers
            
        Returns:
            dict: Outlier detection results
        """
        if variables is None:
            variables = [var for var in KEY_FINANCIAL_VARS if var in self.df.columns]
        
        outlier_summary = {}
        
        for var in variables:
            if var in self.df.columns:
                data = self.df[var].dropna()
                
                if len(data) > 0:
                    # Calculate quartiles and IQR
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
                    upper_bound = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
                    
                    # Identify outliers
                    outliers = self.df[(self.df[var] < lower_bound) | (self.df[var] > upper_bound)]
                    lower_outliers = self.df[self.df[var] < lower_bound]
                    upper_outliers = self.df[self.df[var] > upper_bound]
                    
                    outlier_summary[var] = {
                        'Q1': Q1,
                        'Q3': Q3,
                        'IQR': IQR,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'total_outliers': len(outliers),
                        'outlier_percentage': len(outliers) / len(self.df) * 100,
                        'lower_outliers': len(lower_outliers),
                        'upper_outliers': len(upper_outliers),
                        'extreme_values': sorted(upper_outliers[var].tolist(), reverse=True)[:5] if len(upper_outliers) > 0 else []
                    }
        
        self.quality_report['outliers'] = outlier_summary
        return outlier_summary
    
    def generate_comprehensive_analysis(self):
        """
        Generate comprehensive data quality analysis.
        
        Returns:
            dict: Complete quality report
        """
        print("🔍 Running comprehensive data quality analysis...")
        
        # Run all assessments
        self.assess_missing_values()
        self.assess_financial_variables()
        self.detect_outliers()
        
        # Add additional checks
        self.quality_report['data_consistency'] = self._check_data_consistency()
        self.quality_report['temporal_consistency'] = self._check_temporal_consistency()
        
        print("✅ Comprehensive analysis completed!")
        return self.quality_report
    
    def _check_data_consistency(self):
        """Check for data consistency issues."""
        consistency_issues = {}
        
        # Check if TotalClaims > TotalPremium (extreme loss ratios)
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            extreme_loss = self.df[self.df['TotalClaims'] > self.df['TotalPremium']]
            consistency_issues['extreme_loss_ratios'] = {
                'count': len(extreme_loss),
                'percentage': len(extreme_loss) / len(self.df) * 100
            }
        
        # Check for unrealistic registration years
        if 'RegistrationYear' in self.df.columns:
            current_year = datetime.now().year
            unrealistic_years = self.df[
                (self.df['RegistrationYear'] < 1900) | 
                (self.df['RegistrationYear'] > current_year)
            ]
            consistency_issues['unrealistic_registration_years'] = {
                'count': len(unrealistic_years),
                'percentage': len(unrealistic_years) / len(self.df) * 100
            }
        
        return consistency_issues
    
    def _check_temporal_consistency(self):
        """Check temporal data consistency."""
        temporal_issues = {}
        
        if 'TransactionMonth' in self.df.columns:
            # Check for date range
            date_range = {
                'min_date': self.df['TransactionMonth'].min(),
                'max_date': self.df['TransactionMonth'].max(),
                'unique_months': self.df['TransactionMonth'].nunique()
            }
            temporal_issues['date_range'] = date_range
        
        return temporal_issues

    def print_quality_summary(self):
        """Print a summary of data quality findings."""
        if not self.quality_report:
            # Run basic assessments
            self.assess_missing_values()
            self.assess_financial_variables()
            self.detect_outliers()
        
        print("="*80)
        print("📋 DATA QUALITY SUMMARY")
        print("="*80)
        
        # Missing values summary
        if 'missing_values' in self.quality_report:
            mv = self.quality_report['missing_values']
            print(f"\n🔍 MISSING VALUES:")
            print(f"   Total columns: {mv['total_columns']}")
            print(f"   Columns with missing data: {mv['columns_with_missing']}")
            print(f"   Total missing values: {mv['total_missing']:,}")
        
        # Financial variables summary
        if 'financial_variables' in self.quality_report:
            print(f"\n💰 FINANCIAL VARIABLES:")
            for var, quality in self.quality_report['financial_variables'].items():
                print(f"   {var}:")
                print(f"     Missing: {quality['missing_percentage']:.1f}%")
                print(f"     Zero values: {quality['zero_percentage']:.1f}%")
                if quality['negative_count'] > 0:
                    print(f"     Negative values: {quality['negative_percentage']:.1f}%")
        
        # Outliers summary
        if 'outliers' in self.quality_report:
            print(f"\n📊 OUTLIERS:")
            for var, outlier_info in self.quality_report['outliers'].items():
                print(f"   {var}: {outlier_info['total_outliers']:,} ({outlier_info['outlier_percentage']:.1f}%)")
        
        # Data consistency summary
        if 'data_consistency' in self.quality_report:
            dc = self.quality_report['data_consistency']
            print(f"\n⚠️  DATA CONSISTENCY:")
            for issue, details in dc.items():
                if details['count'] > 0:
                    print(f"   {issue.replace('_', ' ').title()}: {details['count']:,} ({details['percentage']:.1f}%)")
        
        # Temporal consistency summary
        if 'temporal_consistency' in self.quality_report:
            tc = self.quality_report['temporal_consistency']
            if 'date_range' in tc:
                dr = tc['date_range']
                print(f"\n📅 TEMPORAL CONSISTENCY:")
                print(f"   Date range: {dr['min_date']} to {dr['max_date']}")
                print(f"   Unique months: {dr['unique_months']}")

def assess_data_quality(df):
    """
    Convenience function to assess data quality.
    
    Args:
        df (pd.DataFrame): Dataset to assess
        
    Returns:
        DataQualityAssessor: Assessor instance with complete report
    """
    assessor = DataQualityAssessor(df)
    assessor.assess_missing_values()
    assessor.assess_financial_variables()
    assessor.detect_outliers()
    return assessor
