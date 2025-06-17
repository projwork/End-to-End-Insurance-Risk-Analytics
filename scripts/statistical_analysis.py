"""
Statistical Analysis module for Insurance Risk Analytics
Handles descriptive statistics, correlation analysis, and statistical tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, jarque_bera, skew, kurtosis, pearsonr, spearmanr
import warnings

from .config import KEY_FINANCIAL_VARS, KEY_CATEGORICAL_VARS, CORRELATION_THRESHOLD, STATISTICAL_SIGNIFICANCE

warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Statistical analysis class for insurance risk analytics.
    Provides comprehensive statistical analysis and testing.
    """
    
    def __init__(self, df):
        """
        Initialize the statistical analyzer.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
        """
        self.df = df
        self.analysis_results = {}
    
    def descriptive_statistics(self, variables=None):
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            variables (list): Variables to analyze
            
        Returns:
            dict: Descriptive statistics for each variable
        """
        if variables is None:
            variables = [var for var in KEY_FINANCIAL_VARS if var in self.df.columns]
        
        desc_stats = {}
        
        for var in variables:
            if var in self.df.columns:
                data = self.df[var].dropna()
                
                if len(data) > 0:
                    desc_stats[var] = {
                        'count': len(data),
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'q25': data.quantile(0.25),
                        'median': data.median(),
                        'q75': data.quantile(0.75),
                        'max': data.max(),
                        'skewness': skew(data),
                        'kurtosis': kurtosis(data),
                        'coefficient_of_variation': data.std() / data.mean() if data.mean() != 0 else np.nan
                    }
        
        self.analysis_results['descriptive_statistics'] = desc_stats
        return desc_stats
    
    def normality_tests(self, variables=None):
        """
        Perform normality tests on numerical variables.
        
        Args:
            variables (list): Variables to test
            
        Returns:
            dict: Normality test results
        """
        if variables is None:
            variables = [var for var in KEY_FINANCIAL_VARS if var in self.df.columns]
        
        normality_results = {}
        
        for var in variables:
            if var in self.df.columns:
                data = self.df[var].dropna()
                
                if len(data) > 8:  # Minimum sample size for tests
                    # Jarque-Bera test
                    jb_stat, jb_p = jarque_bera(data)
                    
                    # Shapiro-Wilk test (sample if data is too large)
                    if len(data) > 5000:
                        sample_data = data.sample(5000, random_state=42)
                    else:
                        sample_data = data
                    
                    shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                    
                    normality_results[var] = {
                        'jarque_bera_statistic': jb_stat,
                        'jarque_bera_p_value': jb_p,
                        'jarque_bera_normal': jb_p >= STATISTICAL_SIGNIFICANCE,
                        'shapiro_statistic': shapiro_stat,
                        'shapiro_p_value': shapiro_p,
                        'shapiro_normal': shapiro_p >= STATISTICAL_SIGNIFICANCE,
                        'sample_size': len(sample_data)
                    }
        
        self.analysis_results['normality_tests'] = normality_results
        return normality_results
    
    def correlation_analysis(self, method='pearson'):
        """
        Perform correlation analysis on numerical variables.
        
        Args:
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            tuple: (correlation_matrix, significant_correlations)
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove date-related columns
        numerical_cols = [col for col in numerical_cols if col not in ['Year', 'Month', 'Quarter']]
        
        if method == 'pearson':
            corr_matrix = self.df[numerical_cols].corr()
        else:
            corr_matrix = self.df[numerical_cols].corr(method='spearman')
        
        # Find significant correlations
        significant_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) > CORRELATION_THRESHOLD:
                    significant_corrs.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': corr_val,
                        'abs_correlation': abs(corr_val)
                    })
        
        # Sort by absolute correlation
        significant_corrs = sorted(significant_corrs, key=lambda x: x['abs_correlation'], reverse=True)
        
        self.analysis_results[f'{method}_correlation'] = {
            'matrix': corr_matrix,
            'significant_correlations': significant_corrs
        }
        
        return corr_matrix, significant_corrs
    
    def bivariate_analysis(self, pairs=None):
        """
        Perform bivariate analysis on specified variable pairs.
        
        Args:
            pairs (list): List of variable pairs to analyze
            
        Returns:
            dict: Bivariate analysis results
        """
        if pairs is None:
            pairs = [
                ('TotalPremium', 'TotalClaims'),
                ('CustomValueEstimate', 'TotalPremium'),
                ('SumInsured', 'TotalPremium'),
                ('RegistrationYear', 'LossRatio')
            ]
        
        bivariate_results = {}
        
        for var1, var2 in pairs:
            if var1 in self.df.columns and var2 in self.df.columns:
                # Remove missing values
                valid_data = self.df[[var1, var2]].dropna()
                
                if len(valid_data) > 10:
                    # Pearson correlation
                    pearson_corr, pearson_p = pearsonr(valid_data[var1], valid_data[var2])
                    
                    # Spearman correlation
                    spearman_corr, spearman_p = spearmanr(valid_data[var1], valid_data[var2])
                    
                    bivariate_results[f'{var1}_vs_{var2}'] = {
                        'sample_size': len(valid_data),
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'pearson_significant': pearson_p < STATISTICAL_SIGNIFICANCE,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'spearman_significant': spearman_p < STATISTICAL_SIGNIFICANCE
                    }
        
        self.analysis_results['bivariate_analysis'] = bivariate_results
        return bivariate_results
    
    def categorical_analysis(self, variables=None):
        """
        Analyze categorical variables.
        
        Args:
            variables (list): Categorical variables to analyze
            
        Returns:
            dict: Categorical analysis results
        """
        if variables is None:
            variables = [var for var in KEY_CATEGORICAL_VARS if var in self.df.columns]
        
        categorical_results = {}
        
        for var in variables:
            if var in self.df.columns:
                value_counts = self.df[var].value_counts()
                proportions = value_counts / len(self.df)
                
                # Calculate Herfindahl-Hirschman Index (concentration measure)
                hhi = (proportions ** 2).sum()
                
                categorical_results[var] = {
                    'unique_values': self.df[var].nunique(),
                    'value_counts': value_counts.to_dict(),
                    'proportions': proportions.to_dict(),
                    'top_value': value_counts.index[0],
                    'top_value_percentage': (value_counts.iloc[0] / len(self.df)) * 100,
                    'concentration_index': hhi,
                    'entropy': stats.entropy(proportions)
                }
        
        self.analysis_results['categorical_analysis'] = categorical_results
        return categorical_results
    
    def temporal_analysis(self):
        """
        Perform temporal analysis on the dataset.
        
        Returns:
            dict: Temporal analysis results
        """
        if 'TransactionMonth' not in self.df.columns:
            return {}
        
        # Monthly aggregation
        monthly_trends = self.df.groupby('TransactionMonth').agg({
            'TotalPremium': ['sum', 'mean', 'count'],
            'TotalClaims': ['sum', 'mean'],
            'LossRatio': 'mean',
            'PolicyID': 'nunique'
        }).round(4)
        
        # Flatten column names
        monthly_trends.columns = ['_'.join(col).strip() for col in monthly_trends.columns.values]
        monthly_trends = monthly_trends.reset_index()
        
        # Calculate monthly loss ratio based on totals
        monthly_trends['Monthly_LossRatio'] = monthly_trends['TotalClaims_sum'] / monthly_trends['TotalPremium_sum']
        
        # Calculate trend statistics
        first_month = monthly_trends.iloc[0]
        last_month = monthly_trends.iloc[-1]
        
        temporal_results = {
            'monthly_data': monthly_trends,
            'period_start': monthly_trends['TransactionMonth'].min(),
            'period_end': monthly_trends['TransactionMonth'].max(),
            'total_months': len(monthly_trends),
            'avg_monthly_premium': monthly_trends['TotalPremium_sum'].mean(),
            'avg_monthly_claims': monthly_trends['TotalClaims_sum'].mean(),
            'avg_monthly_loss_ratio': monthly_trends['Monthly_LossRatio'].mean(),
            'premium_growth': ((last_month['TotalPremium_sum'] / first_month['TotalPremium_sum']) - 1) * 100,
            'claims_growth': ((last_month['TotalClaims_sum'] / first_month['TotalClaims_sum']) - 1) * 100,
            'best_month': {
                'date': monthly_trends.loc[monthly_trends['Monthly_LossRatio'].idxmin(), 'TransactionMonth'],
                'loss_ratio': monthly_trends['Monthly_LossRatio'].min()
            },
            'worst_month': {
                'date': monthly_trends.loc[monthly_trends['Monthly_LossRatio'].idxmax(), 'TransactionMonth'],
                'loss_ratio': monthly_trends['Monthly_LossRatio'].max()
            },
            'profitable_months': len(monthly_trends[monthly_trends['Monthly_LossRatio'] < 1.0])
        }
        
        self.analysis_results['temporal_analysis'] = temporal_results
        return temporal_results
    
    def generate_comprehensive_analysis(self):
        """
        Generate a comprehensive statistical analysis.
        
        Returns:
            dict: Complete statistical analysis results
        """
        print("📊 Conducting comprehensive statistical analysis...")
        
        self.descriptive_statistics()
        self.normality_tests()
        self.correlation_analysis('pearson')
        self.correlation_analysis('spearman')
        self.bivariate_analysis()
        self.categorical_analysis()
        self.temporal_analysis()
        
        print("✅ Statistical analysis completed!")
        return self.analysis_results
    
    def print_analysis_summary(self):
        """Print a summary of statistical analysis findings."""
        if not self.analysis_results:
            self.generate_comprehensive_analysis()
        
        print("="*80)
        print("📊 STATISTICAL ANALYSIS SUMMARY")
        print("="*80)
        
        # Descriptive statistics summary
        if 'descriptive_statistics' in self.analysis_results:
            print(f"\n📈 DESCRIPTIVE STATISTICS:")
            for var, stats in self.analysis_results['descriptive_statistics'].items():
                print(f"   {var}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, CV={stats['coefficient_of_variation']:.3f}")
        
        # Correlation summary
        if 'pearson_correlation' in self.analysis_results:
            pearson_corrs = self.analysis_results['pearson_correlation']['significant_correlations']
            print(f"\n🔗 CORRELATIONS (Pearson):")
            print(f"   Found {len(pearson_corrs)} significant correlations (|r| > {CORRELATION_THRESHOLD})")
            for corr in pearson_corrs[:5]:  # Top 5
                print(f"   {corr['variable1']} - {corr['variable2']}: {corr['correlation']:.3f}")
        
        # Normality tests summary
        if 'normality_tests' in self.analysis_results:
            print(f"\n📋 NORMALITY TESTS:")
            for var, test in self.analysis_results['normality_tests'].items():
                jb_result = "Normal" if test['jarque_bera_normal'] else "Not Normal"
                print(f"   {var}: {jb_result} (JB p-value: {test['jarque_bera_p_value']:.4f})")
        
        # Temporal analysis summary
        if 'temporal_analysis' in self.analysis_results:
            temporal = self.analysis_results['temporal_analysis']
            print(f"\n⏰ TEMPORAL ANALYSIS:")
            print(f"   Analysis period: {temporal['period_start'].strftime('%B %Y')} to {temporal['period_end'].strftime('%B %Y')}")
            print(f"   Profitable months: {temporal['profitable_months']}/{temporal['total_months']}")
            print(f"   Average loss ratio: {temporal['avg_monthly_loss_ratio']:.4f}")

def analyze_statistics(df):
    """
    Convenience function to perform statistical analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        StatisticalAnalyzer: Analyzer instance with complete results
    """
    analyzer = StatisticalAnalyzer(df)
    analyzer.generate_comprehensive_analysis()
    return analyzer
