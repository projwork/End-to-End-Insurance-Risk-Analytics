"""
Hypothesis Testing Module for Insurance Risk Analytics
Provides comprehensive A/B testing and statistical validation capabilities
for risk drivers and business metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, levene
import warnings
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns

from .config import STATISTICAL_SIGNIFICANCE, RANDOM_STATE
from .utils import print_section_header

warnings.filterwarnings('ignore')

class HypothesisTestingFramework:
    """
    Comprehensive framework for A/B hypothesis testing in insurance risk analytics.
    
    Supports testing hypotheses about:
    - Risk differences across provinces
    - Risk differences between zip codes
    - Margin differences between zip codes
    - Risk differences between genders
    """
    
    def __init__(self, df: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize the hypothesis testing framework.
        
        Args:
            df (pd.DataFrame): Insurance dataset
            significance_level (float): Statistical significance threshold
        """
        self.df = df.copy()
        self.significance_level = significance_level
        self.results = {}
        
        # Calculate key metrics
        self._calculate_risk_metrics()
        
    def _calculate_risk_metrics(self):
        """Calculate risk and profitability metrics."""
        print("📊 Calculating risk and profitability metrics...")
        
        # Claim Frequency: Proportion of policies with at least one claim
        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
        
        # Claim Severity: Average claim amount given a claim occurred
        # For policies with no claims, severity is NaN (will be handled in analysis)
        self.df['ClaimSeverity'] = np.where(
            self.df['TotalClaims'] > 0, 
            self.df['TotalClaims'], 
            np.nan
        )
        
        # Margin: Profit/Loss per policy
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        
        # Loss Ratio (for additional analysis)
        self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
        
        print(f"✅ Metrics calculated:")
        print(f"   - Claim Frequency: {self.df['HasClaim'].mean():.3f}")
        print(f"   - Avg Claim Severity: ${self.df['ClaimSeverity'].mean():,.2f}")
        print(f"   - Avg Margin: ${self.df['Margin'].mean():,.2f}")
        
    def test_provincial_risk_differences(self) -> Dict:
        """
        Test H₀: There are no risk differences across provinces
        
        Returns:
            Dict: Test results for provincial risk differences
        """
        print_section_header("🌍 Testing Provincial Risk Differences")
        
        results = {
            'hypothesis': 'H₀: There are no risk differences across provinces',
            'tests': {}
        }
        
        # Test 1: Claim Frequency across provinces (Chi-squared test)
        print("📋 Test 1: Claim Frequency across Provinces")
        contingency_table = pd.crosstab(self.df['Province'], self.df['HasClaim'])
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        results['tests']['claim_frequency'] = {
            'test_type': 'Chi-squared test',
            'statistic': chi2,
            'p_value': p_value_freq,
            'degrees_of_freedom': dof,
            'significant': p_value_freq < self.significance_level,
            'interpretation': self._interpret_result(p_value_freq, 'claim frequency across provinces')
        }
        
        print(f"   Chi-squared statistic: {chi2:.4f}")
        print(f"   p-value: {p_value_freq:.6f}")
        print(f"   Result: {'Reject H₀' if p_value_freq < self.significance_level else 'Fail to reject H₀'}")
        
        return results
    
    def test_zipcode_risk_differences(self, min_policies: int = 100) -> Dict:
        """
        Test H₀: There are no risk differences between zip codes
        
        Args:
            min_policies (int): Minimum policies per zip code for statistical significance
            
        Returns:
            Dict: Test results for zip code risk differences
        """
        print_section_header("📮 Testing Zip Code Risk Differences")
        
        results = {
            'hypothesis': 'H₀: There are no risk differences between zip codes',
            'tests': {},
            'min_policies_threshold': min_policies
        }
        
        # Filter zip codes with sufficient data
        zipcode_counts = self.df['PostalCode'].value_counts()
        valid_zipcodes = zipcode_counts[zipcode_counts >= min_policies].index
        filtered_df = self.df[self.df['PostalCode'].isin(valid_zipcodes)]
        
        print(f"📋 Analyzing {len(valid_zipcodes)} zip codes with ≥{min_policies} policies")
        print(f"   Total policies in analysis: {len(filtered_df):,}")
        
        if len(valid_zipcodes) < 2:
            print("⚠️ Insufficient zip codes for comparison")
            return results
        
        # Test 1: Claim Frequency variance across zip codes
        print("\n📋 Test 1: Claim Frequency Variance Across Zip Codes")
        zipcode_freq = filtered_df.groupby('PostalCode')['HasClaim'].mean()
        
        # Use F-test for variance in means
        overall_freq = filtered_df['HasClaim'].mean()
        zipcode_variance = zipcode_freq.var()
        
        # Chi-squared test for independence
        if len(valid_zipcodes) >= 2:
            # Sample top zip codes for pairwise comparison
            top_zipcodes = zipcode_counts.head(10).index
            comparison_df = self.df[self.df['PostalCode'].isin(top_zipcodes)]
            
            if len(comparison_df) > 0:
                contingency_table = pd.crosstab(comparison_df['PostalCode'], comparison_df['HasClaim'])
                chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
                
                results['tests']['claim_frequency'] = {
                    'test_type': 'Chi-squared test (top zip codes)',
                    'statistic': chi2,
                    'p_value': p_value_freq,
                    'degrees_of_freedom': dof,
                    'zip_codes_tested': len(top_zipcodes),
                    'significant': p_value_freq < self.significance_level,
                    'interpretation': self._interpret_result(p_value_freq, 'claim frequency across zip codes')
                }
                
                print(f"   Chi-squared statistic: {chi2:.4f}")
                print(f"   p-value: {p_value_freq:.6f}")
                print(f"   Zip codes tested: {len(top_zipcodes)}")
                print(f"   Result: {'Reject H₀' if p_value_freq < self.significance_level else 'Fail to reject H₀'}")
        
        # Summary statistics by zip code
        zipcode_stats = filtered_df.groupby('PostalCode').agg({
            'HasClaim': 'mean',
            'ClaimSeverity': 'mean',
            'LossRatio': 'mean',
            'TotalPremium': 'count'
        }).round(4)
        zipcode_stats.columns = ['Claim_Frequency', 'Avg_Claim_Severity', 'Avg_Loss_Ratio', 'Policy_Count']
        
        # Show top 10 highest and lowest risk zip codes
        top_risk = zipcode_stats.nlargest(5, 'Claim_Frequency')
        low_risk = zipcode_stats.nsmallest(5, 'Claim_Frequency')
        
        results['top_risk_zipcodes'] = top_risk
        results['low_risk_zipcodes'] = low_risk
        
        print(f"\n📊 Top 5 Highest Risk Zip Codes:")
        print(top_risk)
        print(f"\n📊 Top 5 Lowest Risk Zip Codes:")
        print(low_risk)
        
        self.results['zipcode_risk'] = results
        return results
    
    def test_zipcode_margin_differences(self, min_policies: int = 100) -> Dict:
        """
        Test H₀: There are no significant margin (profit) differences between zip codes
        
        Args:
            min_policies (int): Minimum policies per zip code for statistical significance
            
        Returns:
            Dict: Test results for zip code margin differences
        """
        print_section_header("💰 Testing Zip Code Margin Differences")
        
        results = {
            'hypothesis': 'H₀: There are no significant margin differences between zip codes',
            'tests': {},
            'min_policies_threshold': min_policies
        }
        
        # Filter zip codes with sufficient data
        zipcode_counts = self.df['PostalCode'].value_counts()
        valid_zipcodes = zipcode_counts[zipcode_counts >= min_policies].index
        filtered_df = self.df[self.df['PostalCode'].isin(valid_zipcodes)]
        
        print(f"📋 Analyzing {len(valid_zipcodes)} zip codes with ≥{min_policies} policies")
        
        if len(valid_zipcodes) < 2:
            print("⚠️ Insufficient zip codes for comparison")
            return results
        
        # Test 1: ANOVA for margin differences across zip codes
        print("\n📋 Test 1: Margin Differences Across Zip Codes (ANOVA)")
        
        zipcode_margins = []
        zipcode_names = []
        
        # Get top 10 zip codes by policy count for robust comparison
        top_zipcodes = zipcode_counts.head(10).index
        
        for zipcode in top_zipcodes:
            zipcode_data = filtered_df[filtered_df['PostalCode'] == zipcode]['Margin']
            if len(zipcode_data) >= min_policies:
                zipcode_margins.append(zipcode_data)
                zipcode_names.append(zipcode)
        
        if len(zipcode_margins) >= 2:
            f_stat, p_value_margin = stats.f_oneway(*zipcode_margins)
            
            results['tests']['margin_anova'] = {
                'test_type': 'ANOVA F-test',
                'statistic': f_stat,
                'p_value': p_value_margin,
                'zip_codes_tested': len(zipcode_names),
                'significant': p_value_margin < self.significance_level,
                'interpretation': self._interpret_result(p_value_margin, 'margin differences across zip codes')
            }
            
            print(f"   F-statistic: {f_stat:.4f}")
            print(f"   p-value: {p_value_margin:.6f}")
            print(f"   Zip codes tested: {len(zipcode_names)}")
            print(f"   Result: {'Reject H₀' if p_value_margin < self.significance_level else 'Fail to reject H₀'}")
        
        # Test 2: Pairwise comparison of most and least profitable zip codes
        zipcode_margins_summary = filtered_df.groupby('PostalCode')['Margin'].agg(['mean', 'count', 'std']).round(2)
        zipcode_margins_summary = zipcode_margins_summary[zipcode_margins_summary['count'] >= min_policies]
        
        if len(zipcode_margins_summary) >= 2:
            most_profitable = zipcode_margins_summary.loc[zipcode_margins_summary['mean'].idxmax()]
            least_profitable = zipcode_margins_summary.loc[zipcode_margins_summary['mean'].idxmin()]
            
            # T-test between most and least profitable
            most_data = filtered_df[filtered_df['PostalCode'] == zipcode_margins_summary['mean'].idxmax()]['Margin']
            least_data = filtered_df[filtered_df['PostalCode'] == zipcode_margins_summary['mean'].idxmin()]['Margin']
            
            t_stat, p_value_ttest = ttest_ind(most_data, least_data)
            
            results['tests']['extreme_comparison'] = {
                'test_type': 'Two-sample t-test',
                'most_profitable_zipcode': zipcode_margins_summary['mean'].idxmax(),
                'least_profitable_zipcode': zipcode_margins_summary['mean'].idxmin(),
                'statistic': t_stat,
                'p_value': p_value_ttest,
                'significant': p_value_ttest < self.significance_level,
                'margin_difference': most_profitable['mean'] - least_profitable['mean'],
                'interpretation': self._interpret_result(p_value_ttest, 'margin difference between extreme zip codes')
            }
            
            print(f"\n📋 Test 2: Extreme Zip Code Comparison")
            print(f"   Most profitable: {zipcode_margins_summary['mean'].idxmax()} (${most_profitable['mean']:.2f})")
            print(f"   Least profitable: {zipcode_margins_summary['mean'].idxmin()} (${least_profitable['mean']:.2f})")
            print(f"   Margin difference: ${most_profitable['mean'] - least_profitable['mean']:.2f}")
            print(f"   t-statistic: {t_stat:.4f}")
            print(f"   p-value: {p_value_ttest:.6f}")
            print(f"   Result: {'Reject H₀' if p_value_ttest < self.significance_level else 'Fail to reject H₀'}")
        
        # Summary statistics
        top_profitable = zipcode_margins_summary.nlargest(5, 'mean')
        low_profitable = zipcode_margins_summary.nsmallest(5, 'mean')
        
        results['top_profitable_zipcodes'] = top_profitable
        results['low_profitable_zipcodes'] = low_profitable
        
        print(f"\n📊 Top 5 Most Profitable Zip Codes:")
        print(top_profitable)
        print(f"\n📊 Top 5 Least Profitable Zip Codes:")
        print(low_profitable)
        
        self.results['zipcode_margin'] = results
        return results
    
    def test_gender_risk_differences(self) -> Dict:
        """
        Test H₀: There are no significant risk differences between Women and Men
        
        Returns:
            Dict: Test results for gender risk differences
        """
        print_section_header("👥 Testing Gender Risk Differences")
        
        results = {
            'hypothesis': 'H₀: There are no significant risk differences between Women and Men',
            'tests': {}
        }
        
        # Filter for Male and Female only (exclude other categories)
        gender_df = self.df[self.df['Gender'].isin(['Male', 'Female'])].copy()
        
        if len(gender_df) == 0:
            print("⚠️ No gender data available for analysis")
            return results
        
        print(f"📋 Analyzing {len(gender_df):,} policies with gender data")
        print(f"   Male policies: {(gender_df['Gender'] == 'Male').sum():,}")
        print(f"   Female policies: {(gender_df['Gender'] == 'Female').sum():,}")
        
        # Test 1: Claim Frequency difference (Chi-squared test)
        print("\n📋 Test 1: Claim Frequency by Gender")
        contingency_table = pd.crosstab(gender_df['Gender'], gender_df['HasClaim'])
        chi2, p_value_freq, dof, expected = chi2_contingency(contingency_table)
        
        results['tests']['claim_frequency'] = {
            'test_type': 'Chi-squared test',
            'statistic': chi2,
            'p_value': p_value_freq,
            'degrees_of_freedom': dof,
            'significant': p_value_freq < self.significance_level,
            'interpretation': self._interpret_result(p_value_freq, 'claim frequency between genders')
        }
        
        print(f"   Chi-squared statistic: {chi2:.4f}")
        print(f"   p-value: {p_value_freq:.6f}")
        print(f"   Result: {'Reject H₀' if p_value_freq < self.significance_level else 'Fail to reject H₀'}")
        
        # Test 2: Claim Severity difference (t-test)
        print("\n📋 Test 2: Claim Severity by Gender")
        male_severity = gender_df[(gender_df['Gender'] == 'Male') & (gender_df['ClaimSeverity'].notna())]['ClaimSeverity']
        female_severity = gender_df[(gender_df['Gender'] == 'Female') & (gender_df['ClaimSeverity'].notna())]['ClaimSeverity']
        
        if len(male_severity) > 0 and len(female_severity) > 0:
            # Check for equal variances
            levene_stat, levene_p = levene(male_severity, female_severity)
            equal_var = levene_p > 0.05
            
            t_stat, p_value_sev = ttest_ind(male_severity, female_severity, equal_var=equal_var)
            
            results['tests']['claim_severity'] = {
                'test_type': f'Two-sample t-test (equal_var={equal_var})',
                'statistic': t_stat,
                'p_value': p_value_sev,
                'equal_variances': equal_var,
                'levene_test_p': levene_p,
                'significant': p_value_sev < self.significance_level,
                'interpretation': self._interpret_result(p_value_sev, 'claim severity between genders')
            }
            
            print(f"   t-statistic: {t_stat:.4f}")
            print(f"   p-value: {p_value_sev:.6f}")
            print(f"   Equal variances: {equal_var}")
            print(f"   Result: {'Reject H₀' if p_value_sev < self.significance_level else 'Fail to reject H₀'}")
        
        # Test 3: Overall Loss Ratio difference (t-test)
        print("\n📋 Test 3: Loss Ratio by Gender")
        male_loss_ratio = gender_df[gender_df['Gender'] == 'Male']['LossRatio']
        female_loss_ratio = gender_df[gender_df['Gender'] == 'Female']['LossRatio']
        
        # Check for equal variances
        levene_stat, levene_p = levene(male_loss_ratio, female_loss_ratio)
        equal_var = levene_p > 0.05
        
        t_stat, p_value_loss = ttest_ind(male_loss_ratio, female_loss_ratio, equal_var=equal_var)
        
        results['tests']['loss_ratio'] = {
            'test_type': f'Two-sample t-test (equal_var={equal_var})',
            'statistic': t_stat,
            'p_value': p_value_loss,
            'equal_variances': equal_var,
            'significant': p_value_loss < self.significance_level,
            'interpretation': self._interpret_result(p_value_loss, 'loss ratio between genders')
        }
        
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_value_loss:.6f}")
        print(f"   Equal variances: {equal_var}")
        print(f"   Result: {'Reject H₀' if p_value_loss < self.significance_level else 'Fail to reject H₀'}")
        
        # Summary statistics by gender
        gender_stats = gender_df.groupby('Gender').agg({
            'HasClaim': 'mean',
            'ClaimSeverity': 'mean',
            'LossRatio': 'mean',
            'Margin': 'mean',
            'TotalPremium': ['count', 'mean'],
            'TotalClaims': 'mean'
        }).round(4)
        
        results['summary_stats'] = gender_stats
        
        print(f"\n📊 Gender Risk Summary:")
        print(gender_stats)
        
        self.results['gender_risk'] = results
        return results
    
    def _interpret_result(self, p_value: float, test_description: str) -> str:
        """
        Interpret statistical test results in business terms.
        
        Args:
            p_value (float): p-value from statistical test
            test_description (str): Description of what was tested
            
        Returns:
            str: Business interpretation of the result
        """
        if p_value < self.significance_level:
            if p_value < 0.001:
                significance = "highly significant"
            elif p_value < 0.01:
                significance = "very significant"
            else:
                significance = "significant"
            
            return f"We reject the null hypothesis (p={p_value:.6f}). " \
                   f"There is {significance} evidence of differences in {test_description}. " \
                   f"This suggests that segmentation based on this factor may be warranted."
        else:
            return f"We fail to reject the null hypothesis (p={p_value:.6f}). " \
                   f"There is insufficient evidence of significant differences in {test_description}. " \
                   f"This factor may not be a strong segmentation criterion."
    
    def run_all_tests(self) -> Dict:
        """
        Run all hypothesis tests and return comprehensive results.
        
        Returns:
            Dict: Complete test results for all hypotheses
        """
        print_section_header("🧪 Running Complete Hypothesis Testing Suite")
        
        # Run all hypothesis tests
        provincial_results = self.test_provincial_risk_differences()
        zipcode_risk_results = self.test_zipcode_risk_differences()
        zipcode_margin_results = self.test_zipcode_margin_differences()
        gender_results = self.test_gender_risk_differences()
        
        # Compile summary
        summary = {
            'test_summary': {
                'provincial_risk': self._summarize_test_outcome(provincial_results),
                'zipcode_risk': self._summarize_test_outcome(zipcode_risk_results),
                'zipcode_margin': self._summarize_test_outcome(zipcode_margin_results),
                'gender_risk': self._summarize_test_outcome(gender_results)
            },
            'detailed_results': {
                'provincial_risk': provincial_results,
                'zipcode_risk': zipcode_risk_results,
                'zipcode_margin': zipcode_margin_results,
                'gender_risk': gender_results
            },
            'business_recommendations': self._generate_business_recommendations()
        }
        
        self.results['complete_analysis'] = summary
        return summary
    
    def _summarize_test_outcome(self, test_results: Dict) -> Dict:
        """Summarize the outcome of a hypothesis test."""
        if 'tests' not in test_results:
            return {'status': 'No tests conducted', 'significant_tests': 0, 'total_tests': 0}
        
        significant_tests = sum(1 for test in test_results['tests'].values() 
                              if test.get('significant', False))
        total_tests = len(test_results['tests'])
        
        return {
            'hypothesis': test_results.get('hypothesis', 'Unknown'),
            'significant_tests': significant_tests,
            'total_tests': total_tests,
            'overall_significance': significant_tests > 0,
            'recommendation': 'Reject H₀' if significant_tests > 0 else 'Fail to reject H₀'
        }
    
    def _generate_business_recommendations(self) -> List[str]:
        """Generate business recommendations based on test results."""
        recommendations = []
        
        # Analyze each test category
        for category, results in self.results.items():
            if category == 'complete_analysis':
                continue
                
            if 'tests' in results:
                for test_name, test_result in results['tests'].items():
                    if test_result.get('significant', False):
                        recommendations.append(
                            f"**{category.replace('_', ' ').title()}**: {test_result['interpretation']}"
                        )
        
        if not recommendations:
            recommendations.append(
                "No statistically significant differences were found in any of the tested factors. "
                "Current pricing may not require segmentation adjustments based on these variables."
            )
        
        return recommendations
    
    def print_executive_summary(self):
        """Print an executive summary of all hypothesis test results."""
        if 'complete_analysis' not in self.results:
            print("❌ No complete analysis available. Run run_all_tests() first.")
            return
        
        print_section_header("📋 EXECUTIVE SUMMARY - HYPOTHESIS TESTING RESULTS")
        
        summary = self.results['complete_analysis']['test_summary']
        recommendations = self.results['complete_analysis']['business_recommendations']
        
        print("🎯 **HYPOTHESIS TEST OUTCOMES:**\n")
        
        for test_name, outcome in summary.items():
            print(f"**{test_name.replace('_', ' ').upper()}**")
            print(f"   Hypothesis: {outcome['hypothesis']}")
            print(f"   Recommendation: {outcome['recommendation']}")
            print(f"   Significant tests: {outcome['significant_tests']}/{outcome['total_tests']}")
            print(f"   Overall: {'STATISTICALLY SIGNIFICANT' if outcome['overall_significance'] else 'NOT SIGNIFICANT'}")
            print()
        
        print("🚀 **BUSINESS RECOMMENDATIONS:**\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}\n")


def run_hypothesis_testing(df: pd.DataFrame, significance_level: float = 0.05) -> HypothesisTestingFramework:
    """
    Convenience function to run complete hypothesis testing analysis.
    
    Args:
        df (pd.DataFrame): Insurance dataset
        significance_level (float): Statistical significance threshold
        
    Returns:
        HypothesisTestingFramework: Configured testing framework with results
    """
    framework = HypothesisTestingFramework(df, significance_level)
    framework.run_all_tests()
    framework.print_executive_summary()
    return framework 