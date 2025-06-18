"""
Business Analysis module for Insurance Risk Analytics
Handles business-specific analyses, KPI calculations, and insights generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from .config import (BREAK_EVEN_LOSS_RATIO, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD, 
                    MIN_POLICY_COUNT)

warnings.filterwarnings('ignore')

class BusinessAnalyzer:
    """
    Business analysis class for insurance risk analytics.
    Provides business-focused analysis and KPI calculations.
    """
    
    def __init__(self, df):
        """
        Initialize the business analyzer.
        
        Args:
            df (pd.DataFrame): The dataset to analyze
        """
        self.df = df
        self.business_metrics = {}
    
    def calculate_overall_metrics(self):
        """
        Calculate overall portfolio metrics.
        
        Returns:
            dict: Overall business metrics
        """
        overall_premium = self.df['TotalPremium'].sum()
        overall_claims = self.df['TotalClaims'].sum()
        overall_loss_ratio = overall_claims / overall_premium if overall_premium > 0 else np.nan
        
        metrics = {
            'total_premium': overall_premium,
            'total_claims': overall_claims,
            'overall_loss_ratio': overall_loss_ratio,
            'total_policies': self.df['PolicyID'].nunique(),
            'avg_premium_per_policy': overall_premium / self.df['PolicyID'].nunique(),
            'avg_claims_per_policy': overall_claims / self.df['PolicyID'].nunique(),
            'claims_frequency': len(self.df[self.df['TotalClaims'] > 0]) / len(self.df),
            'is_profitable': overall_loss_ratio < BREAK_EVEN_LOSS_RATIO
        }
        
        self.business_metrics['overall'] = metrics
        return metrics
    
    def analyze_loss_ratio_by_dimension(self, dimension):
        """
        Analyze loss ratio by a specific dimension.
        
        Args:
            dimension (str): Column name to analyze by
            
        Returns:
            pd.DataFrame: Loss ratio analysis by dimension
        """
        if dimension not in self.df.columns:
            return pd.DataFrame()
        
        analysis = self.df.groupby(dimension).agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).reset_index()
        
        analysis['LossRatio'] = analysis['TotalClaims'] / analysis['TotalPremium']
        analysis['PremiumShare'] = (analysis['TotalPremium'] / analysis['TotalPremium'].sum()) * 100
        analysis['RiskCategory'] = pd.cut(
            analysis['LossRatio'],
            bins=[-np.inf, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, np.inf],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Sort by loss ratio
        analysis = analysis.sort_values('LossRatio', ascending=False)
        
        self.business_metrics[f'loss_ratio_by_{dimension}'] = analysis
        return analysis
    
    def analyze_geographical_performance(self):
        """
        Analyze performance by geographical regions.
        
        Returns:
            dict: Geographical performance analysis
        """
        if 'Province' not in self.df.columns:
            return {}
        
        province_analysis = self.analyze_loss_ratio_by_dimension('Province')
        
        # Additional geographical insights
        geographical_insights = {
            'province_analysis': province_analysis,
            'total_provinces': province_analysis.shape[0],
            'profitable_provinces': len(province_analysis[province_analysis['LossRatio'] < BREAK_EVEN_LOSS_RATIO]),
            'unprofitable_provinces': len(province_analysis[province_analysis['LossRatio'] >= BREAK_EVEN_LOSS_RATIO]),
            'top_5_by_premium': province_analysis.nlargest(5, 'TotalPremium')[['Province', 'TotalPremium', 'LossRatio']],
            'worst_5_by_loss_ratio': province_analysis.nlargest(5, 'LossRatio')[['Province', 'LossRatio', 'PremiumShare']],
            'best_5_by_loss_ratio': province_analysis.nsmallest(5, 'LossRatio')[['Province', 'LossRatio', 'PremiumShare']],
            'market_concentration': {
                'top_3_share': province_analysis.nlargest(3, 'PremiumShare')['PremiumShare'].sum(),
                'top_5_share': province_analysis.nlargest(5, 'PremiumShare')['PremiumShare'].sum()
            }
        }
        
        self.business_metrics['geographical'] = geographical_insights
        return geographical_insights
    
    def analyze_vehicle_risk_profile(self):
        """
        Analyze risk profiles by vehicle characteristics.
        
        Returns:
            dict: Vehicle risk analysis
        """
        vehicle_insights = {}
        
        # Analysis by Vehicle Type
        if 'VehicleType' in self.df.columns:
            vehicle_type_analysis = self.analyze_loss_ratio_by_dimension('VehicleType')
            vehicle_insights['by_vehicle_type'] = vehicle_type_analysis
        
        # Analysis by Vehicle Make
        if 'make' in self.df.columns:
            # Filter for statistical significance
            make_analysis = self.df.groupby('make').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'nunique'
            }).reset_index()
            
            make_analysis['LossRatio'] = make_analysis['TotalClaims'] / make_analysis['TotalPremium']
            make_analysis['AvgClaimAmount'] = make_analysis['TotalClaims'] / make_analysis['PolicyID']
            
            # Filter for statistical significance
            significant_makes = make_analysis[make_analysis['PolicyID'] >= MIN_POLICY_COUNT]
            
            vehicle_insights['by_make'] = {
                'all_makes': make_analysis,
                'significant_makes': significant_makes,
                'highest_claims': significant_makes.nlargest(5, 'AvgClaimAmount'),
                'lowest_claims': significant_makes.nsmallest(5, 'AvgClaimAmount'),
                'highest_risk': significant_makes.nlargest(5, 'LossRatio'),
                'lowest_risk': significant_makes.nsmallest(5, 'LossRatio')
            }
        
        self.business_metrics['vehicle_risk'] = vehicle_insights
        return vehicle_insights
    
    def analyze_temporal_performance(self):
        """
        Analyze temporal performance trends.
        
        Returns:
            dict: Temporal performance analysis
        """
        if 'TransactionMonth' not in self.df.columns:
            return {}
        
        # Monthly analysis
        monthly_trends = self.df.groupby('TransactionMonth').agg({
            'TotalPremium': ['sum', 'mean', 'count'],
            'TotalClaims': ['sum', 'mean'],
            'PolicyID': 'nunique'
        }).round(4)
        
        monthly_trends.columns = ['_'.join(col).strip() for col in monthly_trends.columns.values]
        monthly_trends = monthly_trends.reset_index()
        monthly_trends['Monthly_LossRatio'] = monthly_trends['TotalClaims_sum'] / monthly_trends['TotalPremium_sum']
        monthly_trends['ClaimFrequency'] = monthly_trends['TotalClaims_sum'] / monthly_trends['PolicyID_nunique']
        
        # Calculate trends
        first_month = monthly_trends.iloc[0]
        last_month = monthly_trends.iloc[-1]
        
        # Seasonal analysis
        self.df['Month_Name'] = self.df['TransactionMonth'].dt.month_name()
        seasonal_lr = self.df.groupby('Month_Name').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).reset_index()
        seasonal_lr['LossRatio'] = seasonal_lr['TotalClaims'] / seasonal_lr['TotalPremium']
        
        temporal_insights = {
            'monthly_data': monthly_trends,
            'period_start': monthly_trends['TransactionMonth'].min(),
            'period_end': monthly_trends['TransactionMonth'].max(),
            'total_months': len(monthly_trends),
            'profitable_months': len(monthly_trends[monthly_trends['Monthly_LossRatio'] < BREAK_EVEN_LOSS_RATIO]),
            'avg_monthly_loss_ratio': monthly_trends['Monthly_LossRatio'].mean(),
            'trends': {
                'premium_growth': ((last_month['TotalPremium_sum'] / first_month['TotalPremium_sum']) - 1) * 100,
                'claims_growth': ((last_month['TotalClaims_sum'] / first_month['TotalClaims_sum']) - 1) * 100,
                'loss_ratio_change': last_month['Monthly_LossRatio'] - first_month['Monthly_LossRatio'],
                'claim_frequency_change': ((last_month['ClaimFrequency'] - first_month['ClaimFrequency']) / first_month['ClaimFrequency']) * 100
            },
            'extremes': {
                'best_month': {
                    'date': monthly_trends.loc[monthly_trends['Monthly_LossRatio'].idxmin(), 'TransactionMonth'],
                    'loss_ratio': monthly_trends['Monthly_LossRatio'].min()
                },
                'worst_month': {
                    'date': monthly_trends.loc[monthly_trends['Monthly_LossRatio'].idxmax(), 'TransactionMonth'],
                    'loss_ratio': monthly_trends['Monthly_LossRatio'].max()
                }
            },
            'seasonal_patterns': seasonal_lr.sort_values('LossRatio', ascending=False)
        }
        
        self.business_metrics['temporal'] = temporal_insights
        return temporal_insights
    
    def analyze_customer_segments(self):
        """
        Analyze performance by customer segments.
        
        Returns:
            dict: Customer segment analysis
        """
        customer_insights = {}
        
        # Gender analysis
        if 'Gender' in self.df.columns:
            gender_analysis = self.analyze_loss_ratio_by_dimension('Gender')
            customer_insights['by_gender'] = gender_analysis
        
        # Age analysis (if age data available through vehicle registration year)
        if 'RegistrationYear' in self.df.columns:
            current_year = datetime.now().year
            self.df['VehicleAge'] = current_year - self.df['RegistrationYear']
            
            # Create age categories
            self.df['VehicleAgeCategory'] = pd.cut(
                self.df['VehicleAge'],
                bins=[0, 5, 10, 15, np.inf],
                labels=['New (0-5)', 'Recent (6-10)', 'Older (11-15)', 'Very Old (15+)']
            )
            
            age_analysis = self.analyze_loss_ratio_by_dimension('VehicleAgeCategory')
            customer_insights['by_vehicle_age'] = age_analysis
        
        self.business_metrics['customer_segments'] = customer_insights
        return customer_insights
    
    def generate_business_recommendations(self):
        """
        Generate actionable business recommendations.
        
        Returns:
            dict: Business recommendations
        """
        recommendations = {
            'immediate_actions': [],
            'pricing_strategies': [],
            'risk_management': [],
            'portfolio_optimization': [],
            'data_improvements': []
        }
        
        # Check if analysis has been run
        if 'overall' not in self.business_metrics:
            self.calculate_overall_metrics()
        
        overall = self.business_metrics['overall']
        
        # Overall profitability recommendations
        if not overall['is_profitable']:
            recommendations['immediate_actions'].append({
                'priority': 'HIGH',
                'action': 'Portfolio Review',
                'description': f"Overall loss ratio of {overall['overall_loss_ratio']:.3f} indicates unprofitable portfolio",
                'impact': 'Critical - affects entire business viability'
            })
        
        # Geographical recommendations
        if 'geographical' in self.business_metrics:
            geo = self.business_metrics['geographical']
            
            if geo['unprofitable_provinces'] > 0:
                recommendations['risk_management'].append({
                    'priority': 'HIGH',
                    'action': 'Geographical Risk Assessment',
                    'description': f"{geo['unprofitable_provinces']} provinces showing losses",
                    'impact': 'Reduce exposure in high-risk areas'
                })
            
            # Market concentration
            if geo['market_concentration']['top_3_share'] > 70:
                recommendations['portfolio_optimization'].append({
                    'priority': 'MEDIUM',
                    'action': 'Market Diversification',
                    'description': f"Top 3 provinces control {geo['market_concentration']['top_3_share']:.1f}% of premium",
                    'impact': 'Reduce concentration risk'
                })
        
        # Temporal recommendations
        if 'temporal' in self.business_metrics:
            temporal = self.business_metrics['temporal']
            
            if temporal['profitable_months'] < temporal['total_months'] * 0.6:
                recommendations['pricing_strategies'].append({
                    'priority': 'HIGH',
                    'action': 'Seasonal Pricing Review',
                    'description': f"Only {temporal['profitable_months']}/{temporal['total_months']} months profitable",
                    'impact': 'Implement seasonal risk adjustments'
                })
        
        # Vehicle risk recommendations
        if 'vehicle_risk' in self.business_metrics:
            vehicle = self.business_metrics['vehicle_risk']
            
            if 'by_make' in vehicle and 'highest_risk' in vehicle['by_make']:
                recommendations['pricing_strategies'].append({
                    'priority': 'MEDIUM',
                    'action': 'Vehicle-Specific Pricing',
                    'description': 'Significant variation in loss ratios by vehicle make',
                    'impact': 'Implement make/model risk factors'
                })
        
        # Data quality recommendations
        recommendations['data_improvements'].append({
            'priority': 'MEDIUM',
            'action': 'Enhanced Data Collection',
            'description': 'Improve data quality for missing values and outliers',
            'impact': 'Better risk assessment and pricing accuracy'
        })
        
        self.business_metrics['recommendations'] = recommendations
        return recommendations
    
    def generate_executive_summary(self):
        """
        Generate executive summary of key findings.
        
        Returns:
            dict: Executive summary
        """
        if 'overall' not in self.business_metrics:
            self.calculate_overall_metrics()
        
        overall = self.business_metrics['overall']
        
        summary = {
            'portfolio_status': 'PROFITABLE' if overall['is_profitable'] else 'UNPROFITABLE',
            'overall_loss_ratio': overall['overall_loss_ratio'],
            'total_premium': overall['total_premium'],
            'total_claims': overall['total_claims'],
            'policy_count': overall['total_policies'],
            'key_metrics': {
                'claims_frequency': overall['claims_frequency'],
                'avg_premium_per_policy': overall['avg_premium_per_policy'],
                'avg_claims_per_policy': overall['avg_claims_per_policy']
            }
        }
        
        # Add geographical insights if available
        if 'geographical' in self.business_metrics:
            geo = self.business_metrics['geographical']
            summary['geographical_insights'] = {
                'profitable_provinces': geo['profitable_provinces'],
                'total_provinces': geo['total_provinces'],
                'market_concentration': geo['market_concentration']['top_3_share']
            }
        
        # Add temporal insights if available
        if 'temporal' in self.business_metrics:
            temporal = self.business_metrics['temporal']
            summary['temporal_insights'] = {
                'profitable_months': temporal['profitable_months'],
                'total_months': temporal['total_months'],
                'avg_monthly_loss_ratio': temporal['avg_monthly_loss_ratio']
            }
        
        self.business_metrics['executive_summary'] = summary
        return summary
    
    def generate_comprehensive_analysis(self):
        """
        Generate comprehensive business analysis.
        
        Returns:
            dict: Complete business analysis results
        """
        print("💼 Conducting comprehensive business analysis...")
        
        self.calculate_overall_metrics()
        self.analyze_geographical_performance()
        self.analyze_vehicle_risk_profile()
        self.analyze_temporal_performance()
        self.analyze_customer_segments()
        self.generate_business_recommendations()
        self.generate_executive_summary()
        
        print("✅ Business analysis completed!")
        return self.business_metrics
    
    def print_business_summary(self):
        """Print a summary of business analysis findings."""
        if 'executive_summary' not in self.business_metrics:
            self.generate_comprehensive_analysis()
        
        summary = self.business_metrics['executive_summary']
        
        print("="*80)
        print("💼 BUSINESS ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 PORTFOLIO OVERVIEW:")
        print(f"   Status: {summary['portfolio_status']}")
        print(f"   Overall Loss Ratio: {summary['overall_loss_ratio']:.4f}")
        print(f"   Total Premium: ${summary['total_premium']:,.2f}")
        print(f"   Total Claims: ${summary['total_claims']:,.2f}")
        print(f"   Policy Count: {summary['policy_count']:,}")
        
        print(f"\n🎯 KEY METRICS:")
        metrics = summary['key_metrics']
        print(f"   Claims Frequency: {metrics['claims_frequency']:.1%}")
        print(f"   Avg Premium per Policy: ${metrics['avg_premium_per_policy']:,.2f}")
        print(f"   Avg Claims per Policy: ${metrics['avg_claims_per_policy']:,.2f}")
        
        if 'geographical_insights' in summary:
            geo = summary['geographical_insights']
            print(f"\n🌍 GEOGRAPHICAL INSIGHTS:")
            print(f"   Profitable Provinces: {geo['profitable_provinces']}/{geo['total_provinces']}")
            print(f"   Market Concentration (Top 3): {geo['market_concentration']:.1f}%")
        
        if 'temporal_insights' in summary:
            temporal = summary['temporal_insights']
            print(f"\n📅 TEMPORAL INSIGHTS:")
            print(f"   Profitable Months: {temporal['profitable_months']}/{temporal['total_months']}")
            print(f"   Avg Monthly Loss Ratio: {temporal['avg_monthly_loss_ratio']:.4f}")

def analyze_business_performance(df):
    """
    Convenience function to perform business analysis.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
        
    Returns:
        BusinessAnalyzer: Analyzer instance with complete results
    """
    analyzer = BusinessAnalyzer(df)
    analyzer.generate_comprehensive_analysis()
    return analyzer
