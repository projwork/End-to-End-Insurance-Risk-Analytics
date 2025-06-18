"""
Visualization module for Insurance Risk Analytics
Handles all plotting and visualization needs for the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import warnings

from .config import (FIGURE_SIZE_LARGE, FIGURE_SIZE_MEDIUM, FIGURE_SIZE_SMALL, 
                    PLOT_STYLE, COLORS, BREAK_EVEN_LOSS_RATIO)

warnings.filterwarnings('ignore')

class InsuranceVisualizer:
    """
    Visualization class for insurance risk analytics.
    Provides comprehensive plotting capabilities.
    """
    
    def __init__(self, df):
        """
        Initialize the visualizer.
        
        Args:
            df (pd.DataFrame): The dataset to visualize
        """
        self.df = df
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Set up plotting style and parameters."""
        plt.style.use(PLOT_STYLE)
        sns.set_palette("husl")
        pio.templates.default = "plotly_white"
    
    def plot_financial_distributions(self, save_path=None):
        """
        Plot distribution of key financial variables.
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_MEDIUM)
        fig.suptitle('Distribution of Key Financial Variables', fontsize=16, fontweight='bold')
        
        # TotalPremium distribution
        axes[0, 0].hist(self.df['TotalPremium'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Total Premium')
        axes[0, 0].set_xlabel('Total Premium')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(self.df['TotalPremium'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["TotalPremium"].mean():.2f}')
        axes[0, 0].axvline(self.df['TotalPremium'].median(), color='green', linestyle='--', 
                          label=f'Median: {self.df["TotalPremium"].median():.2f}')
        axes[0, 0].legend()
        
        # TotalClaims distribution
        axes[0, 1].hist(self.df['TotalClaims'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Distribution of Total Claims')
        axes[0, 1].set_xlabel('Total Claims')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(self.df['TotalClaims'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {self.df["TotalClaims"].mean():.2f}')
        axes[0, 1].axvline(self.df['TotalClaims'].median(), color='green', linestyle='--', 
                          label=f'Median: {self.df["TotalClaims"].median():.2f}')
        axes[0, 1].legend()
        
        # LossRatio distribution (filtered)
        if 'LossRatio' in self.df.columns:
            loss_ratio_filtered = self.df['LossRatio'][(self.df['LossRatio'] >= 0) & (self.df['LossRatio'] <= 5)]
            axes[1, 0].hist(loss_ratio_filtered, bins=50, alpha=0.7, color='gold', edgecolor='black')
            axes[1, 0].set_title('Distribution of Loss Ratio (0-5 range)')
            axes[1, 0].set_xlabel('Loss Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(loss_ratio_filtered.mean(), color='red', linestyle='--', 
                              label=f'Mean: {loss_ratio_filtered.mean():.4f}')
            axes[1, 0].axvline(loss_ratio_filtered.median(), color='green', linestyle='--', 
                              label=f'Median: {loss_ratio_filtered.median():.4f}')
            axes[1, 0].axvline(BREAK_EVEN_LOSS_RATIO, color='orange', linestyle='-', linewidth=2, 
                              label='Break-even (1.0)')
            axes[1, 0].legend()
        
        # CustomValueEstimate distribution (log scale)
        if 'CustomValueEstimate' in self.df.columns:
            cve_positive = self.df['CustomValueEstimate'][self.df['CustomValueEstimate'] > 0]
            if len(cve_positive) > 0:
                axes[1, 1].hist(np.log10(cve_positive), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1, 1].set_title('Distribution of Custom Value Estimate (Log10 Scale)')
                axes[1, 1].set_xlabel('Log10(Custom Value Estimate)')
                axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_categorical_distributions(self, save_path=None):
        """
        Plot distribution of key categorical variables.
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Distribution of Key Categorical Variables', fontsize=16, fontweight='bold')
        
        # Province distribution
        if 'Province' in self.df.columns:
            province_counts = self.df['Province'].value_counts().head(10)
            axes[0, 0].bar(range(len(province_counts)), province_counts.values, color='lightblue', edgecolor='black')
            axes[0, 0].set_title('Top 10 Provinces by Policy Count')
            axes[0, 0].set_xlabel('Province')
            axes[0, 0].set_ylabel('Number of Policies')
            axes[0, 0].set_xticks(range(len(province_counts)))
            axes[0, 0].set_xticklabels(province_counts.index, rotation=45, ha='right')
        
        # Gender distribution
        if 'Gender' in self.df.columns:
            gender_counts = self.df['Gender'].value_counts()
            axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Gender Distribution')
        
        # Vehicle Type distribution
        if 'VehicleType' in self.df.columns:
            vehicle_counts = self.df['VehicleType'].value_counts()
            axes[1, 0].bar(range(len(vehicle_counts)), vehicle_counts.values, color='lightgreen', edgecolor='black')
            axes[1, 0].set_title('Vehicle Type Distribution')
            axes[1, 0].set_xlabel('Vehicle Type')
            axes[1, 0].set_ylabel('Number of Policies')
            axes[1, 0].set_xticks(range(len(vehicle_counts)))
            axes[1, 0].set_xticklabels(vehicle_counts.index, rotation=45, ha='right')
        
        # Top vehicle makes
        if 'make' in self.df.columns:
            make_counts = self.df['make'].value_counts().head(10)
            axes[1, 1].barh(range(len(make_counts)), make_counts.values, color='gold', edgecolor='black')
            axes[1, 1].set_title('Top 10 Vehicle Makes')
            axes[1, 1].set_xlabel('Number of Policies')
            axes[1, 1].set_ylabel('Vehicle Make')
            axes[1, 1].set_yticks(range(len(make_counts)))
            axes[1, 1].set_yticklabels(make_counts.index)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, method='pearson', save_path=None):
        """
        Plot correlation heatmap for numerical variables.
        
        Args:
            method (str): Correlation method
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['Year', 'Month', 'Quarter']]
        
        corr_matrix = self.df[numerical_cols].corr(method=method)
        
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title(f'{method.title()} Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_bivariate_relationships(self, save_path=None):
        """
        Plot key bivariate relationships.
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_MEDIUM)
        fig.suptitle('Key Bivariate Relationships', fontsize=16, fontweight='bold')
        
        # TotalPremium vs TotalClaims
        axes[0, 0].scatter(self.df['TotalPremium'], self.df['TotalClaims'], alpha=0.5, s=10)
        axes[0, 0].set_xlabel('Total Premium')
        axes[0, 0].set_ylabel('Total Claims')
        axes[0, 0].set_title('Total Premium vs Total Claims')
        max_val = max(self.df['TotalPremium'].max(), self.df['TotalClaims'].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='Break-even line')
        axes[0, 0].legend()
        
        # CustomValueEstimate vs TotalPremium
        if 'CustomValueEstimate' in self.df.columns:
            valid_data = self.df[(self.df['CustomValueEstimate'] > 0) & (self.df['TotalPremium'] > 0)]
            axes[0, 1].scatter(valid_data['CustomValueEstimate'], valid_data['TotalPremium'], alpha=0.5, s=10)
            axes[0, 1].set_xlabel('Custom Value Estimate')
            axes[0, 1].set_ylabel('Total Premium')
            axes[0, 1].set_title('Custom Value Estimate vs Total Premium')
        
        # RegistrationYear vs LossRatio
        if 'RegistrationYear' in self.df.columns and 'LossRatio' in self.df.columns:
            valid_data = self.df[(self.df['RegistrationYear'] > 1990) & 
                                (self.df['LossRatio'].notna()) & 
                                (self.df['LossRatio'] >= 0) & 
                                (self.df['LossRatio'] <= 5)]
            axes[1, 0].scatter(valid_data['RegistrationYear'], valid_data['LossRatio'], alpha=0.5, s=10)
            axes[1, 0].set_xlabel('Registration Year')
            axes[1, 0].set_ylabel('Loss Ratio')
            axes[1, 0].set_title('Registration Year vs Loss Ratio')
            axes[1, 0].axhline(y=BREAK_EVEN_LOSS_RATIO, color='r', linestyle='--', label='Break-even')
            axes[1, 0].legend()
        
        # SumInsured vs TotalPremium
        if 'SumInsured' in self.df.columns:
            valid_data = self.df[(self.df['SumInsured'] > 0) & (self.df['TotalPremium'] > 0)]
            axes[1, 1].scatter(valid_data['SumInsured'], valid_data['TotalPremium'], alpha=0.5, s=10)
            axes[1, 1].set_xlabel('Sum Insured')
            axes[1, 1].set_ylabel('Total Premium')
            axes[1, 1].set_title('Sum Insured vs Total Premium')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_trends(self, save_path=None):
        """
        Plot temporal trends in key metrics.
        
        Args:
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if 'TransactionMonth' not in self.df.columns:
            return None
        
        # Aggregate data by month
        monthly_trends = self.df.groupby('TransactionMonth').agg({
            'TotalPremium': ['sum', 'count'],
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).round(4)
        
        monthly_trends.columns = ['_'.join(col).strip() for col in monthly_trends.columns.values]
        monthly_trends = monthly_trends.reset_index()
        monthly_trends['Monthly_LossRatio'] = monthly_trends['TotalClaims_sum'] / monthly_trends['TotalPremium_sum']
        
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LARGE)
        fig.suptitle('Temporal Trends in Insurance Metrics', fontsize=16, fontweight='bold')
        
        # Total Premium over time
        axes[0, 0].plot(monthly_trends['TransactionMonth'], monthly_trends['TotalPremium_sum'], 
                       marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Total Monthly Premium')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Total Premium')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total Claims over time
        axes[0, 1].plot(monthly_trends['TransactionMonth'], monthly_trends['TotalClaims_sum'], 
                       marker='o', linewidth=2, markersize=6, color='red')
        axes[0, 1].set_title('Total Monthly Claims')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Total Claims')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Monthly Loss Ratio
        axes[1, 0].plot(monthly_trends['TransactionMonth'], monthly_trends['Monthly_LossRatio'], 
                       marker='o', linewidth=2, markersize=6, color='green')
        axes[1, 0].axhline(y=BREAK_EVEN_LOSS_RATIO, color='orange', linestyle='--', 
                          linewidth=2, label='Break-even (1.0)')
        axes[1, 0].set_title('Monthly Loss Ratio')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Loss Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        
        # Policy Count over time
        axes[1, 1].plot(monthly_trends['TransactionMonth'], monthly_trends['TotalPremium_count'], 
                       marker='o', linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_title('Monthly Policy Count')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Number of Policies')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_risk_profitability_matrix(self):
        """
        Create interactive risk-profitability matrix by province.
        
        Returns:
            plotly.graph_objects.Figure: Interactive plotly figure
        """
        # Prepare data
        province_data = self.df.groupby('Province').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).reset_index()
        
        province_data['LossRatio'] = province_data['TotalClaims'] / province_data['TotalPremium']
        province_data['MarketShare'] = (province_data['TotalPremium'] / province_data['TotalPremium'].sum()) * 100
        
        fig = go.Figure()
        
        for _, row in province_data.iterrows():
            color = COLORS['unprofitable'] if row['LossRatio'] > BREAK_EVEN_LOSS_RATIO else COLORS['profitable']
            
            fig.add_trace(go.Scatter(
                x=[row['MarketShare']],
                y=[row['LossRatio']],
                mode='markers+text',
                marker=dict(
                    size=row['PolicyID']/50,
                    color=color,
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                text=row['Province'] if row['MarketShare'] > 3 else '',
                textposition='top center',
                name=row['Province'],
                showlegend=False,
                hovertemplate=f"<b>{row['Province']}</b><br>" +
                             f"Market Share: {row['MarketShare']:.1f}%<br>" +
                             f"Loss Ratio: {row['LossRatio']:.3f}<br>" +
                             f"Policies: {row['PolicyID']:,}<br>" +
                             f"Status: {'UNPROFITABLE' if row['LossRatio'] > BREAK_EVEN_LOSS_RATIO else 'PROFITABLE'}<extra></extra>"
            ))
        
        fig.add_hline(y=BREAK_EVEN_LOSS_RATIO, line_dash="dash", line_color="orange", line_width=3,
                      annotation_text="Break-even Line (Loss Ratio = 1.0)", 
                      annotation_position="top right")
        
        fig.update_layout(
            title=dict(
                text="<b>Risk-Profitability Matrix by Province</b><br><sub>Bubble size represents policy count</sub>",
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title="Market Share (%)",
            yaxis_title="Loss Ratio",
            width=900,
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_temporal_evolution_dashboard(self):
        """
        Create comprehensive temporal evolution dashboard.
        
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        if 'TransactionMonth' not in self.df.columns:
            return None
        
        monthly_data = self.df.groupby('TransactionMonth').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique'
        }).reset_index()
        
        monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
        monthly_data['ProfitLoss'] = monthly_data['TotalPremium'] - monthly_data['TotalClaims']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Premium vs Claims Over Time', 'Monthly Loss Ratio Evolution', 
                           'Monthly Profit/Loss', 'Portfolio Growth (Policy Count)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # Premium vs Claims
        fig.add_trace(
            go.Scatter(x=monthly_data['TransactionMonth'], y=monthly_data['TotalPremium'],
                       mode='lines+markers', name='Premium', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_data['TransactionMonth'], y=monthly_data['TotalClaims'],
                       mode='lines+markers', name='Claims', line=dict(color='red', width=3)),
            row=1, col=1
        )
        
        # Loss Ratio with color coding
        colors = ['red' if lr > BREAK_EVEN_LOSS_RATIO else 'green' for lr in monthly_data['LossRatio']]
        fig.add_trace(
            go.Scatter(x=monthly_data['TransactionMonth'], y=monthly_data['LossRatio'],
                       mode='lines+markers', name='Loss Ratio', 
                       line=dict(color='orange', width=3),
                       marker=dict(size=10, color=colors)),
            row=1, col=2
        )
        fig.add_hline(y=BREAK_EVEN_LOSS_RATIO, line_dash="dash", line_color="orange", row=1, col=2)
        
        # Profit/Loss
        profit_colors = ['green' if pl > 0 else 'red' for pl in monthly_data['ProfitLoss']]
        fig.add_trace(
            go.Bar(x=monthly_data['TransactionMonth'], y=monthly_data['ProfitLoss'],
                   name='Profit/Loss', marker_color=profit_colors, opacity=0.8),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
        
        # Policy Count Growth
        fig.add_trace(
            go.Scatter(x=monthly_data['TransactionMonth'], y=monthly_data['PolicyID'],
                       mode='lines+markers', name='Policy Count', 
                       line=dict(color='purple', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text="<b>Temporal Evolution of Insurance Portfolio Performance</b>",
                x=0.5,
                font=dict(size=18)
            ),
            height=800,
            showlegend=False,
            template="plotly_white"
        )
        
        return fig

def create_visualizer(df):
    """
    Convenience function to create a visualizer instance.
    
    Args:
        df (pd.DataFrame): Dataset to visualize
        
    Returns:
        InsuranceVisualizer: Visualizer instance
    """
    return InsuranceVisualizer(df)
