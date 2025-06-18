"""
Machine Learning Evaluation module for Insurance Risk Analytics
Provides comprehensive model evaluation, comparison, and reporting capabilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve, validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import FIGURE_SIZE_LARGE, FIGURE_SIZE_MEDIUM, COLORS
from .utils import print_section_header

warnings.filterwarnings('ignore')

class MLEvaluator:
    """
    Comprehensive ML model evaluation class.
    
    Provides evaluation metrics, visualizations, and reports for:
    - Regression models (claim severity, premium prediction)
    - Classification models (claim probability)
    - Model comparison and selection
    - Business impact analysis
    """
    
    def __init__(self):
        """Initialize the ML evaluator."""
        self.evaluation_results = {}
        self.model_comparisons = {}
        
    def evaluate_regression_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str, dataset_type: str = 'test') -> Dict:
        """
        Comprehensive evaluation of regression models.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            dataset_type (str): 'train' or 'test'
            
        Returns:
            Dict: Evaluation metrics
        """
        # Core regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        max_error = np.max(np.abs(y_true - y_pred))
        median_ae = np.median(np.abs(y_true - y_pred))
        
        # Residual analysis
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Prediction intervals
        prediction_error_95th = np.percentile(np.abs(residuals), 95)
        prediction_error_90th = np.percentile(np.abs(residuals), 90)
        
        # Business metrics for insurance
        if np.mean(y_true) > 0:  # Avoid division by zero
            relative_rmse = rmse / np.mean(y_true)
            relative_mae = mae / np.mean(y_true)
        else:
            relative_rmse = relative_mae = np.nan
        
        evaluation = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'max_error': max_error,
            'median_absolute_error': median_ae,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'prediction_error_95th': prediction_error_95th,
            'prediction_error_90th': prediction_error_90th,
            'relative_rmse': relative_rmse,
            'relative_mae': relative_mae,
            'sample_size': len(y_true)
        }
        
        return evaluation
    
    def evaluate_classification_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    y_pred_proba: np.ndarray, model_name: str, 
                                    dataset_type: str = 'test') -> Dict:
        """
        Comprehensive evaluation of classification models.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            dataset_type (str): 'train' or 'test'
            
        Returns:
            Dict: Evaluation metrics
        """
        # Core classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Probability-based metrics
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # Business-relevant metrics for insurance
        claim_rate_actual = np.mean(y_true)
        claim_rate_predicted = np.mean(y_pred)
        
        # Gini coefficient (alternative to AUC)
        gini = 2 * auc_roc - 1
        
        evaluation = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'average_precision': avg_precision,
            'specificity': specificity,
            'npv': npv,
            'gini_coefficient': gini,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'claim_rate_actual': claim_rate_actual,
            'claim_rate_predicted': claim_rate_predicted,
            'sample_size': len(y_true)
        }
        
        return evaluation
    
    def create_regression_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str) -> go.Figure:
        """
        Create comprehensive regression evaluation plots.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Predictions vs Actual', 'Residual Plot',
                'Distribution of Residuals', 'Prediction Error Distribution'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # 1. Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred, mode='markers',
                name='Predictions', opacity=0.6,
                marker=dict(size=5, color='blue')
            ),
            row=1, col=1
        )
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Residual Plot
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode='markers',
                name='Residuals', opacity=0.6,
                marker=dict(size=5, color='green')
            ),
            row=1, col=2
        )
        # Zero line
        fig.add_trace(
            go.Scatter(
                x=[y_pred.min(), y_pred.max()], y=[0, 0],
                mode='lines', name='Zero Line',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Distribution of Residuals
        fig.add_trace(
            go.Histogram(
                x=residuals, name='Residual Distribution',
                opacity=0.7, nbinsx=30
            ),
            row=2, col=1
        )
        
        # 4. Prediction Error Distribution
        abs_errors = np.abs(residuals)
        fig.add_trace(
            go.Histogram(
                x=abs_errors, name='Absolute Error Distribution',
                opacity=0.7, nbinsx=30
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Regression Model Evaluation: {model_name}",
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_xaxes(title_text="Residuals", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Absolute Error", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def create_classification_plots(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                   model_name: str) -> go.Figure:
        """
        Create comprehensive classification evaluation plots.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred_proba (np.ndarray): Predicted probabilities
            model_name (str): Name of the model
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'ROC Curve', 'Precision-Recall Curve',
                'Probability Distribution', 'Calibration Plot'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'ROC Curve (AUC = {auc:.3f})',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        # Diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision, mode='lines',
                name=f'PR Curve (AP = {avg_precision:.3f})',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        # 3. Probability Distribution by Class
        prob_class_0 = y_pred_proba[y_true == 0]
        prob_class_1 = y_pred_proba[y_true == 1]
        
        fig.add_trace(
            go.Histogram(
                x=prob_class_0, name='No Claim',
                opacity=0.7, nbinsx=30, 
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Histogram(
                x=prob_class_1, name='Claim',
                opacity=0.7, nbinsx=30,
                marker_color='lightcoral'
            ),
            row=2, col=1
        )
        
        # 4. Calibration Plot
        # Bin the predictions and calculate actual vs predicted rates
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        predicted_rates = []
        actual_rates = []
        bin_centers = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            if np.sum(in_bin) > 0:
                predicted_rates.append(np.mean(y_pred_proba[in_bin]))
                actual_rates.append(np.mean(y_true[in_bin]))
                bin_centers.append((bin_lower + bin_upper) / 2)
        
        if predicted_rates:
            fig.add_trace(
                go.Scatter(
                    x=predicted_rates, y=actual_rates, mode='markers+lines',
                    name='Calibration',
                    marker=dict(size=8, color='purple')
                ),
                row=2, col=2
            )
            # Perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode='lines',
                    name='Perfect Calibration',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Classification Model Evaluation: {model_name}",
            height=800,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Mean Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Fraction of Positives", row=2, col=2)
        
        return fig
    
    def generate_model_comparison_report(self, model_results: Dict, 
                                       model_type: str = 'regression') -> pd.DataFrame:
        """
        Generate comprehensive model comparison report.
        
        Args:
            model_results (Dict): Results from multiple models
            model_type (str): 'regression' or 'classification'
            
        Returns:
            pd.DataFrame: Comparison report
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            if model_type == 'regression':
                # Extract regression metrics
                comparison_data.append({
                    'Model': model_name,
                    'RMSE': results.get('test_rmse', np.nan),
                    'MAE': results.get('test_mae', np.nan),
                    'R²': results.get('test_r2', np.nan),
                    'MAPE (%)': results.get('test_mape', np.nan),
                    'CV_RMSE': results.get('cv_rmse', np.nan),
                    'CV_STD': results.get('cv_std', np.nan),
                    'Relative_RMSE': results.get('relative_rmse', np.nan),
                    'Training_Time': results.get('training_time', np.nan)
                })
            else:
                # Extract classification metrics
                comparison_data.append({
                    'Model': model_name,
                    'AUC-ROC': results.get('test_auc', np.nan),
                    'Average_Precision': results.get('test_ap', np.nan),
                    'Accuracy': results.get('test_accuracy', np.nan),
                    'Precision': results.get('test_precision', np.nan),
                    'Recall': results.get('test_recall', np.nan),
                    'F1_Score': results.get('test_f1', np.nan),
                    'Gini': results.get('gini_coefficient', np.nan),
                    'CV_AUC': results.get('cv_auc', np.nan)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by best performance metric
        if model_type == 'regression':
            comparison_df = comparison_df.sort_values('R²', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
        
        return comparison_df
    
    def calculate_business_impact(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_type: str, business_params: Dict = None) -> Dict:
        """
        Calculate business impact metrics for insurance models.
        
        Args:
            y_true (np.ndarray): True values/labels
            y_pred (np.ndarray): Predicted values/probabilities
            model_type (str): 'claim_severity', 'claim_probability', or 'premium'
            business_params (Dict): Business parameters for calculations
            
        Returns:
            Dict: Business impact metrics
        """
        if business_params is None:
            business_params = {
                'expense_ratio': 0.25,
                'profit_margin': 0.15,
                'cost_of_capital': 0.08
            }
        
        business_impact = {}
        
        if model_type == 'claim_severity':
            # Reserve accuracy
            total_actual_claims = np.sum(y_true)
            total_predicted_claims = np.sum(y_pred)
            reserve_accuracy = 1 - abs(total_predicted_claims - total_actual_claims) / total_actual_claims
            
            # Economic impact of prediction errors
            prediction_errors = np.abs(y_true - y_pred)
            avg_prediction_error = np.mean(prediction_errors)
            
            business_impact = {
                'reserve_accuracy': reserve_accuracy,
                'total_actual_claims': total_actual_claims,
                'total_predicted_claims': total_predicted_claims,
                'avg_prediction_error': avg_prediction_error,
                'economic_impact_per_policy': avg_prediction_error * business_params['cost_of_capital']
            }
            
        elif model_type == 'claim_probability':
            # Classification business metrics
            if len(np.unique(y_true)) == 2:  # Binary classification
                # Calculate lift and capture rates
                # Sort by predicted probability
                sorted_indices = np.argsort(y_pred)[::-1]  # Descending order
                sorted_y_true = y_true[sorted_indices]
                
                # Top decile performance
                top_decile_size = len(y_true) // 10
                top_decile_capture_rate = np.sum(sorted_y_true[:top_decile_size]) / np.sum(y_true)
                
                # Expected vs actual claim rates
                baseline_claim_rate = np.mean(y_true)
                
                business_impact = {
                    'baseline_claim_rate': baseline_claim_rate,
                    'top_decile_capture_rate': top_decile_capture_rate,
                    'model_lift': top_decile_capture_rate / (1/10),  # Lift vs random
                    'potential_premium_adjustment': top_decile_capture_rate * 100  # % adjustment
                }
                
        elif model_type == 'premium':
            # Premium optimization metrics
            actual_premiums = y_true
            predicted_premiums = y_pred
            
            # Revenue impact
            total_actual_revenue = np.sum(actual_premiums)
            total_predicted_revenue = np.sum(predicted_premiums)
            revenue_accuracy = 1 - abs(total_predicted_revenue - total_actual_revenue) / total_actual_revenue
            
            # Pricing efficiency
            premium_errors = np.abs(actual_premiums - predicted_premiums)
            avg_premium_error = np.mean(premium_errors)
            relative_premium_error = avg_premium_error / np.mean(actual_premiums)
            
            business_impact = {
                'revenue_accuracy': revenue_accuracy,
                'total_actual_revenue': total_actual_revenue,
                'total_predicted_revenue': total_predicted_revenue,
                'avg_premium_error': avg_premium_error,
                'relative_premium_error': relative_premium_error,
                'pricing_efficiency_score': 1 - relative_premium_error
            }
        
        return business_impact


def create_ml_evaluator() -> MLEvaluator:
    """
    Convenience function to create ML evaluator.
    
    Returns:
        MLEvaluator: Configured evaluator instance
    """
    return MLEvaluator() 