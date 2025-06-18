"""
Machine Learning Models for Insurance Risk Analytics
Provides comprehensive modeling capabilities for claim prediction and premium optimization.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)

# Model imports
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Interpretability imports
import shap

from .config import RANDOM_STATE, STATISTICAL_SIGNIFICANCE
from .utils import print_section_header

warnings.filterwarnings('ignore')

class InsuranceMLFramework:
    """
    Comprehensive machine learning framework for insurance analytics.
    
    Supports:
    - Claim severity prediction (regression)
    - Claim probability prediction (classification) 
    - Premium optimization modeling
    - Feature engineering and preprocessing
    - Model evaluation and interpretation
    """
    
    def __init__(self, df: pd.DataFrame, random_state: int = RANDOM_STATE):
        """
        Initialize the ML framework.
        
        Args:
            df (pd.DataFrame): Insurance dataset
            random_state (int): Random state for reproducibility
        """
        self.df = df.copy()
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.shap_values = {}
        
        # Initialize preprocessing components
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def prepare_features(self, target_type: str = 'claim_severity') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling based on target type.
        
        Args:
            target_type (str): Type of target ('claim_severity', 'claim_probability', 'premium')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        print(f"🔧 Preparing features for {target_type} modeling...")
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Define target variable
        if target_type == 'claim_severity':
            # Only policies with claims > 0
            df_processed = df_processed[df_processed['TotalClaims'] > 0].copy()
            target = df_processed['TotalClaims']
            print(f"   Claim severity dataset: {len(df_processed):,} policies with claims")
            
        elif target_type == 'claim_probability':
            # Binary classification: HasClaim (0/1)
            df_processed['HasClaim'] = (df_processed['TotalClaims'] > 0).astype(int)
            target = df_processed['HasClaim']
            print(f"   Claim probability dataset: {len(df_processed):,} policies")
            print(f"   Claim rate: {target.mean():.3%}")
            
        elif target_type == 'premium':
            # Premium prediction
            target = df_processed['TotalPremium']
            print(f"   Premium optimization dataset: {len(df_processed):,} policies")
            
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        # Select features
        features = self._select_features(df_processed, target_type)
        
        print(f"   Features selected: {len(features.columns)}")
        print(f"   Target variable: {target.name if hasattr(target, 'name') else 'target'}")
        
        return features, target
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for modeling.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        print("   Engineering features...")
        
        # Vehicle age
        if 'RegistrationYear' in df.columns:
            current_year = pd.Timestamp.now().year
            df['VehicleAge'] = current_year - df['RegistrationYear']
            df['VehicleAge'] = df['VehicleAge'].clip(0, 50).astype(float)  # Ensure numeric type
        
        # Date features - ensure they are numeric
        if 'TransactionMonth' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['TransactionMonth']):
                df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
            
            df['Year'] = df['TransactionMonth'].dt.year.astype(float)
            df['Month'] = df['TransactionMonth'].dt.month.astype(float)
            df['Quarter'] = df['TransactionMonth'].dt.quarter.astype(float)
        
        # Risk indicators
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
        df['LossRatio'] = df['LossRatio'].fillna(0).clip(0, 10).astype(float)
        
        # Premium per sum insured ratio
        if 'SumInsured' in df.columns:
            df['PremiumRate'] = df['TotalPremium'] / df['SumInsured'].replace(0, np.nan)
            df['PremiumRate'] = df['PremiumRate'].fillna(df['PremiumRate'].median()).astype(float)
        
        # Value estimation ratio
        if 'CustomValueEstimate' in df.columns:
            df['ValueRatio'] = df['SumInsured'] / df['CustomValueEstimate'].replace(0, np.nan)
            df['ValueRatio'] = df['ValueRatio'].fillna(1).clip(0, 5).astype(float)
        
        # Categorical feature engineering - ensure string types
        # Convert all original categorical columns to strings first
        categorical_cols_original = ['Province', 'PostalCode', 'make', 'VehicleType', 
                                   'Gender', 'MaritalStatus', 'Citizenship', 'LegalType']
        
        for col in categorical_cols_original:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('Unknown')
        
        if 'PostalCode' in df.columns:
            # Group low-frequency postal codes
            postal_counts = df['PostalCode'].value_counts()
            low_freq_postal = postal_counts[postal_counts < 50].index
            df['PostalCode_Grouped'] = df['PostalCode'].replace(low_freq_postal, 'Other').astype(str)
        
        if 'make' in df.columns:
            # Group low-frequency vehicle makes
            make_counts = df['make'].value_counts()
            low_freq_makes = make_counts[make_counts < 100].index
            df['make_Grouped'] = df['make'].replace(low_freq_makes, 'Other').astype(str)
        
        print(f"   Engineered features created. Dataset shape: {df.shape}")
        return df
    
    def _select_features(self, df: pd.DataFrame, target_type: str) -> pd.DataFrame:
        """
        Select relevant features for modeling.
        
        Args:
            df (pd.DataFrame): Processed dataset
            target_type (str): Type of target variable
            
        Returns:
            pd.DataFrame: Selected features
        """
        # Define feature categories
        numerical_features = [
            'VehicleAge', 'SumInsured', 'CustomValueEstimate', 'TotalPremium',
            'PremiumRate', 'ValueRatio', 'Year', 'Month', 'Quarter'
        ]
        
        categorical_features = [
            'Province', 'PostalCode_Grouped', 'make_Grouped', 'VehicleType',
            'Gender', 'MaritalStatus', 'Citizenship', 'LegalType'
        ]
        
        # Select features that exist in the dataset
        available_numerical = [f for f in numerical_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        # Combine features
        selected_features = available_numerical + available_categorical
        
        # Remove target-related features for certain models
        if target_type == 'claim_severity':
            # Remove premium-related features to avoid data leakage
            selected_features = [f for f in selected_features if 'Premium' not in f]
        elif target_type == 'claim_probability':
            # Remove claim-related features
            selected_features = [f for f in selected_features if 'Claim' not in f and 'Loss' not in f]
        
        # Return selected features
        feature_df = df[selected_features].copy()
        
        # Ensure proper data types
        numerical_features = [
            'VehicleAge', 'SumInsured', 'CustomValueEstimate', 'TotalPremium',
            'PremiumRate', 'ValueRatio', 'Year', 'Month', 'Quarter'
        ]
        
        categorical_features = [
            'Province', 'PostalCode_Grouped', 'make_Grouped', 'VehicleType',
            'Gender', 'MaritalStatus', 'Citizenship', 'LegalType'
        ]
        
        # Convert numerical columns to float
        for col in numerical_features:
            if col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').astype(float)
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        
        # Convert categorical columns to string
        for col in categorical_features:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].astype(str).fillna('Unknown')
        
        return feature_df
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """
        Preprocess data for machine learning.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set proportion
            
        Returns:
            Tuple: Processed train/test splits
        """
        print(f"🔄 Preprocessing data...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )
        
        # Identify numerical and categorical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Additional type cleaning for categorical columns
        for col in categorical_cols:
            # Convert any remaining mixed types to strings
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Categorical features: {len(categorical_cols)}")
        
        # Debug: Check for any problematic columns
        for col in categorical_cols:
            unique_types = set(type(x).__name__ for x in X_train[col].dropna().iloc[:100])
            if len(unique_types) > 1:
                print(f"   Warning: {col} has mixed types: {unique_types}")
                # Force convert to string
                X_train[col] = X_train[col].astype(str)
                X_test[col] = X_test[col].astype(str)
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Fit and transform data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Store preprocessor
        self.preprocessor = preprocessor
        
        # Get feature names after preprocessing
        feature_names = (
            numerical_cols +
            list(preprocessor.named_transformers_['cat']
                 .named_steps['onehot'].get_feature_names_out(categorical_cols))
        )
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        # Convert to DataFrames for easier handling
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
        
        print(f"   Final feature count: {X_train_processed.shape[1]}")
        print(f"   Training set: {X_train_processed.shape[0]:,} samples")
        print(f"   Test set: {X_test_processed.shape[0]:,} samples")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def build_regression_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series,
                               model_type: str = 'claim_severity') -> Dict:
        """
        Build and train regression models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_type (str): Type of regression model
            
        Returns:
            Dict: Trained models and results
        """
        print_section_header(f"🤖 Building Regression Models - {model_type}")
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_root_mean_squared_error', n_jobs=-1)
                cv_rmse = -cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_rmse': cv_rmse,
                    'cv_std': cv_std,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test
                }
                
                print(f"   Train RMSE: {train_rmse:,.2f}")
                print(f"   Test RMSE: {test_rmse:,.2f}")
                print(f"   Test R²: {test_r2:.4f}")
                print(f"   CV RMSE: {cv_rmse:,.2f} (±{cv_std:,.2f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        self.models[model_type] = results
        return results
    
    def build_classification_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Build and train classification models for claim probability.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dict: Trained models and results
        """
        print_section_header("🎯 Building Classification Models - Claim Probability")
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1, verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                y_pred_proba_train = model.predict_proba(X_train)[:, 1]
                y_pred_proba_test = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                train_auc = roc_auc_score(y_train, y_pred_proba_train)
                test_auc = roc_auc_score(y_test, y_pred_proba_test)
                train_ap = average_precision_score(y_train, y_pred_proba_train)
                test_ap = average_precision_score(y_test, y_pred_proba_test)
                
                # Cross-validation
                cv_auc_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                              scoring='roc_auc', n_jobs=-1)
                cv_auc = cv_auc_scores.mean()
                cv_auc_std = cv_auc_scores.std()
                
                results[name] = {
                    'model': model,
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'train_ap': train_ap,
                    'test_ap': test_ap,
                    'cv_auc': cv_auc,
                    'cv_auc_std': cv_auc_std,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test,
                    'probabilities_train': y_pred_proba_train,
                    'probabilities_test': y_pred_proba_test
                }
                
                print(f"   Train AUC: {train_auc:.4f}")
                print(f"   Test AUC: {test_auc:.4f}")
                print(f"   Test AP: {test_ap:.4f}")
                print(f"   CV AUC: {cv_auc:.4f} (±{cv_auc_std:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        self.models['claim_probability'] = results
        return results
    
    def evaluate_models(self, model_type: str) -> pd.DataFrame:
        """
        Create comprehensive model evaluation comparison.
        
        Args:
            model_type (str): Type of models to evaluate
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        print_section_header(f"📊 Model Evaluation - {model_type}")
        
        if model_type not in self.models:
            print(f"❌ No models found for {model_type}")
            return pd.DataFrame()
        
        results = self.models[model_type]
        comparison_data = []
        
        for name, result in results.items():
            if model_type in ['claim_severity', 'premium']:
                # Regression metrics
                row = {
                    'Model': name,
                    'Train_RMSE': result['train_rmse'],
                    'Test_RMSE': result['test_rmse'],
                    'Train_R2': result['train_r2'],
                    'Test_R2': result['test_r2'],
                    'Train_MAE': result['train_mae'],
                    'Test_MAE': result['test_mae'],
                    'CV_RMSE': result['cv_rmse'],
                    'CV_STD': result['cv_std'],
                    'Overfitting': result['train_rmse'] - result['test_rmse']
                }
            else:
                # Classification metrics
                row = {
                    'Model': name,
                    'Train_AUC': result['train_auc'],
                    'Test_AUC': result['test_auc'],
                    'Train_AP': result['train_ap'],
                    'Test_AP': result['test_ap'],
                    'CV_AUC': result['cv_auc'],
                    'CV_AUC_STD': result['cv_auc_std'],
                    'Overfitting': result['train_auc'] - result['test_auc']
                }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by best performance
        if model_type in ['claim_severity', 'premium']:
            comparison_df = comparison_df.sort_values('Test_R2', ascending=False)
            print("📈 Regression Model Performance (sorted by Test R²):")
        else:
            comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)
            print("📈 Classification Model Performance (sorted by Test AUC):")
        
        print(comparison_df.round(4))
        
        return comparison_df
    
    def analyze_feature_importance(self, model_type: str, top_n: int = 10) -> Dict:
        """
        Analyze feature importance for the best model.
        
        Args:
            model_type (str): Type of model
            top_n (int): Number of top features to analyze
            
        Returns:
            Dict: Feature importance analysis
        """
        print_section_header(f"🔍 Feature Importance Analysis - {model_type}")
        
        if model_type not in self.models:
            print(f"❌ No models found for {model_type}")
            return {}
        
        # Get the best model (by test performance)
        results = self.models[model_type]
        if model_type in ['claim_severity', 'premium']:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        else:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        
        best_model = results[best_model_name]['model']
        print(f"🏆 Best model: {best_model_name}")
        
        # Extract feature importance
        feature_importance = {}
        
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            importances = best_model.feature_importances_
            feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(len(importances))])
            
        elif hasattr(best_model, 'coef_'):
            # Linear models
            importances = np.abs(best_model.coef_)
            feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(len(importances))])
            
        else:
            print("❌ Model doesn't support feature importance extraction")
            return {}
        
        # Ensure feature names and importances have the same length
        if len(feature_names) != len(importances):
            print(f"⚠️  Mismatch: {len(feature_names)} feature names, {len(importances)} importance values")
            min_length = min(len(feature_names), len(importances))
            feature_names = feature_names[:min_length]
            importances = importances[:min_length]
            print(f"   Truncated to {min_length} features")
        
        # Create importance dataframe
        if len(importances.shape) > 1:
            importances = importances[0]  # For multi-class problems
            
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display top features
        top_features = importance_df.head(top_n)
        print(f"\n📊 Top {top_n} Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"   {row['Feature']}: {row['Importance']:.4f}")
        
        feature_importance[model_type] = {
            'best_model': best_model_name,
            'importance_df': importance_df,
            'top_features': top_features
        }
        
        self.feature_importance[model_type] = feature_importance[model_type]
        return feature_importance
    
    def generate_shap_analysis(self, model_type: str, X_sample: pd.DataFrame, 
                              sample_size: int = 1000) -> Dict:
        """
        Generate SHAP analysis for model interpretability.
        
        Args:
            model_type (str): Type of model
            X_sample (pd.DataFrame): Sample data for SHAP analysis
            sample_size (int): Sample size for SHAP calculation
            
        Returns:
            Dict: SHAP analysis results
        """
        print_section_header(f"🔮 SHAP Analysis - {model_type}")
        
        if model_type not in self.models:
            print(f"❌ No models found for {model_type}")
            return {}
        
        # Get the best model
        results = self.models[model_type]
        if model_type in ['claim_severity', 'premium']:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        else:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        
        best_model = results[best_model_name]['model']
        
        # Sample data for SHAP (to reduce computation time)
        if len(X_sample) > sample_size:
            X_shap = X_sample.sample(sample_size, random_state=self.random_state)
        else:
            X_shap = X_sample.copy()
        
        print(f"🔧 Computing SHAP values for {best_model_name} with {len(X_shap):,} samples...")
        
        try:
            # Create SHAP explainer
            if 'Linear' in best_model_name:
                explainer = shap.LinearExplainer(best_model, X_shap)
            elif 'XGBoost' in best_model_name:
                explainer = shap.TreeExplainer(best_model)
            elif 'LightGBM' in best_model_name:
                explainer = shap.TreeExplainer(best_model)
            elif 'Random Forest' in best_model_name:
                explainer = shap.TreeExplainer(best_model)
            else:
                # Use kernel explainer as fallback
                explainer = shap.KernelExplainer(best_model.predict, X_shap.iloc[:100])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_shap)
            
            # Handle classification models with multiple outputs
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Create SHAP summary
            feature_names = X_shap.columns.tolist()
            
            # Ensure consistent lengths for SHAP values
            if len(feature_names) != shap_values.shape[1]:
                print(f"⚠️  SHAP mismatch: {len(feature_names)} feature names, {shap_values.shape[1]} SHAP columns")
                min_length = min(len(feature_names), shap_values.shape[1])
                feature_names = feature_names[:min_length]
                shap_values = shap_values[:, :min_length]
                print(f"   Truncated to {min_length} features")
            
            shap_importance = pd.DataFrame({
                'Feature': feature_names,
                'SHAP_Importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('SHAP_Importance', ascending=False)
            
            print(f"✅ SHAP analysis completed")
            print(f"\n📊 Top 10 Features by SHAP Importance:")
            for idx, row in shap_importance.head(10).iterrows():
                print(f"   {row['Feature']}: {row['SHAP_Importance']:.4f}")
            
            shap_results = {
                'model_name': best_model_name,
                'shap_values': shap_values,
                'explainer': explainer,
                'feature_importance': shap_importance,
                'sample_data': X_shap
            }
            
            self.shap_values[model_type] = shap_results
            return shap_results
            
        except Exception as e:
            print(f"❌ Error in SHAP analysis: {e}")
            return {}


def create_ml_framework(df: pd.DataFrame) -> InsuranceMLFramework:
    """
    Convenience function to create ML framework.
    
    Args:
        df (pd.DataFrame): Insurance dataset
        
    Returns:
        InsuranceMLFramework: Configured ML framework
    """
    return InsuranceMLFramework(df) 