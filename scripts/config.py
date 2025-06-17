"""
Configuration module for Insurance Risk Analytics
Contains all constants, settings, and configurations used across the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "MachineLearningRating_v3.txt"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"

# Data processing settings
SAMPLE_SIZE = 100000  # Number of rows to sample for analysis
RANDOM_STATE = 42     # For reproducible sampling

# Statistical thresholds
CORRELATION_THRESHOLD = 0.3        # Threshold for "strong" correlation
OUTLIER_IQR_MULTIPLIER = 1.5      # IQR multiplier for outlier detection
STATISTICAL_SIGNIFICANCE = 0.05    # P-value threshold for significance
MIN_POLICY_COUNT = 50              # Minimum policies for statistical significance

# Business metrics
BREAK_EVEN_LOSS_RATIO = 1.0       # Break-even point for loss ratio
HIGH_RISK_THRESHOLD = 1.0          # Loss ratio threshold for high risk
LOW_RISK_THRESHOLD = 0.8           # Loss ratio threshold for low risk

# Visualization settings
FIGURE_SIZE_LARGE = (16, 12)
FIGURE_SIZE_MEDIUM = (15, 12)
FIGURE_SIZE_SMALL = (10, 8)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8'

# Color schemes
COLORS = {
    'profitable': 'green',
    'unprofitable': 'red',
    'neutral': 'orange',
    'primary': 'blue',
    'secondary': 'purple',
    'accent': 'gold'
}

# Column categories for organized analysis
COLUMN_CATEGORIES = {
    'policy': ['UnderwrittenCoverID', 'PolicyID'],
    'client': ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
               'Bank', 'AccountType', 'MaritalStatus', 'Gender'],
    'location': ['Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone'],
    'vehicle': ['ItemType', 'mmcode', 'VehicleType', 'RegistrationYear', 'make', 'Model',
                'Cylinders', 'cubiccapacity', 'kilowatts', 'bodytype', 'NumberOfDoors',
                'VehicleIntroDate', 'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice',
                'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted',
                'CrossBorder', 'NumberOfVehiclesInFleet'],
    'plan': ['SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 'ExcessSelected',
             'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product',
             'StatutoryClass', 'StatutoryRiskType'],
    'financial': ['TotalPremium', 'TotalClaims', 'LossRatio']
}

# Key variables for analysis
KEY_FINANCIAL_VARS = ['TotalPremium', 'TotalClaims', 'LossRatio', 'CustomValueEstimate', 'SumInsured']
KEY_CATEGORICAL_VARS = ['Province', 'Gender', 'VehicleType', 'make', 'CoverType', 'CoverGroup']

# Display settings
PANDAS_DISPLAY_OPTIONS = {
    'display.max_columns': None,
    'display.width': None,
    'display.max_colwidth': None
} 