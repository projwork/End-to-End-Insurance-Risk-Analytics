# Insurance Risk Analytics - Modular Scripts

This directory contains modular Python scripts for comprehensive insurance risk analytics. The code has been refactored from a monolithic notebook approach to a clean, maintainable, and reusable modular structure.

## 📁 Module Structure

```
scripts/
├── 📄 config.py                    # Configuration and constants
├── 📄 data_loader.py              # Data loading and preprocessing
├── 📄 data_quality.py             # Data quality assessment
├── 📄 statistical_analysis.py     # Statistical tests and analysis
├── 📄 visualization.py            # All plotting functions
├── 📄 business_analysis.py        # Business metrics and insights
├── 📄 utils.py                    # Helper functions and utilities
├── 📄 __init__.py                 # Package initialization
└── 📄 README.md                   # This documentation
```

## 🚀 Quick Start

### Basic Usage

```python
# Import all modules
from scripts import *

# Load and analyze data in one line
results = quick_analysis()

# Or with custom parameters
results = quick_analysis(sample_size=50000)
```

### Step-by-Step Analysis

```python
# Import specific modules
from scripts.data_loader import InsuranceDataLoader
from scripts.data_quality import DataQualityAssessor
from scripts.statistical_analysis import StatisticalAnalyzer
from scripts.visualization import InsuranceVisualizer
from scripts.business_analysis import BusinessAnalyzer

# Load data
loader = InsuranceDataLoader()
df = loader.load_and_prepare_data()

# Assess data quality
quality_assessor = DataQualityAssessor(df)
quality_report = quality_assessor.generate_comprehensive_analysis()

# Perform statistical analysis
statistical_analyzer = StatisticalAnalyzer(df)
stats_results = statistical_analyzer.generate_comprehensive_analysis()

# Create visualizations
visualizer = InsuranceVisualizer(df)
fig = visualizer.plot_financial_distributions()

# Analyze business metrics
business_analyzer = BusinessAnalyzer(df)
business_results = business_analyzer.generate_comprehensive_analysis()
```

## 🎯 Benefits of Modular Approach

### ✅ **Maintainability**

- Clear separation of concerns
- Easy to update individual components
- Reduced code duplication

### ✅ **Reusability**

- Components can be used across projects
- Consistent analysis methodology
- Plug-and-play functionality

### ✅ **Testability**

- Each module can be tested independently
- Better error isolation
- Easier debugging

### ✅ **Scalability**

- Easy to add new analysis types
- Modular expansion of capabilities
- Performance optimization by module

### ✅ **Collaboration**

- Multiple developers can work on different modules
- Clear interfaces between components
- Better version control

## 🔧 Example Usage

```python
# Complete analysis in one line
results = quick_analysis()

# Generate all visualizations
plot_paths = generate_all_visualizations(df)

# Print comprehensive summary
print_comprehensive_summary(df)
```

## 📄 Module Overview

- **config.py**: Central configuration and constants
- **data_loader.py**: Data loading and preprocessing
- **data_quality.py**: Data quality assessment
- **statistical_analysis.py**: Statistical tests and analysis
- **visualization.py**: All plotting functions
- **business_analysis.py**: Business metrics and insights
- **utils.py**: Helper functions and utilities

## 🚀 Getting Started

1. Import the modules: `from scripts import *`
2. Run quick analysis: `results = quick_analysis()`
3. Explore results with the various analyzer classes

This modular approach provides a clean, maintainable, and reusable framework for insurance risk analytics.
