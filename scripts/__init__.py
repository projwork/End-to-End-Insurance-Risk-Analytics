"""
Insurance Risk Analytics Scripts
Modular approach for comprehensive insurance risk analysis.
"""

# Import all modules for easy access
from .config import *
from .data_loader import InsuranceDataLoader, load_insurance_data
from .data_quality import DataQualityAssessor, assess_data_quality
from .statistical_analysis import StatisticalAnalyzer, analyze_statistics
from .visualization import InsuranceVisualizer, create_visualizer
from .business_analysis import BusinessAnalyzer, analyze_business_performance
from .utils import *

__version__ = "1.0.0"
__author__ = "Insurance Risk Analytics Team"

# Quick start functions
def quick_analysis(data_path=None, sample_size=100000):
    """
    Perform a quick comprehensive analysis of insurance data.
    
    Args:
        data_path (str): Path to the data file
        sample_size (int): Number of records to sample
        
    Returns:
        dict: Complete analysis results
    """
    print("🚀 Starting Quick Analysis...")
    
    # Load data
    loader = InsuranceDataLoader(data_path, sample_size)
    df = loader.load_and_prepare_data()
    
    # Run all analyses
    quality_assessor = assess_data_quality(df)
    statistical_analyzer = analyze_statistics(df)
    business_analyzer = analyze_business_performance(df)
    visualizer = create_visualizer(df)
    
    results = {
        'data_info': {
            'shape': df.shape,
            'columns': list(df.columns),
            'sample_size': len(df)
        },
        'data_quality': quality_assessor.quality_report,
        'statistical_analysis': statistical_analyzer.analysis_results,
        'business_analysis': business_analyzer.business_metrics,
        'visualizer': visualizer
    }
    
    print("✅ Quick Analysis completed!")
    return results

def generate_all_visualizations(df, output_dir="output/plots"):
    """
    Generate all standard visualizations for insurance data.
    
    Args:
        df (pd.DataFrame): Insurance dataset
        output_dir (str): Directory to save plots
        
    Returns:
        dict: Paths to generated plots
    """
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    visualizer = create_visualizer(df)
    plot_paths = {}
    
    print("📊 Generating visualizations...")
    
    # Generate static plots
    try:
        fig1 = visualizer.plot_financial_distributions(f"{output_dir}/financial_distributions.png")
        plot_paths['financial_distributions'] = f"{output_dir}/financial_distributions.png"
    except Exception as e:
        print(f"Warning: Could not generate financial distributions plot: {e}")
    
    try:
        fig2 = visualizer.plot_categorical_distributions(f"{output_dir}/categorical_distributions.png")
        plot_paths['categorical_distributions'] = f"{output_dir}/categorical_distributions.png"
    except Exception as e:
        print(f"Warning: Could not generate categorical distributions plot: {e}")
    
    try:
        fig3 = visualizer.plot_correlation_heatmap('pearson', f"{output_dir}/correlation_heatmap.png")
        plot_paths['correlation_heatmap'] = f"{output_dir}/correlation_heatmap.png"
    except Exception as e:
        print(f"Warning: Could not generate correlation heatmap: {e}")
    
    try:
        fig4 = visualizer.plot_bivariate_relationships(f"{output_dir}/bivariate_relationships.png")
        plot_paths['bivariate_relationships'] = f"{output_dir}/bivariate_relationships.png"
    except Exception as e:
        print(f"Warning: Could not generate bivariate relationships plot: {e}")
    
    try:
        fig5 = visualizer.plot_temporal_trends(f"{output_dir}/temporal_trends.png")
        plot_paths['temporal_trends'] = f"{output_dir}/temporal_trends.png"
    except Exception as e:
        print(f"Warning: Could not generate temporal trends plot: {e}")
    
    print("✅ Visualizations generated!")
    return plot_paths

def print_comprehensive_summary(df):
    """
    Print a comprehensive summary of all analyses.
    
    Args:
        df (pd.DataFrame): Insurance dataset
    """
    print_section_header("COMPREHENSIVE INSURANCE RISK ANALYTICS SUMMARY")
    
    # Data overview
    print_dataframe_info(df, "Insurance Dataset")
    
    # Quick analyses
    quality_assessor = assess_data_quality(df)
    quality_assessor.print_quality_summary()
    
    statistical_analyzer = analyze_statistics(df)
    statistical_analyzer.print_analysis_summary()
    
    business_analyzer = analyze_business_performance(df)
    business_analyzer.print_business_summary()

# Convenience imports for common functions
__all__ = [
    # Classes
    'InsuranceDataLoader',
    'DataQualityAssessor', 
    'StatisticalAnalyzer',
    'InsuranceVisualizer',
    'BusinessAnalyzer',
    
    # Quick functions
    'load_insurance_data',
    'assess_data_quality',
    'analyze_statistics',
    'create_visualizer',
    'analyze_business_performance',
    'quick_analysis',
    'generate_all_visualizations',
    'print_comprehensive_summary',
    
    # Utilities
    'setup_plotting_environment',
    'format_currency',
    'format_percentage',
    'format_number',
    'print_dataframe_info',
    'print_section_header',
    'print_subsection_header',
    'ProgressTracker',
    
    # Config constants
    'DATA_PATH',
    'SAMPLE_SIZE',
    'RANDOM_STATE',
    'KEY_FINANCIAL_VARS',
    'KEY_CATEGORICAL_VARS',
    'BREAK_EVEN_LOSS_RATIO'
]
