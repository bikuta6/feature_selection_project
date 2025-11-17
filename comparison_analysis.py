"""
Comprehensive comparison script for Feature Enhancement across datasets and models.

This script:
1. Runs feature enhancement on all available datasets using Ridge regression for enhancement
2. Compares performance across multiple regression models
3. Generates a CSV file with detailed results
4. Provides summary statistics and visualizations
"""

import json
import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add the project to path
sys.path.append(str(Path(__file__).parent))

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from feature_enhancer import DatasetLoader, FeatureEnhancer


def get_regression_models():
    """Get dictionary of regression models to test."""
    return {
        'Linear': LinearRegression(n_jobs=-1),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42, max_iter=2000),
        'RandomForest': RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        'DecisionTree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=50, random_state=42
        ),
        'MLP': MLPRegressor(
            hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
        ),
        'SVR': SVR(kernel='rbf', C=1.0)
    }


def evaluate_regression_model(model, X_train, X_test, y_train, y_test):
    """Evaluate a regression model and return metrics."""
    try:
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'train_time': train_time,
            'predict_time': predict_time,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'mae': np.nan,
            'mse': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'train_time': np.nan,
            'predict_time': np.nan,
            'success': False,
            'error': str(e)
        }


def process_dataset(dataset_path, dataset_name, target_column=-1):
    """Process a single dataset and return results for all models."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    results = []
    
    try:
        # Load and preprocess dataset
        print(f"Loading dataset...")
        X, y = DatasetLoader.load_csv(dataset_path, target_column=target_column)
        
        # Get dataset info
        dataset_info = DatasetLoader.get_dataset_info(X, y)
        print(f"Dataset shape: {X.shape}")
        print(f"Target type: {dataset_info['target_type']}")
        
        # Preprocess
        X_processed, y_processed = DatasetLoader.preprocess_dataset(
            X, y, handle_missing="drop", encode_categorical=True, target_type="auto"
        )
        
        print(f"Processed shape: {X_processed.shape}")
        
        # Split data
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_processed, y_processed, test_size=0.2, random_state=42
        )
        
        # Feature Enhancement using Ridge regression
        print(f"Applying feature enhancement with Ridge regression...")
        enhancer = FeatureEnhancer(
            synthesis_config={},  # Use default synthesis config
            selection_config={},  # Use default selection config
            scale_features=True,
            random_state=42,
            verbose=False
        )
        
        # Use Ridge regression for the enhancement process
        ridge_model = Ridge(random_state=42)
        
        # Apply enhancement
        enhancement_start = time.time()
        X_enhanced = enhancer.fit_transform(X_train_full, y_train_full, ridge_model)
        X_test_enhanced = enhancer.transform(X_test)
        enhancement_time = time.time() - enhancement_start
        
        # Get feature info
        feature_info = enhancer.get_feature_info()
        
        print(f"Enhancement completed in {enhancement_time:.2f}s")
        print(f"Original features: {feature_info['n_features_original']}")
        print(f"Final features: {feature_info['n_features_final']}")
        print(f"Synthesis performed: {feature_info['synthesis_performed']}")
        print(f"Selection performed: {feature_info['selection_performed']}")
        
        # Test all regression models
        models = get_regression_models()
        
        print(f"\nTesting {len(models)} regression models...")
        
        for model_name, model in models.items():
            print(f"  Testing {model_name}...")
            
            # Baseline evaluation
            baseline_results = evaluate_regression_model(
                model, X_train_full, X_test, y_train_full, y_test
            )
            
            # Enhanced evaluation
            enhanced_results = evaluate_regression_model(
                model, X_enhanced, X_test_enhanced, y_train_full, y_test
            )
            
            # Calculate improvements (focusing on MAE as primary metric)
            mae_improvement_pct = (
                (baseline_results['mae'] - enhanced_results['mae']) / baseline_results['mae'] * 100
                if baseline_results['success'] and enhanced_results['success'] and baseline_results['mae'] != 0
                else np.nan
            )
            
            mae_absolute_improvement = (
                baseline_results['mae'] - enhanced_results['mae']
                if baseline_results['success'] and enhanced_results['success']
                else np.nan
            )
            
            r2_improvement = (
                enhanced_results['r2'] - baseline_results['r2']
                if baseline_results['success'] and enhanced_results['success']
                else np.nan
            )
            
            # Store results
            result = {
                'dataset': dataset_name,
                'model': model_name,
                'original_features': feature_info['n_features_original'],
                'final_features': feature_info['n_features_final'],
                'synthesis_performed': feature_info['synthesis_performed'],
                'selection_performed': feature_info['selection_performed'],
                'feature_reduction_ratio': feature_info['summary']['feature_reduction_ratio'],
                'enhancement_time_seconds': enhancement_time,
                
                # Baseline metrics
                'baseline_r2': baseline_results['r2'],
                'baseline_mae': baseline_results['mae'],
                'baseline_mse': baseline_results['mse'],
                'baseline_rmse': baseline_results['rmse'],
                'baseline_train_time': baseline_results['train_time'],
                'baseline_predict_time': baseline_results['predict_time'],
                'baseline_success': baseline_results['success'],
                'baseline_error': baseline_results['error'],
                
                # Enhanced metrics
                'enhanced_r2': enhanced_results['r2'],
                'enhanced_mae': enhanced_results['mae'],
                'enhanced_mse': enhanced_results['mse'],
                'enhanced_rmse': enhanced_results['rmse'],
                'enhanced_train_time': enhanced_results['train_time'],
                'enhanced_predict_time': enhanced_results['predict_time'],
                'enhanced_success': enhanced_results['success'],
                'enhanced_error': enhanced_results['error'],
                
                # Improvements (MAE as primary metric)
                'mae_improvement_pct': mae_improvement_pct,
                'mae_absolute_improvement': mae_absolute_improvement,
                'r2_improvement': r2_improvement,
                
                # Dataset info
                'n_samples': X_processed.shape[0],
                'train_samples': X_train_full.shape[0],
                'test_samples': X_test.shape[0],
                'target_type': dataset_info['target_type']
            }
            
            results.append(result)
            
            # Print brief results
            if baseline_results['success'] and enhanced_results['success']:
                print(f"    Baseline MAE: {baseline_results['mae']:.4f}, Enhanced MAE: {enhanced_results['mae']:.4f}")
                print(f"    MAE improvement: {mae_improvement_pct:+.2f}%")
                print(f"    R² improvement: {r2_improvement:+.4f}")
            else:
                print(f"    Error occurred during evaluation")
    
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def generate_summary_report(df, output_dir):
    """Generate summary statistics and save to file."""
    print(f"\n{'='*60}")
    print(f"SUMMARY REPORT")
    print(f"{'='*60}")
    
    # Filter successful results
    successful_df = df[(df['baseline_success'] == True) & (df['enhanced_success'] == True)]
    
    if len(successful_df) == 0:
        print("No successful evaluations to summarize.")
        return
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"Total evaluations: {len(df)}")
    print(f"Successful evaluations: {len(successful_df)}")
    print(f"Success rate: {len(successful_df)/len(df)*100:.1f}%")
    
    # Performance improvements (MAE as primary metric)
    print(f"\nMAE Improvements (Primary Metric):")
    mae_improvements = successful_df['mae_improvement_pct'].dropna()
    print(f"Mean improvement: {mae_improvements.mean():+.2f}%")
    print(f"Median improvement: {mae_improvements.median():+.2f}%")
    print(f"Std improvement: {mae_improvements.std():.2f}%")
    print(f"Min improvement: {mae_improvements.min():+.2f}%")
    print(f"Max improvement: {mae_improvements.max():+.2f}%")
    print(f"Positive improvements: {(mae_improvements > 0).sum()}/{len(mae_improvements)} ({(mae_improvements > 0).mean()*100:.1f}%)")
    
    # R² improvements (secondary metric)
    print(f"\nR² Improvements (Secondary):")
    r2_improvements = successful_df['r2_improvement'].dropna()
    print(f"Mean improvement: {r2_improvements.mean():+.4f}")
    print(f"Median improvement: {r2_improvements.median():+.4f}")
    print(f"Positive improvements: {(r2_improvements > 0).sum()}/{len(r2_improvements)} ({(r2_improvements > 0).mean()*100:.1f}%)")
    
    # Best performing models
    print(f"\nTop 5 Best MAE Improvements:")
    top_mae = successful_df.nlargest(5, 'mae_improvement_pct')[['dataset', 'model', 'mae_improvement_pct', 'baseline_mae', 'enhanced_mae']]
    for _, row in top_mae.iterrows():
        print(f"  {row['dataset']} + {row['model']}: {row['mae_improvement_pct']:+.2f}% ({row['baseline_mae']:.4f} → {row['enhanced_mae']:.4f})")
        
    print(f"\nTop 5 Best R² Improvements:")
    top_r2 = successful_df.nlargest(5, 'r2_improvement')[['dataset', 'model', 'r2_improvement', 'baseline_r2', 'enhanced_r2']]
    for _, row in top_r2.iterrows():
        print(f"  {row['dataset']} + {row['model']}: {row['r2_improvement']:+.4f} ({row['baseline_r2']:.4f} → {row['enhanced_r2']:.4f})")
    
    # Model performance summary
    print(f"\nModel Performance Summary:")
    model_summary = successful_df.groupby('model').agg({
        'mae_improvement_pct': ['mean', 'std', 'count'],
        'r2_improvement': ['mean', 'std']
    }).round(4)
    
    for model in model_summary.index:
        mae_mean = model_summary.loc[model, ('mae_improvement_pct', 'mean')]
        mae_std = model_summary.loc[model, ('mae_improvement_pct', 'std')]
        count = model_summary.loc[model, ('mae_improvement_pct', 'count')]
        r2_mean = model_summary.loc[model, ('r2_improvement', 'mean')]
        print(f"  {model}: MAE {mae_mean:+.2f}±{mae_std:.2f}% (n={count}), R² {r2_mean:+.4f}")
    
    # Dataset performance summary
    print(f"\nDataset Performance Summary:")
    dataset_summary = successful_df.groupby('dataset').agg({
        'mae_improvement_pct': ['mean', 'std', 'count'],
        'r2_improvement': ['mean', 'std'],
        'original_features': 'first',
        'final_features': 'first',
        'feature_reduction_ratio': 'first'
    }).round(4)
    
    for dataset in dataset_summary.index:
        mae_mean = dataset_summary.loc[dataset, ('mae_improvement_pct', 'mean')]
        mae_std = dataset_summary.loc[dataset, ('mae_improvement_pct', 'std')]
        count = dataset_summary.loc[dataset, ('mae_improvement_pct', 'count')]
        r2_mean = dataset_summary.loc[dataset, ('r2_improvement', 'mean')]
        orig_feat = dataset_summary.loc[dataset, ('original_features', 'first')]
        final_feat = dataset_summary.loc[dataset, ('final_features', 'first')]
        reduction = dataset_summary.loc[dataset, ('feature_reduction_ratio', 'first')]
        print(f"  {dataset}: MAE {mae_mean:+.2f}±{mae_std:.2f}% (n={count}), R² {r2_mean:+.4f}")
        print(f"    Features: {orig_feat} → {final_feat} ({reduction:.2%} reduction)")
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "summary_report.txt")
    with open(summary_file, 'w') as f:
        f.write("Feature Enhancement Comparison Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Overall Statistics:\n")
        f.write(f"Total evaluations: {len(df)}\n")
        f.write(f"Successful evaluations: {len(successful_df)}\n")
        f.write(f"Success rate: {len(successful_df)/len(df)*100:.1f}%\n\n")
        
        f.write(f"MAE Improvements (Primary Metric):\n")
        f.write(f"Mean improvement: {mae_improvements.mean():+.2f}%\n")
        f.write(f"Median improvement: {mae_improvements.median():+.2f}%\n")
        f.write(f"Std improvement: {mae_improvements.std():.2f}%\n")
        f.write(f"Min improvement: {mae_improvements.min():+.2f}%\n")
        f.write(f"Max improvement: {mae_improvements.max():+.2f}%\n")
        f.write(f"Positive improvements: {(mae_improvements > 0).sum()}/{len(mae_improvements)} ({(mae_improvements > 0).mean()*100:.1f}%)\n\n")
        
        f.write(f"R² Improvements (Secondary):\n")
        f.write(f"Mean improvement: {r2_improvements.mean():+.4f}\n")
        f.write(f"Median improvement: {r2_improvements.median():+.4f}\n")
        f.write(f"Positive improvements: {(r2_improvements > 0).sum()}/{len(r2_improvements)} ({(r2_improvements > 0).mean()*100:.1f}%)\n\n")


def main():
    """Main function to run comparison analysis."""
    print("Feature Enhancement Comparison Analysis")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    output_dir = project_root / "comparison_results"
    output_dir.mkdir(exist_ok=True)
    
    # Define datasets to process
    datasets = [
        ("California.csv", "California", -1),
        ("Diabetes.csv", "Diabetes", -1),
    ]
    
    # Process all datasets
    all_results = []
    
    for dataset_file, dataset_name, target_col in datasets:
        dataset_path = data_dir / dataset_file
        
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_path} not found, skipping...")
            continue
        
        dataset_results = process_dataset(str(dataset_path), dataset_name, target_col)
        all_results.extend(dataset_results)
    
    if not all_results:
        print("No results to save.")
        return
    
    # Create DataFrame and save results
    df = pd.DataFrame(all_results)
    
    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"feature_enhancement_comparison_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_file}")
    
    # Generate and save summary report
    generate_summary_report(df, output_dir)
    
    # Save latest results (overwrite)
    latest_csv = output_dir / "latest_comparison_results.csv"
    df.to_csv(latest_csv, index=False)
    
    print(f"Latest results also saved to: {latest_csv}")
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()