"""
Visualization script for Feature Enhancement Comparison Results.

This script creates visualizations from the comparison analysis CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

def load_latest_results():
    """Load the latest comparison results."""
    results_dir = Path("comparison_results")
    csv_file = results_dir / "latest_comparison_results.csv"
    
    if not csv_file.exists():
        print(f"Results file not found: {csv_file}")
        print("Please run the comparison analysis first.")
        return None
    
    df = pd.read_csv(csv_file)
    return df

def create_visualizations(df, output_dir):
    """Create various visualizations from the results."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter successful results
    df_success = df[(df['baseline_success'] == True) & (df['enhanced_success'] == True)]
    
    if len(df_success) == 0:
        print("No successful results to visualize.")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. MAE Improvement by Model (Primary)
    plt.subplot(3, 3, 1)
    model_mae = df_success.groupby('model')['mae_improvement_pct'].agg(['mean', 'std']).reset_index()
    model_mae = model_mae.sort_values('mean', ascending=True)
    
    bars = plt.barh(range(len(model_mae)), model_mae['mean'], 
                   xerr=model_mae['std'], capsize=5, alpha=0.8)
    plt.yticks(range(len(model_mae)), model_mae['model'])
    plt.xlabel('Mean MAE Improvement (%)')
    plt.title('MAE Improvement by Model (Primary Metric)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # Color bars based on positive/negative improvement
    for i, bar in enumerate(bars):
        if model_mae.iloc[i]['mean'] >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 2. MAE Improvement by Dataset (Primary)
    plt.subplot(3, 3, 2)
    dataset_mae = df_success.groupby('dataset')['mae_improvement_pct'].agg(['mean', 'std']).reset_index()
    dataset_mae = dataset_mae.sort_values('mean', ascending=True)
    
    bars = plt.barh(range(len(dataset_mae)), dataset_mae['mean'], 
                   xerr=dataset_mae['std'], capsize=5, alpha=0.8)
    plt.yticks(range(len(dataset_mae)), dataset_mae['dataset'])
    plt.xlabel('Mean MAE Improvement (%)')
    plt.title('MAE Improvement by Dataset (Primary Metric)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # Color bars
    for i, bar in enumerate(bars):
        if dataset_mae.iloc[i]['mean'] >= 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')
    
    # 3. MAE Improvements Heatmap
    plt.subplot(3, 3, 3)
    pivot_mae = df_success.pivot(index='model', columns='dataset', values='mae_improvement_pct')
    sns.heatmap(pivot_mae, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'MAE Improvement (%)'})
    plt.title('MAE Improvement Heatmap (Primary Metric)')
    plt.ylabel('Model')
    plt.xlabel('Dataset')
    
    # 4. R² Improvement by Model (Secondary)
    plt.subplot(3, 3, 4)
    model_r2 = df_success.groupby('model')['r2_improvement'].agg(['mean', 'std']).reset_index()
    model_r2 = model_r2.sort_values('mean', ascending=True)
    
    bars = plt.barh(range(len(model_r2)), model_r2['mean'], 
                   xerr=model_r2['std'], capsize=5, alpha=0.8, color='orange')
    plt.yticks(range(len(model_r2)), model_r2['model'])
    plt.xlabel('Mean R² Improvement')
    plt.title('R² Improvement by Model (Secondary Metric)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # 5. Feature Reduction vs MAE Performance
    plt.subplot(3, 3, 5)
    plt.scatter(df_success['feature_reduction_ratio'], df_success['mae_improvement_pct'],
               alpha=0.6, s=60)
    plt.xlabel('Feature Reduction Ratio')
    plt.ylabel('MAE Improvement (%)')
    plt.title('Feature Reduction vs MAE Improvement')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(alpha=0.3)
    
    # Add dataset labels
    for _, row in df_success.iterrows():
        plt.annotate(f"{row['dataset'][:3]}-{row['model'][:3]}", 
                    (row['feature_reduction_ratio'], row['mae_improvement_pct']),
                    fontsize=8, alpha=0.7)
    
    # 6. Training Time Comparison
    plt.subplot(3, 3, 6)
    time_comparison = df_success.groupby('model')[['baseline_train_time', 'enhanced_train_time']].mean()
    
    x = np.arange(len(time_comparison))
    width = 0.35
    
    plt.bar(x - width/2, time_comparison['baseline_train_time'], width, 
           label='Baseline', alpha=0.8)
    plt.bar(x + width/2, time_comparison['enhanced_train_time'], width, 
           label='Enhanced', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Mean Training Time (s)')
    plt.title('Training Time Comparison')
    plt.xticks(x, time_comparison.index, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 7. Distribution of MAE Improvements
    plt.subplot(3, 3, 7)
    plt.hist(df_success['mae_improvement_pct'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('MAE Improvement (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of MAE Improvements (Primary Metric)')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    plt.axvline(x=df_success['mae_improvement_pct'].mean(), color='green', 
               linestyle='--', alpha=0.8, label=f'Mean: {df_success["mae_improvement_pct"].mean():.2f}%')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 8. Baseline vs Enhanced MAE Scatter
    plt.subplot(3, 3, 8)
    plt.scatter(df_success['baseline_mae'], df_success['enhanced_mae'], 
               alpha=0.6, s=60, c=df_success['mae_improvement_pct'], cmap='RdYlGn')
    
    # Diagonal line (no improvement)
    min_mae = min(df_success['baseline_mae'].min(), df_success['enhanced_mae'].min())
    max_mae = max(df_success['baseline_mae'].max(), df_success['enhanced_mae'].max())
    plt.plot([min_mae, max_mae], [min_mae, max_mae], 'r--', alpha=0.5, label='No improvement')
    
    plt.xlabel('Baseline MAE')
    plt.ylabel('Enhanced MAE')
    plt.title('Baseline vs Enhanced MAE (Primary Metric)')
    plt.legend()
    plt.colorbar(label='MAE Improvement (%)')
    plt.grid(alpha=0.3)
    
    # 9. Success Rate Summary (MAE Priority)
    plt.subplot(3, 3, 9)
    success_stats = {
        'Positive MAE Improvement': (df_success['mae_improvement_pct'] > 0).sum(),
        'Negative MAE Improvement': (df_success['mae_improvement_pct'] <= 0).sum(),
        'Positive R² Improvement': (df_success['r2_improvement'] > 0).sum(),
        'Negative R² Improvement': (df_success['r2_improvement'] <= 0).sum()
    }
    
    categories = ['MAE Improvements\n(Primary)', 'R² Improvements\n(Secondary)']
    positive = [success_stats['Positive MAE Improvement'], success_stats['Positive R² Improvement']]
    negative = [success_stats['Negative MAE Improvement'], success_stats['Negative R² Improvement']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, positive, width, label='Positive', color='green', alpha=0.8)
    plt.bar(x + width/2, negative, width, label='Negative', color='red', alpha=0.8)
    
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.title('Improvement Success Rates (MAE Priority)')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (pos, neg) in enumerate(zip(positive, negative)):
        plt.text(i - width/2, pos + 0.5, str(pos), ha='center')
        plt.text(i + width/2, neg + 0.5, str(neg), ha='center')
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = output_dir / "comparison_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    plt.show()

def print_top_improvements(df):
    """Print top improvements in different categories."""
    df_success = df[(df['baseline_success'] == True) & (df['enhanced_success'] == True)]
    
    print("\n" + "="*60)
    print("TOP IMPROVEMENTS (MAE Priority)")
    print("="*60)
    
    # Top MAE improvements (Primary metric)
    print("\nTop 10 MAE Improvements (Primary Metric):")
    top_mae = df_success.nlargest(10, 'mae_improvement_pct')[
        ['dataset', 'model', 'baseline_mae', 'enhanced_mae', 'mae_improvement_pct']
    ]
    
    for i, row in top_mae.iterrows():
        absolute_improvement = row['baseline_mae'] - row['enhanced_mae']
        print(f"  {row['dataset']:10} + {row['model']:15}: "
              f"{row['baseline_mae']:.4f} → {row['enhanced_mae']:.4f} "
              f"({row['mae_improvement_pct']:+.2f}%, {absolute_improvement:+.4f})")
    
    # Top R² improvements (Secondary metric)
    print("\nTop 10 R² Improvements (Secondary Metric):")
    top_r2 = df_success.nlargest(10, 'r2_improvement')[
        ['dataset', 'model', 'baseline_r2', 'enhanced_r2', 'r2_improvement']
    ]
    
    for i, row in top_r2.iterrows():
        improvement_pct = (row['r2_improvement'] / row['baseline_r2'] * 100) if row['baseline_r2'] != 0 else 0
        print(f"  {row['dataset']:10} + {row['model']:15}: "
              f"{row['baseline_r2']:.4f} → {row['enhanced_r2']:.4f} "
              f"({row['r2_improvement']:+.4f}, {improvement_pct:+.1f}%)")

def main():
    """Main function to create visualizations."""
    print("Feature Enhancement Results Visualization")
    print("="*50)
    
    # Load results
    df = load_latest_results()
    if df is None:
        return
    
    print(f"Loaded {len(df)} results")
    
    # Create output directory
    output_dir = Path("comparison_results")
    
    # Print summary statistics
    df_success = df[(df['baseline_success'] == True) & (df['enhanced_success'] == True)]
    print(f"Successful evaluations: {len(df_success)}/{len(df)} ({len(df_success)/len(df)*100:.1f}%)")
    print(f"Mean MAE improvement: {df_success['mae_improvement_pct'].mean():+.2f}%")
    print(f"Mean R² improvement: {df_success['r2_improvement'].mean():+.4f}")
    
    # Print top improvements
    print_top_improvements(df)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    try:
        create_visualizations(df_success, output_dir)
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()