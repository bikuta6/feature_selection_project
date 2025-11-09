#!/usr/bin/env python3
"""
Advanced Example: Using Custom Crossover and Mutation Operators

This script demonstrates how to use the new crossover_type and mutation_type
parameters in feature synthesis and selection configurations for different
scenarios and datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).parent.parent))

from main import run_feature_enhancement


def create_sample_dataset(dataset_type="regression", n_samples=200, n_features=15):
    """Create sample datasets for testing."""
    if dataset_type == "regression":
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=0.1,
            random_state=42,
        )
        return pd.DataFrame(
            X, columns=[f"feature_{i}" for i in range(n_features)]
        ), pd.Series(y, name="target")

    elif dataset_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_clusters_per_class=1,
            random_state=42,
        )
        return pd.DataFrame(
            X, columns=[f"feature_{i}" for i in range(n_features)]
        ), pd.Series(y, name="target")


def example_1_exploratory_synthesis():
    """
    Example 1: Exploratory Feature Synthesis

    Use case: Discovering novel feature combinations
    Strategy: High exploration with random crossover and diverse mutations
    """
    print("=" * 80)
    print("EXAMPLE 1: Exploratory Feature Synthesis")
    print("=" * 80)

    # Create regression dataset
    X_df, y_series = create_sample_dataset("regression", n_samples=150, n_features=8)

    # Save temporary CSV
    temp_data = pd.concat([X_df, y_series], axis=1)
    temp_data.to_csv("temp_exploratory.csv", index=False)

    # High exploration synthesis configuration
    synthesis_config = {
        "population_size": 50,
        "max_generations": 20,
        "crossover_prob": 0.7,
        "mutation_prob": 0.15,
        "max_depth": 5,
        "crossover_type": "random",  # Creates novel combinations
        "mutation_type": "random",  # Diverse mutation strategies
        "n_features_to_create": 3,
        "elitism": True,
    }

    # Conservative selection to focus on synthesis results
    selection_config = {
        "population_size": 30,
        "generations": 15,
        "crossover_type": "single_point",  # Simple, reliable crossover
        "mutation_type": "uniform",  # Minimal disruption
        "mutation_prob": 0.005,
        "secondary_objective": "sparsity",
    }

    print("Configuration Focus: Novel feature discovery through high exploration")
    print(
        f"Synthesis: {synthesis_config['crossover_type']} crossover + {synthesis_config['mutation_type']} mutation"
    )
    print(
        f"Selection: {selection_config['crossover_type']} crossover + {selection_config['mutation_type']} mutation"
    )

    results = run_feature_enhancement(
        "temp_exploratory.csv",
        target_column="target",
        synthesis_config=synthesis_config,
        selection_config=selection_config,
        model_name="ridge",
        verbose=False,
    )

    print(
        f"\nResults: {results['feature_info']['n_features_original']} → {results['feature_info']['n_features_final']} features"
    )
    print(f"Performance improvement: {results['improvement']:+.4f}")

    # Cleanup
    Path("temp_exploratory.csv").unlink()


def example_2_fine_tuning():
    """
    Example 2: Fine-tuning Existing Features

    Use case: Optimizing feature selection with minimal disruption
    Strategy: Precise operators that make small, controlled changes
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Fine-tuning Feature Selection")
    print("=" * 80)

    # Create classification dataset with many features
    X_df, y_series = create_sample_dataset(
        "classification", n_samples=200, n_features=20
    )

    # Save temporary CSV
    temp_data = pd.concat([X_df, y_series], axis=1)
    temp_data.to_csv("temp_finetuning.csv", index=False)

    # Minimal synthesis for feature refinement
    synthesis_config = {
        "population_size": 30,
        "max_generations": 15,
        "crossover_prob": 0.8,
        "mutation_prob": 0.05,
        "max_depth": 3,
        "crossover_type": "point",  # Precise, local changes
        "mutation_type": "parameter",  # Fine-tune constants and features
        "n_features_to_create": 1,
    }

    # Adaptive selection for balance
    selection_config = {
        "population_size": 40,
        "generations": 25,
        "crossover_type": "arithmetic",  # Smooth transitions
        "mutation_type": "adaptive",  # Self-regulating
        "mutation_prob": 0.02,
        "arithmetic_alpha": 0.6,
        "adaptive_min_prob": 0.005,
        "adaptive_max_prob": 0.04,
        "secondary_objective": "sparsity",
        "metric": "accuracy",
    }

    print("Configuration Focus: Fine-tuning with precise, controlled operators")
    print(
        f"Synthesis: {synthesis_config['crossover_type']} crossover + {synthesis_config['mutation_type']} mutation"
    )
    print(
        f"Selection: {selection_config['crossover_type']} crossover + {selection_config['mutation_type']} mutation"
    )

    results = run_feature_enhancement(
        "temp_finetuning.csv",
        target_column="target",
        synthesis_config=synthesis_config,
        selection_config=selection_config,
        model_name="logistic",
        verbose=False,
    )

    print(
        f"\nResults: {results['feature_info']['n_features_original']} → {results['feature_info']['n_features_final']} features"
    )
    print(f"Performance improvement: {results['improvement']:+.4f}")

    # Cleanup
    Path("temp_finetuning.csv").unlink()


def example_3_correlated_features():
    """
    Example 3: Handling Correlated Feature Groups

    Use case: Datasets with feature groups that should be selected together
    Strategy: Block-based operations to maintain feature relationships
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Correlated Feature Groups")
    print("=" * 80)

    # Create dataset with correlated feature groups
    np.random.seed(42)
    n_samples, n_groups, group_size = 150, 4, 3

    X_groups = []
    for i in range(n_groups):
        # Create base features
        base = np.random.randn(n_samples, group_size)
        # Add correlation within group
        corr_matrix = np.random.rand(group_size, group_size)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)

        group_features = base @ np.linalg.cholesky(corr_matrix)
        X_groups.append(group_features)

    X = np.hstack(X_groups)
    y = X[:, 0] + X[:, 3] + X[:, 6] + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    feature_names = [
        f"group_{i // group_size}_feature_{i % group_size}" for i in range(X.shape[1])
    ]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    # Save temporary CSV
    temp_data = pd.concat([X_df, y_series], axis=1)
    temp_data.to_csv("temp_correlated.csv", index=False)

    # Growth-oriented synthesis
    synthesis_config = {
        "population_size": 40,
        "max_generations": 18,
        "crossover_type": "subtree",  # Maintains expression structure
        "mutation_type": "grow",  # Builds complexity gradually
        "mutation_prob": 0.08,
        "max_depth": 4,
        "n_features_to_create": 2,
    }

    # Block-based selection for correlated features
    selection_config = {
        "population_size": 35,
        "generations": 20,
        "crossover_type": "two_point",  # Preserves segments
        "mutation_type": "block",  # Handles groups together
        "mutation_prob": 0.03,
        "block_size": group_size,  # Match actual group size
        "secondary_objective": "correlation",  # Consider feature relationships
    }

    print("Configuration Focus: Preserving feature group relationships")
    print(
        f"Synthesis: {synthesis_config['crossover_type']} crossover + {synthesis_config['mutation_type']} mutation"
    )
    print(
        f"Selection: {selection_config['crossover_type']} crossover + {selection_config['mutation_type']} mutation"
    )
    print(f"Block size: {selection_config['block_size']} (matches feature group size)")

    results = run_feature_enhancement(
        "temp_correlated.csv",
        target_column="target",
        synthesis_config=synthesis_config,
        selection_config=selection_config,
        model_name="ridge",
        verbose=False,
    )

    print(
        f"\nResults: {results['feature_info']['n_features_original']} → {results['feature_info']['n_features_final']} features"
    )
    print(f"Performance improvement: {results['improvement']:+.4f}")

    # Cleanup
    Path("temp_correlated.csv").unlink()


def example_4_high_dimensional():
    """
    Example 4: High-Dimensional Data

    Use case: Many features, need aggressive selection
    Strategy: High mixing crossover with adaptive mutation
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: High-Dimensional Data")
    print("=" * 80)

    # Create high-dimensional dataset
    X_df, y_series = create_sample_dataset("regression", n_samples=100, n_features=50)

    # Save temporary CSV
    temp_data = pd.concat([X_df, y_series], axis=1)
    temp_data.to_csv("temp_highdim.csv", index=False)

    # Conservative synthesis (avoid overfitting)
    synthesis_config = {
        "population_size": 25,
        "max_generations": 10,
        "crossover_type": "subtree",
        "mutation_type": "node",  # Conservative mutations
        "mutation_prob": 0.05,
        "max_depth": 3,  # Shallow trees
        "n_features_to_create": 1,
    }

    # Aggressive selection for dimensionality reduction
    selection_config = {
        "population_size": 60,  # Larger population for exploration
        "generations": 30,
        "crossover_type": "uniform",  # High mixing
        "mutation_type": "random_bit_flip",  # Standard approach
        "mutation_prob": 0.01,
        "uniform_swap_prob": 0.3,  # Moderate mixing
        "secondary_objective": "sparsity",  # Aggressive sparsity
    }

    print("Configuration Focus: Aggressive dimensionality reduction")
    print(
        f"Synthesis: {synthesis_config['crossover_type']} crossover + {synthesis_config['mutation_type']} mutation"
    )
    print(
        f"Selection: {selection_config['crossover_type']} crossover + {selection_config['mutation_type']} mutation"
    )
    print(f"Target: Reduce from {X_df.shape[1]} features significantly")

    results = run_feature_enhancement(
        "temp_highdim.csv",
        target_column="target",
        synthesis_config=synthesis_config,
        selection_config=selection_config,
        model_name="ridge",
        verbose=False,
    )

    reduction_ratio = 1 - (
        results["feature_info"]["n_features_final"]
        / results["feature_info"]["n_features_original"]
    )
    print(
        f"\nResults: {results['feature_info']['n_features_original']} → {results['feature_info']['n_features_final']} features"
    )
    print(f"Dimensionality reduction: {reduction_ratio:.1%}")
    print(f"Performance improvement: {results['improvement']:+.4f}")

    # Cleanup
    Path("temp_highdim.csv").unlink()


def demonstrate_operator_effects():
    """
    Demonstrate the effects of different operator choices on the same dataset.
    """
    print("\n" + "=" * 80)
    print("OPERATOR COMPARISON: Same Dataset, Different Strategies")
    print("=" * 80)

    # Create a standard dataset
    X_df, y_series = create_sample_dataset("regression", n_samples=120, n_features=10)
    temp_data = pd.concat([X_df, y_series], axis=1)
    temp_data.to_csv("temp_comparison.csv", index=False)

    # Define different strategies
    strategies = [
        {
            "name": "Conservative",
            "synthesis": {"crossover_type": "point", "mutation_type": "parameter"},
            "selection": {"crossover_type": "single_point", "mutation_type": "uniform"},
        },
        {
            "name": "Exploratory",
            "synthesis": {"crossover_type": "random", "mutation_type": "random"},
            "selection": {
                "crossover_type": "uniform",
                "mutation_type": "random_bit_flip",
            },
        },
        {
            "name": "Balanced",
            "synthesis": {"crossover_type": "subtree", "mutation_type": "subtree"},
            "selection": {"crossover_type": "two_point", "mutation_type": "adaptive"},
        },
    ]

    results_comparison = []

    for strategy in strategies:
        print(f"\nTesting {strategy['name']} Strategy:")

        synthesis_config = {
            "population_size": 30,
            "max_generations": 15,
            "mutation_prob": 0.1,
            "max_depth": 4,
            **strategy["synthesis"],
        }

        selection_config = {
            "population_size": 30,
            "generations": 15,
            "mutation_prob": 0.02,
            "secondary_objective": "sparsity",
            **strategy["selection"],
        }

        try:
            results = run_feature_enhancement(
                "temp_comparison.csv",
                target_column="target",
                synthesis_config=synthesis_config,
                selection_config=selection_config,
                model_name="ridge",
                verbose=False,
                random_state=42,  # Same seed for fair comparison
            )

            results_comparison.append(
                {
                    "strategy": strategy["name"],
                    "improvement": results["improvement"],
                    "final_features": results["feature_info"]["n_features_final"],
                    "synthesis_ops": f"{strategy['synthesis']['crossover_type']}/{strategy['synthesis']['mutation_type']}",
                    "selection_ops": f"{strategy['selection']['crossover_type']}/{strategy['selection']['mutation_type']}",
                }
            )

            print(f"  Improvement: {results['improvement']:+.4f}")
            print(
                f"  Features: {results['feature_info']['n_features_original']} → {results['feature_info']['n_features_final']}"
            )

        except Exception as e:
            print(f"  Failed: {str(e)}")

    # Summary comparison
    print(f"\nCOMPARISON SUMMARY:")
    print(f"{'Strategy':<12} {'Operators':<25} {'Features':<10} {'Improvement':<12}")
    print("-" * 65)

    for result in results_comparison:
        ops = f"{result['synthesis_ops'][:8]}/{result['selection_ops'][:8]}"
        print(
            f"{result['strategy']:<12} {ops:<25} {result['final_features']:<10} {result['improvement']:+.4f}"
        )

    # Cleanup
    Path("temp_comparison.csv").unlink()


def main():
    """Run all examples to demonstrate operator configuration capabilities."""
    print("Advanced Feature Enhancement: Custom Operator Examples")
    print("This script demonstrates how different operator combinations")
    print("can be used for various data science scenarios.\n")

    try:
        example_1_exploratory_synthesis()
        example_2_fine_tuning()
        example_3_correlated_features()
        example_4_high_dimensional()
        demonstrate_operator_effects()

        print("\n" + "=" * 80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Different operators serve different purposes")
        print("2. Synthesis and selection operators can be independently configured")
        print("3. Problem characteristics should guide operator choice")
        print("4. Multiple strategies can be tested for optimal results")
        print(
            "\nRefer to config_examples/OPERATORS_GUIDE.md for detailed documentation."
        )

    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
