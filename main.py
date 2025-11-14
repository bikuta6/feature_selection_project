"""
Main script for Feature Enhancement using CSV datasets.

This script demonstrates how to:
1. Load CSV datasets
2. Preprocess the data
3. Apply feature enhancement (synthesis + selection)
4. Evaluate results
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")

# Add the project to path
sys.path.append(str(Path(__file__).parent))

from feature_enhancer import DatasetLoader, FeatureEnhancer, print_dataset_summary


def get_model_for_task(task_type: str, model_name: str = "auto"):
    """Get appropriate model for the task type."""
    if task_type == "regression":
        if model_name == "auto" or model_name == "ridge":
            return Ridge(random_state=42)
        elif model_name == "rf":
            return RandomForestRegressor(
                n_estimators=10, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_name == "linear":
            return LinearRegression(n_jobs=-1)
        elif model_name == "lasso":
            return Lasso(random_state=42)
        elif model_name == "knn":
            return KNeighborsRegressor(n_jobs=-1)
        elif model_name == "svm":
            return SVR()
        elif model_name == "dt":
            return DecisionTreeRegressor(random_state=42)
        elif model_name == "gb":
            return GradientBoostingRegressor(random_state=42)
        elif model_name == "mlp":
            return MLPRegressor(random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    elif task_type == "classification":
        if model_name == "auto" or model_name == "logistic":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == "rf":
            return RandomForestClassifier(
                n_estimators=10, max_depth=10, random_state=42, n_jobs=-1
            )
        elif model_name == "knn":
            return KNeighborsClassifier(n_jobs=-1)
        elif model_name == "svm":
            return SVC(random_state=42)
        elif model_name == "dt":
            return DecisionTreeClassifier(random_state=42)
        elif model_name == "gb":
            return GradientBoostingClassifier(random_state=42)
        elif model_name == "mlp":
            return MLPClassifier(random_state=42, max_iter=500)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")

    else:
        raise ValueError(f"Unknown task type: {task_type}")


def evaluate_model(model, X_train, X_test, y_train, y_test, task_type):
    """Evaluate model performance."""
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    if task_type == "regression":
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"    MAE: {mae:.4f}")
        print(f"    R²:  {r2:.4f}")

        return {"mae": mae, "r2": r2}

    else:  # classification
        accuracy = accuracy_score(y_test, y_pred)

        print(f"    Accuracy: {accuracy:.4f}")

        # Detailed classification report for small number of classes
        if len(np.unique(y_test)) <= 10:
            print(f"    Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"      Precision (avg): {report['macro avg']['precision']:.4f}")
            print(f"      Recall (avg):    {report['macro avg']['recall']:.4f}")
            print(f"      F1-score (avg):  {report['macro avg']['f1-score']:.4f}")

        return {"accuracy": accuracy}


def run_feature_enhancement(
    csv_path: str,
    target_column: str = -1,
    synthesis_config: dict = None,
    selection_config: dict = None,
    model_name: str = "auto",
    test_size: float = 0.2,
    scale_features: bool = True,
    random_state: int = 42,
    verbose: bool = True,
    use_multiprocessing: bool = False,
    n_jobs: int = 1,
):
    """
    Complete feature enhancement pipeline for CSV datasets using cross-validation.

    Args:
        csv_path: Path to CSV file
        target_column: Target column name or index
        synthesis_config: Configuration for feature synthesis
        selection_config: Configuration for feature selection
        model_name: Model to use ('auto', 'linear', 'logistic', 'rf')
        test_size: Proportion for test set
        scale_features: Whether to scale features
        random_state: Random seed
        verbose: Print detailed information
        use_multiprocessing: Whether to use multiprocessing for fitness calculations
        n_jobs: Number of processes to use (default: 1, -1 uses all cores)
    """

    print(f"=== Feature Enhancement Pipeline ===")
    print(f"Dataset: {csv_path}")
    print(f"Target: {target_column}")
    print(f"Model: {model_name}")
    print(f"Scaling: {scale_features}")

    # Step 1: Load dataset
    print(f"\n--- Step 1: Loading Dataset ---")
    try:
        X, y = DatasetLoader.load_csv(csv_path, target_column=target_column)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Step 2: Get dataset information
    print(f"\n--- Step 2: Dataset Analysis ---")
    dataset_info = DatasetLoader.get_dataset_info(X, y)
    print_dataset_summary(dataset_info)

    # Step 3: Preprocessing
    print(f"\n--- Step 3: Preprocessing ---")
    X_processed, y_processed = DatasetLoader.preprocess_dataset(
        X, y, handle_missing="drop", encode_categorical=True, target_type="auto"
    )

    # Determine task type
    task_type = "classification" if y_processed.nunique() < 5 else "regression"
    print(f"Task type: {task_type}")

    # Step 4: Data splitting
    print(f"\n--- Step 4: Data Splitting ---")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_processed,
        y_processed,
        test_size=test_size,
        random_state=random_state,
        stratify=y_processed if task_type == "classification" else None,
    )

    print(f"Train+Val: {X_train_full.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    # Get model
    model = get_model_for_task(task_type, model_name)

    # Step 5: Baseline evaluation using cross-validation
    print(f"\n--- Step 5: Baseline Evaluation ---")
    print(f"Evaluating with original {X_train_full.shape[1]} features...")

    baseline_results = evaluate_model(
        model, X_train_full, X_test, y_train_full, y_test, task_type
    )

    # Step 6: Feature Enhancement
    print(f"\n--- Step 6: Feature Enhancement ---")

    # Create enhancer
    enhancer = FeatureEnhancer(
        synthesis_config=synthesis_config,
        selection_config=selection_config,
        scale_features=scale_features,
        random_state=random_state,
        verbose=verbose,
        use_multiprocessing=use_multiprocessing,
        n_jobs=n_jobs,
    )

    # Apply enhancement using cross-validation
    print(f"Applying feature enhancement...")
    X_enhanced = enhancer.fit_transform(X_train_full, y_train_full, model)

    # Transform test set
    X_test_enhanced = enhancer.transform(X_test)

    # Step 7: Enhanced evaluation
    print(f"\n--- Step 7: Enhanced Evaluation ---")
    print(f"Evaluating with enhanced {X_enhanced.shape[1]} features...")

    enhanced_results = evaluate_model(
        model, X_enhanced, X_test_enhanced, y_train_full, y_test, task_type
    )

    # Step 8: Results summary
    print(f"\n--- Step 8: Results Summary ---")

    feature_info = enhancer.get_feature_info()

    print(f"Enhancement Summary:")
    print(f"  Original features: {feature_info['n_features_original']}")
    print(f"  Final features: {feature_info['n_features_final']}")
    print(f"  Scaling performed: {feature_info['scaling_performed']}")
    print(f"  Synthesis performed: {feature_info['synthesis_performed']}")
    print(f"  Selection performed: {feature_info['selection_performed']}")
    print(
        f"  Reduction ratio: {feature_info['summary']['feature_reduction_ratio']:.2%}"
    )

    if feature_info["synthesis_performed"]:
        synth_info = enhancer.get_synthesized_features_info()
        print(f"  Synthesized features: {len(synth_info)}")
        for info in synth_info:
            if info["selected"]:
                print(f"    - {info['name']}: {info['expression'][:50]}...")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    if task_type == "regression":
        baseline_metric = baseline_results["r2"]
        enhanced_metric = enhanced_results["r2"]
        metric_name = "R²"
        improvement = enhanced_metric - baseline_metric
        secondary_baseline = baseline_results["mae"]
        secondary_enhanced = enhanced_results["mae"]
        secondary_improvement = secondary_baseline - secondary_enhanced
    else:
        baseline_metric = baseline_results["accuracy"]
        enhanced_metric = enhanced_results["accuracy"]
        metric_name = "Accuracy"
        improvement = enhanced_metric - baseline_metric

    print(f"  Baseline {metric_name}: {baseline_metric:.4f}")
    print(f"  Enhanced {metric_name}: {enhanced_metric:.4f}")
    print(
        f"  Improvement: {improvement:+.4f} ({improvement / baseline_metric * 100:+.2f}%)"
    )
    if task_type == "regression":
        print(f"  Baseline MAE: {secondary_baseline:.4f}")
        print(f"  Enhanced MAE: {secondary_enhanced:.4f}")
        print(
            f"  MAE Improvement: {secondary_improvement:+.4f} ({secondary_improvement / secondary_baseline * 100:+.2f}%)"
        )

    # Feature details
    if verbose:
        print(f"\nSelected Features Detail:")
        summary_df = enhancer.get_selected_features_summary()
        for _, row in summary_df.iterrows():
            if row["origin_type"] == "original":
                print(f"  {row['final_index']}: {row['original_name']} (original)")
            else:
                print(
                    f"  {row['final_index']}: {row['expression'][:60]}... (synthesized)"
                )

    return {
        "enhancer": enhancer,
        "baseline_results": baseline_results,
        "enhanced_results": enhanced_results,
        "feature_info": feature_info,
        "improvement": improvement,
        "X_test_enhanced": X_test_enhanced,
        "y_test": y_test,
    }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Feature Enhancement for CSV datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument("--csv-path", help="Path to CSV dataset")
    parser.add_argument(
        "--target",
        "-t",
        default=-1,
        help="Target column name or index (default: last column)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        "-m",
        default="auto",
        choices=[
            "auto",
            "linear",
            "logistic",
            "rf",
            "ridge",
            "lasso",
            "knn",
            "svm",
            "dt",
            "gb",
            "mlp",
        ],
        help="Model to use for evaluation",
    )

    # Enhancement arguments
    parser.add_argument("--synthesis-config", help="Path to synthesis config JSON file")
    parser.add_argument("--selection-config", help="Path to selection config JSON file")
    parser.add_argument(
        "--no-scale", action="store_true", help="Disable feature scaling"
    )
    parser.add_argument(
        "--use-multiprocessing",
        action="store_true",
        help="Enable multiprocessing for fitness calculations in feature synthesis",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of processes to use for multiprocessing (default: 1, -1 uses all cores)",
    )

    # Experimental setup
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set proportion"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    args = parser.parse_args()

    # Validate file path
    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' not found")
        return 1

    # Load config files
    synthesis_config = {}
    selection_config = {}

    if args.synthesis_config:
        if not os.path.exists(args.synthesis_config):
            print(f"Error: Synthesis config file '{args.synthesis_config}' not found")
            return 1
        with open(args.synthesis_config, "r") as f:
            synthesis_config = json.load(f)

    if args.selection_config:
        if not os.path.exists(args.selection_config):
            print(f"Error: Selection config file '{args.selection_config}' not found")
            return 1
        with open(args.selection_config, "r") as f:
            selection_config = json.load(f)

    # Run enhancement
    try:
        results = run_feature_enhancement(
            csv_path=args.csv_path,
            target_column=args.target,
            synthesis_config=synthesis_config,
            selection_config=selection_config,
            model_name=args.model,
            test_size=args.test_size,
            scale_features=not args.no_scale,
            random_state=args.random_state,
            verbose=not args.quiet,
            use_multiprocessing=args.use_multiprocessing,
            n_jobs=args.n_jobs,
        )

        if results is None:
            return 1

        print(f"\nFeature enhancement completed successfully!")

        return 0

    except Exception as e:
        print(f"\n❌ Error during feature enhancement: {e}")
        import traceback

        if not args.quiet:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
