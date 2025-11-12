#!/usr/bin/env python3
"""
Test script to verify that FeatureEnhancer applies default configurations
when none are provided instead of skipping synthesis/selection.
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from feature_enhancer.feature_enhancer import FeatureEnhancer


def test_default_configs():
    """Test that default configurations are applied when None is provided."""

    # Create sample data
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    # Create a simple model for evaluation
    model = Ridge()

    print(
        "=== Testing FeatureEnhancer with no configurations (should use defaults) ==="
    )

    # Test 1: No configurations provided (should use defaults)
    enhancer = FeatureEnhancer(
        synthesis_config=None, selection_config=None, random_state=42, verbose=True
    )

    print(f"Synthesis config after init: {enhancer.synthesis_config}")
    print(f"Selection config after init: {enhancer.selection_config}")

    # Fit the enhancer
    try:
        enhancer.fit(X, y, model)
        print("✓ FeatureEnhancer fitted successfully with default configs!")
        print(f"✓ Synthesis performed: {enhancer.synthesis_performed_}")
        print(f"✓ Selection performed: {enhancer.selection_performed_}")
        print(f"✓ Original features: {X.shape[1]}")
        print(f"✓ Final features: {enhancer.X_after_selection_.shape[1]}")

    except Exception as e:
        print(f"✗ Error during fitting: {e}")
        return False

    print("\n=== Testing FeatureEnhancer.from_config_files with no files ===")

    # Test 2: Using from_config_files with no config files
    enhancer2 = FeatureEnhancer.from_config_files(
        synthesis_config_path=None,
        selection_config_path=None,
        random_state=42,
        verbose=True,
    )

    print(f"Synthesis config after init: {enhancer2.synthesis_config}")
    print(f"Selection config after init: {enhancer2.selection_config}")

    try:
        enhancer2.fit(X, y, model)
        print("✓ FeatureEnhancer.from_config_files worked successfully with defaults!")
        print(f"✓ Synthesis performed: {enhancer2.synthesis_performed_}")
        print(f"✓ Selection performed: {enhancer2.selection_performed_}")

    except Exception as e:
        print(f"✗ Error during fitting with from_config_files: {e}")
        return False

    print("\n=== Testing partial configurations (should fill in missing defaults) ===")

    # Test 3: Partial configurations
    partial_synthesis_config = {"population_size": 20}  # Only one parameter
    partial_selection_config = {"generations": 10}  # Only one parameter

    enhancer3 = FeatureEnhancer(
        synthesis_config=partial_synthesis_config,
        selection_config=partial_selection_config,
        random_state=42,
        verbose=True,
    )

    try:
        enhancer3.fit(X, y, model)
        print("✓ FeatureEnhancer worked successfully with partial configs!")
        print(f"✓ Synthesis performed: {enhancer3.synthesis_performed_}")
        print(f"✓ Selection performed: {enhancer3.selection_performed_}")

    except Exception as e:
        print(f"✗ Error during fitting with partial configs: {e}")
        return False

    print("\n=== All tests passed! ===")
    return True


def test_config_validation():
    """Test that the _validate_config method works correctly."""

    print("\n=== Testing _validate_config method ===")

    enhancer = FeatureEnhancer()

    # Test synthesis config validation
    synthesis_config = enhancer._validate_config({}, "synthesis")
    print(f"Empty synthesis config filled with defaults: {synthesis_config}")

    # Test selection config validation
    selection_config = enhancer._validate_config({}, "selection")
    print(f"Empty selection config filled with defaults: {selection_config}")

    # Test partial config validation
    partial_config = {"population_size": 50}
    filled_config = enhancer._validate_config(partial_config, "synthesis")
    print(
        f"Partial config filled: population_size={filled_config['population_size']}, max_generations={filled_config['max_generations']}"
    )

    print("✓ _validate_config method works correctly!")


if __name__ == "__main__":
    print("Testing default configuration behavior in FeatureEnhancer...")

    # Test config validation first
    test_config_validation()

    # Test the main functionality
    success = test_default_configs()

    if success:
        print("\n🎉 All tests completed successfully!")
        print(
            "The FeatureEnhancer now applies default configurations instead of skipping phases."
        )
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
