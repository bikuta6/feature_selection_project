import copy
import json
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from .feature_selection.crossover import (
    ArithmeticCrossover,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
)
from .feature_selection.feature_selector import FeatureSelector
from .feature_selection.mutation import (
    AdaptiveMutation,
    BlockMutation,
    RandomBitFlip,
    UniformMutation,
)
from .feature_synthesis.crossover import (
    PointCrossover,
    RandomCrossover,
    SubtreeCrossover,
)
from .feature_synthesis.feature_synthesis import MultiFeatureGA, SimpleGA
from .feature_synthesis.mutation import (
    GrowMutation,
    NodeMutation,
    ParameterMutation,
    RandomMutation,
    SubtreeMutation,
)


class FeatureEnhancer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature enhancement class that combines feature synthesis and selection.

    This class follows sklearn's transformer interface and can:
    1. Synthesize new features using genetic programming (with default config if none provided)
    2. Select the best features using NSGA-II optimization (with default config if none provided)

    The process is configurable via config dictionaries. If no configuration is provided
    for synthesis or selection, default configurations are applied automatically.
    """

    def __init__(
        self,
        synthesis_config: Optional[Dict[str, Any]] = None,
        selection_config: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        verbose: bool = True,
        scale_features: bool = False,
        use_multiprocessing: bool = False,
        n_jobs: int = 1,
    ):
        """
        Initialize the FeatureEnhancer.

        Args:
            synthesis_config: Configuration for feature synthesis. If None, default config is used.
            selection_config: Configuration for feature selection. If None, default config is used.
            random_state: Random seed for reproducibility.
            verbose: Whether to print progress information.
            scale_features: Whether to apply standard scaling to input features before processing.
            use_multiprocessing: Whether to use multiprocessing for fitness calculations in synthesis.
            n_jobs: Number of processes to use for multiprocessing (default: 1, -1 uses all available cores).
        """
        # Apply default configurations when None is provided
        if synthesis_config is None:
            synthesis_config = {}
        if selection_config is None:
            selection_config = {}

        self.synthesis_config = synthesis_config
        self.selection_config = selection_config
        self.random_state = random_state
        self.verbose = verbose
        self.scale_features = scale_features
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs

        # Internal state
        self.synthesis_engine_ = None
        self.selection_engine_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.feature_names_in_ = None

        # Results tracking
        self.synthesized_features_ = []
        self.synthesized_feature_names_ = []
        self.original_features_ = []
        self.selected_features_ = []
        self.selected_feature_names_ = []
        self.feature_origin_map_ = {}  # Maps final feature index to origin (original/synthesized)
        self.synthesis_performed_ = False
        self.selection_performed_ = False

        # Feature matrices at different stages
        self.X_after_scaling_ = None
        self.X_after_synthesis_ = None
        self.X_after_selection_ = None

    def _validate_config(
        self, config: Dict[str, Any], config_type: str
    ) -> Dict[str, Any]:
        """Validate and set defaults for configuration."""

        config = copy.deepcopy(config)

        if config_type == "synthesis":
            defaults = {
                "population_size": 100,
                "max_generations": 50,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1,
                "tournament_size": 10,
                "max_depth": 30,
                "elitism": False,
                "n_features_to_create": 5,
                "use_multi_feature": True,
                "crossover_type": "subtree",
                "mutation_type": "subtree",
            }
        elif config_type == "selection":
            defaults = {
                "secondary_objective": "sparsity",
                "population_size": 100,
                "generations": 50,
                "crossover_prob": 0.8,
                "mutation_prob": 0.1,
                "objective_weights": [0.9, 0.1],
                "metric": "mae",
                "normalize": False,
                "crossover_type": "uniform",
                "mutation_type": "random_bit_flip",
            }
        else:
            raise ValueError(f"Unknown config type: {config_type}")

        # Fill in defaults for missing keys
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        return config

    def _setup_synthesis_engine(self, config: Dict[str, Any]):
        """Setup the feature synthesis engine based on config."""
        # Override multiprocessing settings if specified in config
        use_multiprocessing = config.get(
            "use_multiprocessing", self.use_multiprocessing
        )
        n_jobs = config.get("n_jobs", self.n_jobs)

        # Get crossover and mutation operators
        crossover_operator = self._get_synthesis_crossover_operator(config)
        mutation_operator = self._get_synthesis_mutation_operator(config)

        if config.get("use_multi_feature", True):
            self.synthesis_engine_ = MultiFeatureGA(
                n_features_to_create=config.get("n_features_to_create", 5),
                population_size=config.get("population_size", 100),
                max_generations=config.get("max_generations", 50),
                crossover_prob=config.get("crossover_prob", 0.8),
                mutation_prob=config.get("mutation_prob", 0.1),
                tournament_size=config.get("tournament_size", 3),
                max_depth=config.get("max_depth", 20),
                crossover_operator=crossover_operator,
                mutation_operator=mutation_operator,
                elitism=config.get("elitism", True),
                verbose=self.verbose,
                use_multiprocessing=use_multiprocessing,
                n_jobs=n_jobs,
            )
        else:
            self.synthesis_engine_ = SimpleGA(
                population_size=config.get("population_size", 100),
                max_generations=config.get("max_generations", 50),
                crossover_prob=config.get("crossover_prob", 0.8),
                mutation_prob=config.get("mutation_prob", 0.1),
                tournament_size=config.get("tournament_size", 3),
                max_depth=config.get("max_depth", 20),
                crossover_operator=crossover_operator,
                mutation_operator=mutation_operator,
                elitism=config.get("elitism", True),
                verbose=self.verbose,
                use_multiprocessing=use_multiprocessing,
                n_jobs=n_jobs,
            )

    def _setup_selection_engine(self, config: Dict[str, Any], model):
        """Setup the feature selection engine based on config."""
        # Get crossover and mutation operators
        crossover_operator = self._get_selection_crossover_operator(config)
        mutation_operator = self._get_selection_mutation_operator(config)

        self.selection_engine_ = FeatureSelector(
            model=model,
            secondary_objective=config.get("secondary_objective", "sparsity"),
            population_size=config.get("population_size", 100),
            generations=config.get("generations", 50),
            crossover_prob=config.get("crossover_prob", 0.8),
            mutation_prob=config.get("mutation_prob", 0.1),
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            objective_weights=config.get("objective_weights", None),
            metric=config.get("metric", "mae"),
            normalize=config.get("normalize", False),
            random_state=self.random_state,
        )

    def _get_synthesis_crossover_operator(self, config: Dict[str, Any]):
        """Get crossover operator for feature synthesis."""
        crossover_type = config.get("crossover_type", "subtree")
        max_depth = config.get("max_depth", 20)

        if crossover_type == "subtree":
            return SubtreeCrossover(max_depth=max_depth)
        elif crossover_type == "random":
            return RandomCrossover(max_depth=max_depth)
        elif crossover_type == "point":
            return PointCrossover()
        else:
            raise ValueError(
                f"Unknown synthesis crossover type: {crossover_type}. "
                f"Available types: 'subtree', 'random', 'point'"
            )

    def _get_synthesis_mutation_operator(self, config: Dict[str, Any]):
        """Get mutation operator for feature synthesis."""
        mutation_type = config.get("mutation_type", "subtree")
        mutation_prob = config.get("mutation_prob", 0.1)
        max_depth = config.get("max_depth", 20)

        if mutation_type == "subtree":
            return SubtreeMutation(mutation_prob, max_depth=max_depth // 2)
        elif mutation_type == "node":
            return NodeMutation(mutation_prob)
        elif mutation_type == "random":
            return RandomMutation(mutation_prob, max_depth=max_depth // 2)
        elif mutation_type == "parameter":
            return ParameterMutation(mutation_prob)
        elif mutation_type == "grow":
            return GrowMutation(mutation_prob, max_depth=max_depth // 2)
        else:
            raise ValueError(
                f"Unknown synthesis mutation type: {mutation_type}. "
                f"Available types: 'subtree', 'node', 'random', 'parameter', 'grow'"
            )

    def _get_selection_crossover_operator(self, config: Dict[str, Any]):
        """Get crossover operator for feature selection."""
        crossover_type = config.get("crossover_type", "uniform")

        if crossover_type == "single_point":
            return SinglePointCrossover()
        elif crossover_type == "two_point":
            return TwoPointCrossover()
        elif crossover_type == "uniform":
            swap_prob = config.get("uniform_swap_prob", 0.5)
            return UniformCrossover(swap_probability=swap_prob)
        elif crossover_type == "arithmetic":
            alpha = config.get("arithmetic_alpha", 0.5)
            return ArithmeticCrossover(alpha=alpha)
        else:
            raise ValueError(
                f"Unknown selection crossover type: {crossover_type}. "
                f"Available types: 'single_point', 'two_point', 'uniform', 'arithmetic'"
            )

    def _get_selection_mutation_operator(self, config: Dict[str, Any]):
        """Get mutation operator for feature selection."""
        mutation_type = config.get("mutation_type", "random_bit_flip")
        mutation_prob = config.get("mutation_prob", 0.01)

        if mutation_type == "random_bit_flip":
            return RandomBitFlip(mutation_prob)
        elif mutation_type == "uniform":
            return UniformMutation(mutation_prob)
        elif mutation_type == "block":
            block_size = config.get("block_size", 3)
            return BlockMutation(mutation_prob, block_size=block_size)
        elif mutation_type == "adaptive":
            min_prob = config.get("adaptive_min_prob", 0.01)
            max_prob = config.get("adaptive_max_prob", 0.1)
            return AdaptiveMutation(mutation_prob, min_prob=min_prob, max_prob=max_prob)
        else:
            raise ValueError(
                f"Unknown selection mutation type: {mutation_type}. "
                f"Available types: 'random_bit_flip', 'uniform', 'block', 'adaptive'"
            )

    def fit(self, X, y, model):
        """
        Fit the feature enhancer using cross-validation.

        Args:
            X: Input features (array-like or pandas DataFrame)
            y: Target values (array-like)
            model: ML model to use for evaluation (must have fit/predict methods)

        Returns:
            self: Fitted instance
        """
        # Convert inputs to numpy arrays and handle feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X = X.values
        else:
            X = np.asarray(X)
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        y = np.asarray(y)
        self.n_features_in_ = X.shape[1]

        # Store original feature information
        self.original_features_ = list(range(self.n_features_in_))

        if self.verbose:
            print(
                f"FeatureEnhancer: Starting with {self.n_features_in_} original features"
            )

        # Validate configurations
        synthesis_config = self._validate_config(self.synthesis_config, "synthesis")
        selection_config = self._validate_config(self.selection_config, "selection")

        # Set random state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize current_X
        current_X = X

        # Phase 0: Feature Scaling (if configured)
        if self.scale_features:
            if self.verbose:
                print("FeatureEnhancer: Applying standard scaling to features...")

            self.scaler_ = StandardScaler()
            current_X = self.scaler_.fit_transform(current_X)

            if self.verbose:
                print("FeatureEnhancer: Features scaled successfully")
        else:
            if self.verbose:
                print("FeatureEnhancer: Skipping feature scaling")

        self.X_after_scaling_ = current_X

        # Phase 1: Feature Synthesis (if configured)
        if self.verbose:
            print("FeatureEnhancer: Starting feature synthesis phase...")

        self._setup_synthesis_engine(synthesis_config)

        if synthesis_config.get("use_multi_feature", False):
            # Multi-feature synthesis (use cross-validation)
            cv_folds = synthesis_config.get("cv", 3)
            best_features = self.synthesis_engine_.evolve_multiple_features(
                current_X, y, cv=cv_folds, predictor_model=model
            )
            self.synthesized_features_ = best_features
            current_X = self.synthesis_engine_.create_multi_enhanced_dataset(current_X)
        else:
            # Single feature synthesis (use cross-validation)
            cv_folds = synthesis_config.get("cv", 3)
            best_feature = self.synthesis_engine_.evolve(
                current_X, y, cv=cv_folds, predictor_model=model
            )
            self.synthesized_features_ = [best_feature] if best_feature else []
            if best_feature:
                current_X = self.synthesis_engine_.create_enhanced_dataset(current_X)

        # Update feature tracking
        n_synthesized = len(self.synthesized_features_)
        self.synthesized_feature_names_ = [
            f"synthesized_feature_{i}" for i in range(n_synthesized)
        ]
        self.synthesis_performed_ = True

        if self.verbose:
            print(f"FeatureEnhancer: Synthesized {n_synthesized} new features")

        self.X_after_synthesis_ = current_X

        # Phase 2: Feature Selection (if configured)
        if selection_config is not None:
            if self.verbose:
                print(
                    "\n=== Starting feature selection phase (NSGA-II with cross-validation) ==="
                )

            self._setup_selection_engine(selection_config, model)

            # Fit the selector using cross-validation (no separate validation split needed)
            self.selection_engine_.fit(current_X, y)

            # Transform the data to get selected features
            current_X = self.selection_engine_.transform(current_X)

            # Update tracking
            self.selected_features_ = self.selection_engine_.selected_features_.tolist()
            self.selection_performed_ = True

            if self.verbose:
                n_selected = len(self.selected_features_)
                total_before_selection = self.X_after_synthesis_.shape[1]
                print(
                    f"FeatureEnhancer: Selected {n_selected} features from {total_before_selection}"
                )
        else:
            if self.verbose:
                print(
                    "FeatureEnhancer: Skipping feature selection (no config provided)"
                )
            # If no selection, all features are "selected"
            self.selected_features_ = list(range(current_X.shape[1]))

        self.X_after_selection_ = current_X

        # Build feature origin mapping and names
        self._build_feature_mapping()

        self.is_fitted_ = True

        if self.verbose:
            final_features = current_X.shape[1]
            print(
                f"FeatureEnhancer: Complete. Final dataset has {final_features} features"
            )

        return self

    def _build_feature_mapping(self):
        """Build mapping of final features to their origins."""
        self.feature_origin_map_ = {}
        self.selected_feature_names_ = []

        # Determine the features after synthesis (before selection)
        total_features_after_synthesis = self.X_after_synthesis_.shape[1]
        n_original = self.n_features_in_
        n_synthesized = total_features_after_synthesis - n_original

        # Create full feature list after synthesis
        full_feature_names = self.feature_names_in_ + [
            f"synthesized_feature_{i}" for i in range(n_synthesized)
        ]

        # Map selected features to their origins
        for final_idx, original_idx in enumerate(self.selected_features_):
            self.selected_feature_names_.append(full_feature_names[original_idx])

            if original_idx < n_original:
                # Original feature
                self.feature_origin_map_[final_idx] = {
                    "type": "original",
                    "original_index": original_idx,
                    "name": self.feature_names_in_[original_idx],
                }
            else:
                # Synthesized feature
                synth_idx = original_idx - n_original
                self.feature_origin_map_[final_idx] = {
                    "type": "synthesized",
                    "synthesis_index": synth_idx,
                    "name": f"synthesized_feature_{synth_idx}",
                    "expression": str(self.synthesized_features_[synth_idx])
                    if synth_idx < len(self.synthesized_features_)
                    else "unknown",
                }

    def transform(self, X):
        """
        Transform features using the fitted enhancer.

        Args:
            X: Features to transform

        Returns:
            X_transformed: Enhanced and selected features
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureEnhancer must be fitted before transform")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)

        # Validate input dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        current_X = X

        # Apply scaling if it was used during fitting
        if self.scale_features and self.scaler_ is not None:
            current_X = self.scaler_.transform(current_X)

        # Apply synthesis transformations
        if self.synthesis_performed_ and self.synthesis_engine_ is not None:
            if hasattr(self.synthesis_engine_, "best_features"):
                # Multi-feature case
                current_X = self.synthesis_engine_.create_multi_enhanced_dataset(
                    current_X
                )
            else:
                # Single feature case
                current_X = self.synthesis_engine_.create_enhanced_dataset(current_X)

        # Apply selection transformations
        if self.selection_performed_ and self.selection_engine_ is not None:
            current_X = self.selection_engine_.transform(current_X)
        else:
            # If no selection was performed, select the features that were available
            current_X = current_X[:, self.selected_features_]

        return current_X

    def fit_transform(self, X, y, model):
        """
        Fit the enhancer and transform the features using cross-validation.

        Args:
            X: Input features
            y: Target values
            model: ML model for evaluation

        Returns:
            X_transformed: Enhanced and selected features
        """
        self.fit(X, y, model)
        return self.transform(X)

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the feature enhancement process.

        Returns:
            Dictionary with detailed feature information
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "FeatureEnhancer must be fitted before getting feature info"
            )

        info = {
            "n_features_original": self.n_features_in_,
            "n_features_final": len(self.selected_features_),
            "scaling_performed": self.scale_features,
            "synthesis_performed": self.synthesis_performed_,
            "selection_performed": self.selection_performed_,
            "feature_names_original": self.feature_names_in_,
            "feature_names_final": self.selected_feature_names_,
            "feature_origin_map": self.feature_origin_map_,
        }

        # Add synthesis info
        if self.synthesis_performed_:
            info["synthesis_info"] = {
                "n_features_synthesized": len(self.synthesized_features_),
                "synthesized_expressions": [str(f) for f in self.synthesized_features_],
                "synthesis_statistics": self.synthesis_engine_.get_statistics()
                if hasattr(self.synthesis_engine_, "get_statistics")
                else None,
            }

        # Add selection info
        if self.selection_performed_:
            info["selection_info"] = self.selection_engine_.get_feature_importance()

        # Summary statistics
        n_original_selected = sum(
            1 for f in self.feature_origin_map_.values() if f["type"] == "original"
        )
        n_synthesized_selected = sum(
            1 for f in self.feature_origin_map_.values() if f["type"] == "synthesized"
        )

        info["summary"] = {
            "original_features_selected": n_original_selected,
            "synthesized_features_selected": n_synthesized_selected,
            "total_features_selected": len(self.selected_features_),
            "feature_reduction_ratio": 1
            - (len(self.selected_features_) / self.n_features_in_)
            if self.n_features_in_ > 0
            else 0,
        }

        return info

    def get_synthesized_features_info(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about synthesized features.

        Returns:
            List of dictionaries with synthesized feature details
        """
        if not self.synthesis_performed_:
            return []

        info = []
        for i, feature in enumerate(self.synthesized_features_):
            feature_info = {
                "index": i,
                "name": f"synthesized_feature_{i}",
                "expression": str(feature),
                "selected": any(
                    f["type"] == "synthesized" and f["synthesis_index"] == i
                    for f in self.feature_origin_map_.values()
                ),
                "complexity": feature.get_size()
                if hasattr(feature, "get_size")
                else None,
                "fitness": feature.fitness if hasattr(feature, "fitness") else None,
            }
            info.append(feature_info)

        return info

    def get_selected_features_summary(self) -> pd.DataFrame:
        """
        Get a summary of selected features in DataFrame format.

        Returns:
            DataFrame with selected feature information
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "FeatureEnhancer must be fitted before getting selected features summary"
            )

        data = []
        for final_idx, origin_info in self.feature_origin_map_.items():
            row = {
                "final_index": final_idx,
                "final_name": self.selected_feature_names_[final_idx],
                "origin_type": origin_info["type"],
                "original_name": origin_info.get("name", ""),
                "expression": origin_info.get("expression", "")
                if origin_info["type"] == "synthesized"
                else "",
            }
            data.append(row)

        return pd.DataFrame(data)

    @classmethod
    def from_config_files(
        cls,
        synthesis_config_path: Optional[str] = None,
        selection_config_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Create FeatureEnhancer from JSON configuration files.

        Args:
            synthesis_config_path: Path to synthesis configuration JSON file. If None, default config is used.
            selection_config_path: Path to selection configuration JSON file. If None, default config is used.
            **kwargs: Additional arguments for FeatureEnhancer constructor

        Returns:
            FeatureEnhancer instance
        """
        synthesis_config = {}  # Use empty dict for defaults instead of None
        selection_config = {}  # Use empty dict for defaults instead of None

        if synthesis_config_path:
            with open(synthesis_config_path, "r") as f:
                synthesis_config = json.load(f)

        if selection_config_path:
            with open(selection_config_path, "r") as f:
                selection_config = json.load(f)

        return cls(
            synthesis_config=synthesis_config,
            selection_config=selection_config,
            **kwargs,
        )

    def get_scaler(self):
        """
        Get the fitted scaler object.

        Returns:
            StandardScaler: The fitted scaler, or None if scaling wasn't performed
        """
        if not self.scale_features:
            return None
        return self.scaler_

    def inverse_scale_features(self, X_scaled):
        """
        Apply inverse scaling to scaled features.

        Args:
            X_scaled: Scaled features to inverse transform

        Returns:
            X_original_scale: Features in original scale
        """
        if not self.scale_features or self.scaler_ is None:
            return X_scaled

        # Only inverse transform if we have the right number of features
        if X_scaled.shape[1] == self.n_features_in_:
            return self.scaler_.inverse_transform(X_scaled)
        else:
            warnings.warn(
                f"Cannot inverse scale features: expected {self.n_features_in_} features, "
                f"got {X_scaled.shape[1]}. Returning original data."
            )
            return X_scaled

    def __str__(self) -> str:
        return f"{self.__class__.__name__}\n\t{str(self.synthesis_engine_)}\n\t{str(self.selection_engine_)}"
