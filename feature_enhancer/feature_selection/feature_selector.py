import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .crossover import SinglePointCrossover
from .fitness import (
    ErrorFitness,
    R2Fitness,
    SparsityFitness,
    CorrelationFitness,
    VarianceFitness,
    InformationGainFitness,
    MutualInformationFitness,  # ← NEW
    RedundancyFitness,          # ← NEW
    MRMRFitness,                # ← NEW
)
from .individual import Individual
from .mutation import RandomBitFlip
from .nsga2 import NSGA2


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model,
        secondary_objective="sparsity",
        population_size=100,
        generations=50,
        crossover_prob=0.9,
        mutation_prob=0.01,
        crossover_operator=None,
        mutation_operator=None,
        objective_weights=None,
        metric="mae",
        normalize=False,
        random_state=None,
        cv=3,
    ):
        """
        Feature selector using NSGA-II multi-objective optimization with cross-validation.

        Args:
            model: ML model to use (must have fit/predict methods)
            secondary_objective: Secondary objective ('sparsity', 'correlation', 'variance', 'information_gain')
            population_size: NSGA-II population size
            generations: Number of generations
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
            crossover_operator: Custom crossover operator
            mutation_operator: Custom mutation operator
            objective_weights: Weights for final solution selection [w_error, w_secondary]
            metric: Metric for ErrorFitness ('mae' or 'accuracy')
            normalize: Whether to normalize features
            random_state: Random seed for reproducibility
            cv: Number of cross-validation folds
        """
        self.model = model
        self.secondary_objective = secondary_objective
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.objective_weights = objective_weights
        self.metric = metric
        self.normalize = normalize
        self.random_state = random_state
        self.cv = cv
        self.final_error = None

        # Validate secondary objective
        valid_objectives = ["sparsity", "correlation", "variance", "information_gain", "mutual_info", "redundancy", "mrmr"]  # ← UPDATED
        if self.secondary_objective not in valid_objectives:
            raise ValueError(f"secondary_objective must be one of: {valid_objectives}")

        # Internal attributes
        self.scaler_ = None
        self.nsga2_ = None
        self.selected_features_ = None
        self.feature_mask_ = None
        self.n_features_in_ = None
        self.baseline_error_ = None
        self.best_individual_ = None
        self.fitness_functions_ = None
        self.is_fitted_ = False

    def _setup_random_state(self):
        """Setup random seed for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _calculate_baseline_error(self, X_train, y_train):
        """Calculate baseline error using all features with cross-validation."""

        model_copy = self._clone_model()

        if self.metric == "mae":
            # For MAE, use negative MAE scoring and convert back
            scores = cross_val_score(
                model_copy,
                X_train,
                y_train,
                cv=self.cv,
                scoring="neg_mean_absolute_error",
            )
            return -np.mean(scores)  # Convert back to positive MAE
        elif self.metric == "accuracy":
            # For accuracy, calculate error rate (1 - accuracy)
            scores = cross_val_score(
                model_copy, X_train, y_train, cv=self.cv, scoring="accuracy"
            )
            return 1 - np.mean(scores)  # Error rate
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _clone_model(self):
        """Clone model to avoid modifying the original."""
        try:
            return clone(self.model)
        except:
            # Fallback if sklearn.base.clone is not available
            import copy

            return copy.deepcopy(self.model)

    def _setup_operators(self):
        """Setup default operators if not provided."""
        if self.crossover_operator is None:
            self.crossover_operator = SinglePointCrossover()

        if self.mutation_operator is None:
            self.mutation_operator = RandomBitFlip(self.mutation_prob)

    def _get_secondary_fitness_function(self):
        """Return secondary fitness function based on user selection."""
        if self.secondary_objective == "sparsity":
            return SparsityFitness()
        elif self.secondary_objective == "correlation":
            return CorrelationFitness()
        elif self.secondary_objective == "variance":
            return VarianceFitness()
        elif self.secondary_objective == "information_gain":
            return InformationGainFitness()
        elif self.secondary_objective == "mutual_info":  # ← NEW
            return MutualInformationFitness(task="regression" if self.metric == "mae" else "classification", random_state=self.random_state)
        elif self.secondary_objective == "redundancy":  # ← NEW
            return RedundancyFitness(random_state=self.random_state)
        elif self.secondary_objective == "mrmr":  # ← NEW
            return MRMRFitness(task="regression" if self.metric == "mae" else "classification", random_state=self.random_state)
        else:
            raise ValueError(
                f"Unrecognized secondary objective: {self.secondary_objective}"
            )

    def _setup_fitness_functions(self, baseline_error):
        """Setup fitness functions: Error (always) + Secondary (selectable)."""
        # First objective: always R2-based fitness (works well with cross-validation)
        error_fitness = R2Fitness()

        # Second objective: user selectable
        secondary_fitness = self._get_secondary_fitness_function()

        self.fitness_functions_ = [error_fitness, secondary_fitness]
        return self.fitness_functions_

    def fit(self, X_train, y_train):
        """
        Fit the feature selector using cross-validation.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            self: Fitted instance
        """
        self._setup_random_state()

        # Validate input
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.n_features_in_ = X_train.shape[1]

        # STEP 1: Feature normalization
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_train_scaled = self.scaler_.fit_transform(X_train)
        else:
            X_train_scaled = X_train

        # STEP 2: Calculate baseline error using cross-validation
        self.baseline_error_ = self._calculate_baseline_error(X_train_scaled, y_train)

        # STEP 3: Setup operators and fitness functions
        self._setup_operators()
        fitness_functions = self._setup_fitness_functions(self.baseline_error_)

        # STEP 4: Run NSGA-II
        self.nsga2_ = NSGA2(
            population_size=self.population_size,
            n_features=self.n_features_in_,
            fitness_functions=fitness_functions,
            crossover_operator=self.crossover_operator,
            mutation_operator=self.mutation_operator,
            crossover_prob=self.crossover_prob,
        )

        self.nsga2_.evolve(
            model=self.model,
            X_train=X_train_scaled,
            y_train=y_train,
            generations=self.generations,
            cv=self.cv,
        )

        # STEP 5: Select best individual
        self.best_individual_ = self.nsga2_.get_best_individual(self.objective_weights)

        if self.best_individual_ is None:
            raise RuntimeError("Could not find a valid solution")

        # STEP 6: Extract feature mask
        self.feature_mask_ = self.best_individual_.chromosome
        self.selected_features_ = np.where(self.feature_mask_)[0]

        # Calculate actual error improvement using cross-validation with selected features
        model_selected = self._clone_model()
        X_train_selected = X_train_scaled[:, self.feature_mask_]

        if self.metric == "mae":
            scores = cross_val_score(
                model_selected,
                X_train_selected,
                y_train,
                cv=self.cv,
                scoring="neg_mean_absolute_error",
            )
            selected_error = -np.mean(scores)  # Convert back to positive MAE
        elif self.metric == "accuracy":
            scores = cross_val_score(
                model_selected,
                X_train_selected,
                y_train,
                cv=self.cv,
                scoring="accuracy",
            )
            selected_error = 1 - np.mean(scores)  # Error rate

        self.final_error = selected_error

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """
        Transform features by selecting only the best ones.

        Args:
            X: Features to transform

        Returns:
            X_transformed: Selected and normalized features
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureSelector must be fitted before transform")

        X = np.asarray(X)

        # Check dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Normalize features
        if self.normalize:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        # Select features
        X_selected = X_scaled[:, self.feature_mask_]

        return X_selected

    def fit_transform(self, X_train, y_train, X_transform=None):
        """
        Fit the selector and transform features using cross-validation.

        Args:
            X_train, y_train: Data for fitting
            X_transform: Data to transform (default X_train)

        Returns:
            X_transformed: Transformed features
        """
        self.fit(X_train, y_train)

        if X_transform is None:
            X_transform = X_train

        return self.transform(X_transform)

    def inverse_transform(self, X_selected):
        """
        Reconstruct original features from selected ones.

        Args:
            X_selected: Selected and normalized features

        Returns:
            X_reconstructed: Features in original space
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "FeatureSelector must be fitted before inverse_transform"
            )

        X_selected = np.asarray(X_selected)

        # Check dimensions
        n_selected = len(self.selected_features_)
        if X_selected.shape[1] != n_selected:
            raise ValueError(
                f"X_selected has {X_selected.shape[1]} features, expected {n_selected}"
            )

        # Reconstruct full array with zeros for non-selected features
        X_full_scaled = np.zeros((X_selected.shape[0], self.n_features_in_))
        X_full_scaled[:, self.feature_mask_] = X_selected

        # Denormalize (inverse transform of scaler)
        if self.normalize:
            X_reconstructed = self.scaler_.inverse_transform(X_full_scaled)
        else:
            X_reconstructed = X_full_scaled

        return X_reconstructed

    def get_feature_importance(self):
        """
        Return information about selected features.

        Returns:
            dict: Feature information
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureSelector must be fitted first")

        return {
            "selected_features": self.selected_features_.tolist(),
            "feature_mask": self.feature_mask_.tolist(),
            "n_features_original": self.n_features_in_,
            "n_features_selected": len(self.selected_features_),
            "reduction_ratio": 1 - len(self.selected_features_) / self.n_features_in_,
            "objectives": {
                "error": self.best_individual_.objectives[0],
                self.secondary_objective: self.best_individual_.objectives[1],
            },
            "baseline_error": self.baseline_error_,
            "final_error": self.final_error,
            "secondary_objective_type": self.secondary_objective,
        }

    def get_pareto_front(self):
        """
        Return all solutions in the Pareto front.

        Returns:
            list: List of individuals in Pareto front
        """
        if not self.is_fitted_:
            raise RuntimeError("FeatureSelector must be fitted first")

        return self.nsga2_.get_pareto_front()

    def plot_pareto_front(self, figsize=(10, 6)):
        """
        Visualize the Pareto front.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.is_fitted_:
            raise RuntimeError("FeatureSelector must be fitted first")

        pareto_front = self.get_pareto_front()

        if not pareto_front:
            print("No solutions in Pareto front")
            return

        objectives_1 = [ind.objectives[0] for ind in pareto_front]
        objectives_2 = [ind.objectives[1] for ind in pareto_front]

        plt.figure(figsize=figsize)
        plt.scatter(objectives_1, objectives_2, c="blue", alpha=0.6)
        plt.xlabel(f"Objective 1: Error ({self.metric})")
        plt.ylabel(f"Objective 2: {self.secondary_objective.title()}")
        plt.title(
            f"Pareto Front - NSGA-II\nError vs {self.secondary_objective.title()}"
        )
        plt.grid(True, alpha=0.3)

        # Mark selected solution
        if self.best_individual_:
            plt.scatter(
                [self.best_individual_.objectives[0]],
                [self.best_individual_.objectives[1]],
                c="red",
                s=100,
                marker="*",
                label="Selected Solution",
            )
            plt.legend()

        plt.tight_layout()
        plt.show()

    def get_available_objectives(self):
        """
        Return available secondary objectives.

        Returns:
            dict: Description of available objectives
        """
        return {
            "sparsity": "Minimizes the number of selected features",
            "correlation": "Minimizes correlation between selected features",
            "variance": "Maximizes variance of selected features",
            "information_gain": "Maximizes information gain of features",
        }

    def __str__(self):
        """
        Return a string representation of the FeatureSelector configuration.

        Returns:
            str: Detailed configuration of the feature selector
        """
        lines = []
        lines.append("FeatureSelector Configuration:")
        lines.append("=" * 40)

        # Model information
        lines.append(f"Model: {type(self.model).__name__}")

        # Objective configuration
        lines.append(f"Secondary Objective: {self.secondary_objective}")
        lines.append(f"Metric: {self.metric}")

        # NSGA-II parameters
        lines.append(f"Population Size: {self.population_size}")
        lines.append(f"Generations: {self.generations}")

        # Operator information
        crossover_name = (
            type(self.crossover_operator).__name__
            if self.crossover_operator
            else "SinglePointCrossover (default)"
        )
        mutation_name = (
            type(self.mutation_operator).__name__
            if self.mutation_operator
            else "RandomBitFlip (default)"
        )
        lines.append(f"Crossover Operator: {crossover_name}")
        lines.append(f"Mutation Operator: {mutation_name}")

        # Probabilities
        lines.append(f"Crossover Probability: {self.crossover_prob}")
        lines.append(f"Mutation Probability: {self.mutation_prob}")

        # Other parameters
        lines.append(f"Objective Weights: {self.objective_weights}")
        lines.append(f"Normalize Features: {self.normalize}")
        lines.append(f"Cross-Validation Folds: {self.cv}")
        lines.append(f"Random State: {self.random_state}")

        # Fitting status and results
        lines.append(f"Fitted: {self.is_fitted_}")

        if self.is_fitted_:
            lines.append("=" * 40)
            lines.append("Results:")
            lines.append(f"Original Features: {self.n_features_in_}")
            lines.append(f"Selected Features: {len(self.selected_features_)}")
            lines.append(
                f"Reduction Ratio: {1 - len(self.selected_features_) / self.n_features_in_:.2%}"
            )
            lines.append(f"Baseline Error: {self.baseline_error_:.4f}")
            lines.append(f"Final Error: {self.final_error:.4f}")
            if self.best_individual_:
                lines.append(
                    f"Best Individual Objectives: {[f'{obj:.4f}' for obj in self.best_individual_.objectives]}"
                )

        return "\n".join(lines)


# Example usage

if __name__ == "__main__":
    # Example usage and testing
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Create selector with different secondary objectives (using cross-validation)
    print("=== Testing different secondary objectives with cross-validation ===")

    print("Number of original features:", X.shape[1])

    for secondary_obj in ["sparsity", "correlation", "variance"]:
        print(f"\n--- Secondary objective: {secondary_obj} ---")

        selector = FeatureSelector(
            model=LinearRegression(),
            secondary_objective=secondary_obj,
            population_size=50,
            generations=20,
            objective_weights=[0.9, 0.1],  # 90% error, 10% secondary
            random_state=42,
            cv=3,  # 3-fold cross-validation
        )

        # Fit and transform using only training data with cross-validation
        X_selected = selector.fit_transform(X, y)
        print(f"Result: {X_selected.shape[1]} features selected")

    print("\nAvailable objectives:")
    selector = FeatureSelector(model=LinearRegression())
    for obj, desc in selector.get_available_objectives().items():
        print(f"  {obj}: {desc}")
