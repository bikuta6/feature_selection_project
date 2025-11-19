"""
Fitness functions for feature selection using multi-objective optimization.

This module provides various fitness functions that can be used as objectives
in the NSGA-II algorithm for feature selection.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import cross_val_score


class FitnessFunction(ABC):
    """
    Abstract base class for fitness functions used in feature selection.

    All fitness functions should return values in [0, 1] where higher values
    indicate better fitness.
    """

    @abstractmethod
    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """
        Calculate fitness for an individual using cross-validation.

        Args:
            individual: Individual with chromosome representing feature selection
            model: ML model to evaluate
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds

        Returns:
            Fitness value in [0, 1]
        """
        pass


class ErrorFitness(FitnessFunction):
    """
    Fitness function based on prediction error reduction.

    Compares the model's error with selected features against a baseline error
    using all features. Higher fitness indicates lower error.
    """

    def __init__(self, metric: str = "mae"):
        """
        Initialize error fitness function.

        Args:
            baseline_error: Baseline error for comparison
            metric: Error metric ('mae' or 'accuracy')
        """
        self.metric = metric

        if metric not in ["mae", "accuracy"]:
            raise ValueError(f"Unsupported metric: {metric}")


    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate error-based fitness using cross-validation."""
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return -np.inf  # No features selected

        X_train_selected = X_train[:, selected_features]

        try:
            if self.metric == "mae":
                # For MAE, use negative MAE scoring and convert back
                scores = cross_val_score(
                    model,
                    X_train_selected,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_absolute_error",
                )
                error = np.mean(scores)  # Convert back to positive MAE
            elif self.metric == "accuracy":
                # For accuracy, calculate error rate (1 - accuracy)
                scores = cross_val_score(
                    model, X_train_selected, y_train, cv=cv, scoring="accuracy"
                )
                error = 1 - np.mean(scores)  # Error rate
            else:
                return 0.0
            
            
            return error

        except Exception:
            # Return low fitness if model fails
            return -np.inf


class R2Fitness(FitnessFunction):
    """
    Fitness function based on R² score.

    Promotes solutions with higher R² scores on validation data.
    """

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate R²-based fitness using cross-validation."""
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0  # No features selected

        X_train_selected = X_train[:, selected_features]

        try:
            # Use R² scoring with cross-validation
            scores = cross_val_score(
                model, X_train_selected, y_train, cv=cv, scoring="r2"
            )
            r2_score = np.mean(scores)
            return r2_score

        except Exception:
            # Return low fitness if model fails
            return -1.0


class SparsityFitness(FitnessFunction):
    """
    Fitness function based on feature sparsity.

    Promotes solutions with fewer features. Higher fitness indicates
    fewer selected features.
    """

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate sparsity-based fitness (fewer features = higher fitness)."""
        n_selected = np.sum(individual.chromosome)
        n_total = len(individual.chromosome)
        return 1.0 - (n_selected / n_total)


class CorrelationFitness(FitnessFunction):
    """
    Fitness function based on minimizing correlation between selected features.

    Promotes solutions with less correlated features to reduce redundancy.
    Higher fitness indicates lower average correlation.
    """

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate correlation-based fitness (lower correlation = higher fitness)."""
        selected_features = individual.chromosome

        if np.sum(selected_features) < 2:
            return 1.0  # Maximum fitness if less than 2 features

        X_selected = X_train[:, selected_features]

        # Calculate average absolute correlation between selected features
        correlations = []
        n_features = X_selected.shape[1]

        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    corr, _ = pearsonr(X_selected[:, i], X_selected[:, j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except Exception:
                    # Handle edge cases (constant features, etc.)
                    correlations.append(0.0)

        if not correlations:
            return 1.0

        avg_correlation = np.mean(correlations)
        return 1.0 - avg_correlation  # Minimize correlation (maximize fitness)


class VarianceFitness(FitnessFunction):
    """
    Fitness function based on maximizing variance of selected features.

    Promotes solutions with high-variance features, potentially more informative
    for discrimination between samples. Uses sigmoid normalization for stability.
    """

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate variance-based fitness (higher variance = higher fitness)."""
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0  # Minimum fitness if no features selected

        X_selected = X_train[:, selected_features]

        try:
            # Calculate average variance of selected features
            variances = np.var(X_selected, axis=0)
            avg_variance = np.mean(variances)

            if np.isnan(avg_variance) or np.isinf(avg_variance):
                return 0.0

            # Normalize by clipping to [0, 1] range
            return np.clip(avg_variance, 0.0, 1.0)

        except Exception:
            return 0.0


class InformationGainFitness(FitnessFunction):
    """
    Fitness function based on information gain of selected features.

    Approximates information gain using correlation with target variable.
    Higher correlation indicates potentially higher information content.
    
    NOTE: Only captures LINEAR relationships. Use MutualInformationFitness
    for non-linear relationships.
    """

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate information gain-based fitness (higher correlation = higher fitness)."""
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0  # Minimum fitness if no features selected

        X_selected = X_train[:, selected_features]

        try:
            # Approximate information gain using correlation with target
            correlations_with_target = []

            for i in range(X_selected.shape[1]):
                corr, _ = pearsonr(X_selected[:, i], y_train)
                if not np.isnan(corr):
                    correlations_with_target.append(abs(corr))

            if not correlations_with_target:
                return 0.0

            return np.mean(correlations_with_target)

        except Exception:
            return 0.0


class MutualInformationFitness(FitnessFunction):
    """
    Fitness function based on Mutual Information with target variable.
    
    Uses sklearn's mutual_info_regression/classif to measure both linear
    and non-linear relationships between features and target.
    
    Advantages over InformationGainFitness (correlation):
    - Captures non-linear relationships (polynomials, exponentials, etc.)
    - Detects complex interactions
    - More robust to outliers
    - Better for real-world data with complex patterns
    
    Args:
        task: 'regression' or 'classification'
        n_neighbors: Number of neighbors for KNN-based MI estimation (default: 3)
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, task='regression', n_neighbors=3, random_state=None):
        """Initialize Mutual Information fitness function."""
        if task not in ['regression', 'classification']:
            raise ValueError(f"Task must be 'regression' or 'classification', got {task}")
        
        self.task = task
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """
        Calculate MI-based fitness (higher MI = higher fitness).
        
        Returns:
            Average mutual information between selected features and target.
            Range: [0, +∞), normalized to [0, 1] for compatibility.
        """
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0  # Minimum fitness if no features selected

        X_selected = X_train[:, selected_features]

        try:
            # Calculate mutual information for each selected feature
            if self.task == 'regression':
                mi_scores = mutual_info_regression(
                    X_selected, 
                    y_train,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state
                )
            else:  # classification
                mi_scores = mutual_info_classif(
                    X_selected,
                    y_train,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state
                )
            
            # Filter out NaN/inf values
            mi_scores = mi_scores[~np.isnan(mi_scores) & ~np.isinf(mi_scores)]
            
            if len(mi_scores) == 0:
                return 0.0
            
            # Average MI across selected features
            avg_mi = np.mean(mi_scores)
            
            # Normalize to [0, 1] range using sigmoid-like function
            # MI values typically range from 0 to ~5 for normalized data
            # Use tanh for smooth normalization
            normalized_mi = np.tanh(avg_mi / 2.0)
            
            return float(normalized_mi)

        except Exception as e:
            # Return minimum fitness if calculation fails
            warnings.warn(f"MI calculation failed: {e}")
            return 0.0


class RedundancyFitness(FitnessFunction):
    """
    Fitness function based on minimizing redundancy between selected features.
    
    Uses Mutual Information to measure redundancy (unlike CorrelationFitness
    which only captures linear redundancy).
    
    Promotes solutions with diverse, non-redundant features by minimizing
    average MI between pairs of selected features.
    
    Args:
        n_neighbors: Number of neighbors for KNN-based MI estimation (default: 3)
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, n_neighbors=3, random_state=None):
        """Initialize Redundancy fitness function."""
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """
        Calculate redundancy-based fitness (lower redundancy = higher fitness).
        
        Returns:
            1.0 - normalized_redundancy, where redundancy is average MI
            between pairs of selected features.
        """
        selected_features = individual.chromosome
        n_selected = np.sum(selected_features)

        if n_selected < 2:
            return 1.0  # Maximum fitness if less than 2 features (no redundancy)

        X_selected = X_train[:, selected_features]

        try:
            # Calculate MI between all pairs of selected features
            redundancy_scores = []
            n_features = X_selected.shape[1]

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    try:
                        # Treat one feature as "target" to compute MI
                        mi_score = mutual_info_regression(
                            X_selected[:, [j]],  # Feature j
                            X_selected[:, i],     # Feature i as "target"
                            n_neighbors=self.n_neighbors,
                            random_state=self.random_state
                        )[0]
                        
                        if not np.isnan(mi_score) and not np.isinf(mi_score):
                            redundancy_scores.append(mi_score)
                    except Exception:
                        # Handle edge cases
                        redundancy_scores.append(0.0)

            if not redundancy_scores:
                return 1.0  # Maximum fitness if no valid redundancy scores

            # Average redundancy
            avg_redundancy = np.mean(redundancy_scores)
            
            # Normalize to [0, 1] and invert (lower redundancy = higher fitness)
            normalized_redundancy = np.tanh(avg_redundancy / 2.0)
            
            return float(1.0 - normalized_redundancy)

        except Exception as e:
            warnings.warn(f"Redundancy calculation failed: {e}")
            return 1.0  # Default to max fitness if calculation fails


class MRMRFitness(FitnessFunction):
    """
    Minimum Redundancy Maximum Relevance (mRMR) fitness function.
    
    Combines MutualInformationFitness and RedundancyFitness to select
    features that are:
    1. Highly relevant to the target (high MI with y)
    2. Minimally redundant with each other (low MI between features)
    
    This is the gold standard for feature selection in many applications.
    
    Args:
        task: 'regression' or 'classification'
        alpha: Weight for relevance (default: 0.7)
        beta: Weight for redundancy (default: 0.3)
        n_neighbors: Number of neighbors for KNN-based MI estimation
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, task='regression', alpha=0.7, beta=0.3, 
                 n_neighbors=3, random_state=None):
        """Initialize mRMR fitness function."""
        if not np.isclose(alpha + beta, 1.0):
            raise ValueError(f"alpha + beta must equal 1.0, got {alpha + beta}")
        
        self.task = task
        self.alpha = alpha  # Weight for relevance
        self.beta = beta    # Weight for redundancy
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        # Create sub-fitness functions
        self.mi_fitness = MutualInformationFitness(
            task=task, 
            n_neighbors=n_neighbors, 
            random_state=random_state
        )
        self.redundancy_fitness = RedundancyFitness(
            n_neighbors=n_neighbors,
            random_state=random_state
        )
    
    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """
        Calculate mRMR fitness.
        
        Returns:
            Weighted combination of relevance (MI with target) and
            non-redundancy (inverse MI between features).
        """
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0

        # Calculate relevance (MI with target)
        relevance = self.mi_fitness(individual, model, X_train, y_train, cv)
        
        # Calculate non-redundancy (1 - redundancy between features)
        non_redundancy = self.redundancy_fitness(individual, model, X_train, y_train, cv)
        
        # Combine with weights
        mrmr_score = self.alpha * relevance + self.beta * non_redundancy
        
        return float(mrmr_score)
