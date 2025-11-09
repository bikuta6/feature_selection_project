"""
Fitness functions for feature selection using multi-objective optimization.

This module provides various fitness functions that can be used as objectives
in the NSGA-II algorithm for feature selection.
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from typing import Any, Union
import warnings


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
    
    def __init__(self, baseline_error: float, metric: str = "mae"):
        """
        Initialize error fitness function.
        
        Args:
            baseline_error: Baseline error for comparison
            metric: Error metric ('mae' or 'accuracy')
        """
        self.baseline_error = baseline_error
        self.metric = metric
        
        if metric not in ['mae', 'accuracy']:
            raise ValueError(f"Unsupported metric: {metric}")

    @classmethod
    def create_with_baseline(cls, model, X_train, y_train, metric="mae", cv=3):
        """
        Create fitness function with automatically calculated baseline error using cross-validation.
        
        Args:
            model: ML model to use for baseline calculation
            X_train: Training features
            y_train: Training labels
            metric: Error metric ('mae' or 'accuracy')
            cv: Number of cross-validation folds
            
        Returns:
            ErrorFitness instance with calculated baseline
        """
        if metric == "mae":
            # For MAE, we use negative MAE scoring (sklearn convention) and convert back
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
            baseline_error = -np.mean(scores)  # Convert back to positive MAE
        elif metric == "accuracy":
            # For accuracy, we calculate error rate (1 - accuracy)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            baseline_error = 1 - np.mean(scores)  # Error rate
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        return cls(baseline_error, metric)

    def __call__(self, individual, model, X_train, y_train, cv=3) -> float:
        """Calculate error-based fitness using cross-validation."""
        selected_features = individual.chromosome

        if not np.any(selected_features):
            return 0.0  # No features selected

        X_train_selected = X_train[:, selected_features]

        try:
            if self.metric == "mae":
                # For MAE, use negative MAE scoring and convert back
                scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='neg_mean_absolute_error')
                error = -np.mean(scores)  # Convert back to positive MAE
            elif self.metric == "accuracy":
                # For accuracy, calculate error rate (1 - accuracy)
                scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
                error = 1 - np.mean(scores)  # Error rate
            else:
                return 0.0

            # Normalize and convert to fitness: fitness = 1 - clamp(error/baseline, 0, 1)
            normalized_error = np.clip(error / self.baseline_error, 0, 1)
            return 1 - normalized_error

        except Exception:
            # Return low fitness if model fails
            return 0.0
        
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
            scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='r2')
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
