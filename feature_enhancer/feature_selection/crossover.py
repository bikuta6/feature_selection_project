"""
Crossover operators for feature selection using NSGA-II.

This module provides various crossover operators for creating offspring
in genetic algorithms for feature selection optimization.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .individual import Individual


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators."""

    @abstractmethod
    def __call__(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.

        Args:
            parent1: First parent individual
            parent2: Second parent individual

        Returns:
            Tuple of two offspring individuals
        """
        pass

    def _apply_constraints(self, individual: Individual) -> None:
        """
        Apply constraints to ensure at least one feature is selected.

        Args:
            individual: Individual to apply constraints to
        """
        if not np.any(individual.chromosome):
            # If no features selected, randomly select one
            individual.chromosome[np.random.randint(len(individual.chromosome))] = True


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover operator.

    Crosses over parents at a single randomly selected point,
    creating two complementary offspring.
    """

    def __call__(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform single-point crossover between two parents."""
        n_features = len(parent1.chromosome)
        crossover_point = np.random.randint(1, n_features)

        # Create offspring
        offspring1 = Individual(n_features)
        offspring2 = Individual(n_features)

        # Single point crossover
        offspring1.chromosome[:crossover_point] = parent1.chromosome[:crossover_point]
        offspring1.chromosome[crossover_point:] = parent2.chromosome[crossover_point:]

        offspring2.chromosome[:crossover_point] = parent2.chromosome[:crossover_point]
        offspring2.chromosome[crossover_point:] = parent1.chromosome[crossover_point:]

        # Apply constraints
        self._apply_constraints(offspring1)
        self._apply_constraints(offspring2)

        return offspring1, offspring2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover operator.

    Crosses over parents between two randomly selected points,
    swapping the middle segment between parents.
    """

    def __call__(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform two-point crossover between two parents."""
        n_features = len(parent1.chromosome)
        point1 = np.random.randint(1, n_features)
        point2 = np.random.randint(point1, n_features)

        offspring1 = Individual(n_features)
        offspring2 = Individual(n_features)

        # Two point crossover
        offspring1.chromosome = parent1.chromosome.copy()
        offspring2.chromosome = parent2.chromosome.copy()

        offspring1.chromosome[point1:point2] = parent2.chromosome[point1:point2]
        offspring2.chromosome[point1:point2] = parent1.chromosome[point1:point2]

        self._apply_constraints(offspring1)
        self._apply_constraints(offspring2)

        return offspring1, offspring2


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator.

    Each gene is independently swapped between parents with a given probability,
    creating diverse offspring with mixed characteristics.
    """

    def __init__(self, swap_probability: float = 0.5):
        """
        Initialize uniform crossover operator.

        Args:
            swap_probability: Probability of swapping each gene between parents
        """
        self.swap_probability = swap_probability

    def __call__(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform uniform crossover between two parents."""
        n_features = len(parent1.chromosome)

        offspring1 = Individual(n_features)
        offspring2 = Individual(n_features)

        # Uniform crossover
        for i in range(n_features):
            if np.random.random() < self.swap_probability:
                offspring1.chromosome[i] = parent2.chromosome[i]
                offspring2.chromosome[i] = parent1.chromosome[i]
            else:
                offspring1.chromosome[i] = parent1.chromosome[i]
                offspring2.chromosome[i] = parent2.chromosome[i]

        self._apply_constraints(offspring1)
        self._apply_constraints(offspring2)

        return offspring1, offspring2


class ArithmeticCrossover(CrossoverOperator):
    """
    Arithmetic crossover operator for binary chromosomes.

    Performs arithmetic combination of parent chromosomes and converts
    back to binary using a threshold. Useful for exploring intermediate solutions.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize arithmetic crossover operator.

        Args:
            alpha: Weight factor for combining parents (0 < alpha < 1)
        """
        self.alpha = alpha

    def __call__(
        self, parent1: Individual, parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Perform arithmetic crossover between two parents."""
        n_features = len(parent1.chromosome)

        offspring1 = Individual(n_features)
        offspring2 = Individual(n_features)

        # Convert to float for arithmetic operations
        p1_float = parent1.chromosome.astype(float)
        p2_float = parent2.chromosome.astype(float)

        # Arithmetic crossover
        o1_float = self.alpha * p1_float + (1 - self.alpha) * p2_float
        o2_float = (1 - self.alpha) * p1_float + self.alpha * p2_float

        # Convert back to boolean (threshold at 0.5)
        offspring1.chromosome = o1_float > 0.5
        offspring2.chromosome = o2_float > 0.5

        self._apply_constraints(offspring1)
        self._apply_constraints(offspring2)

        return offspring1, offspring2
