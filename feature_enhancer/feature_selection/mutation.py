"""
Mutation operators for feature selection using NSGA-II.

This module provides various mutation operators for introducing diversity
in genetic algorithms for feature selection optimization.
"""

from abc import ABC, abstractmethod

import numpy as np

from .individual import Individual


class MutationOperator(ABC):
    """Abstract base class for mutation operators."""

    def __init__(self, probability: float):
        """
        Initialize mutation operator.

        Args:
            probability: Mutation probability (0 < probability < 1)
        """
        self.probability = probability

    @abstractmethod
    def __call__(self, individual: Individual) -> None:
        """
        Apply mutation to an individual.

        Args:
            individual: Individual to mutate (modified in-place)
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


class RandomBitFlip(MutationOperator):
    """
    Random bit-flip mutation operator.

    Each bit in the chromosome is independently flipped with the given probability.
    This provides fine-grained exploration of the solution space.
    """

    def __call__(self, individual: Individual) -> None:
        """Apply random bit-flip mutation to individual."""
        for i in range(len(individual.chromosome)):
            if np.random.random() < self.probability:
                individual.chromosome[i] = not individual.chromosome[i]

        # Apply constraints
        self._apply_constraints(individual)


class UniformMutation(MutationOperator):
    """
    Uniform mutation operator.

    Flips a single randomly selected bit with the given probability.
    Provides controlled exploration with minimal disruption.
    """

    def __call__(self, individual: Individual) -> None:
        """Apply uniform mutation to individual."""
        if np.random.random() < self.probability:
            # Select random position and flip it
            pos = np.random.randint(len(individual.chromosome))
            individual.chromosome[pos] = not individual.chromosome[pos]

            # Apply constraints
            self._apply_constraints(individual)


class BlockMutation(MutationOperator):
    """
    Block mutation operator.

    Flips a contiguous block of bits with the given probability.
    Useful for exploring correlated feature groups.
    """

    def __init__(self, probability: float, block_size: int = 3):
        """
        Initialize block mutation operator.

        Args:
            probability: Mutation probability
            block_size: Size of the block to mutate
        """
        super().__init__(probability)
        self.block_size = block_size

    def __call__(self, individual: Individual) -> None:
        """Apply block mutation to individual."""
        if np.random.random() < self.probability:
            # Select random block and flip all bits in it
            max_start = len(individual.chromosome) - self.block_size + 1
            if max_start > 0:
                start = np.random.randint(max_start)
                end = start + self.block_size
                for i in range(start, end):
                    individual.chromosome[i] = not individual.chromosome[i]

            # Apply constraints
            self._apply_constraints(individual)


class AdaptiveMutation(MutationOperator):
    """
    Adaptive mutation operator.

    Adjusts mutation rate based on the current feature selection ratio.
    Higher rates when too many or too few features are selected.
    """

    def __init__(
        self, probability: float, min_prob: float = 0.01, max_prob: float = 0.1
    ):
        """
        Initialize adaptive mutation operator.

        Args:
            probability: Base mutation probability
            min_prob: Minimum mutation probability
            max_prob: Maximum mutation probability
        """
        super().__init__(probability)
        self.min_prob = min_prob
        self.max_prob = max_prob

    def __call__(self, individual: Individual) -> None:
        """Apply adaptive mutation to individual."""
        # Adaptive probability based on number of selected features
        n_selected = np.sum(individual.chromosome)
        total_features = len(individual.chromosome)

        # Higher mutation rate when too many or too few features are selected
        ratio = n_selected / total_features if total_features > 0 else 0
        if ratio < 0.3 or ratio > 0.7:
            current_prob = self.max_prob
        else:
            current_prob = self.min_prob

        for i in range(len(individual.chromosome)):
            if np.random.random() < current_prob:
                individual.chromosome[i] = not individual.chromosome[i]

        # Apply constraints
        self._apply_constraints(individual)
