"""
Individual representation for feature selection using NSGA-II.

This module defines the Individual class used in the NSGA-II genetic algorithm
for multi-objective feature selection optimization.
"""

from typing import List
import numpy as np


class Individual:
    """
    Represents an individual solution in the NSGA-II population.
    
    Each individual has a binary chromosome representing feature selection,
    objective values, dominance rank, and crowding distance for NSGA-II.
    
    Attributes:
        chromosome (np.ndarray): Binary array indicating selected features
        objectives (List[float]): Multi-objective fitness values
        rank (int): Dominance rank in NSGA-II
        crowding_distance (float): Crowding distance for diversity preservation
    """
    
    def __init__(self, n_features: int):
        """
        Initialize individual with random feature selection.
        
        Args:
            n_features: Total number of features available for selection
        """
        # Generate random chromosome ensuring at least one feature is selected
        self.chromosome = np.random.choice([True, False], size=n_features)
        while not np.any(self.chromosome):
            self.chromosome = np.random.choice([True, False], size=n_features)

        self.objectives: List[float] = [0.0, 0.0]
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another in Pareto sense.
        
        Args:
            other: Another individual to compare against
            
        Returns:
            True if this individual dominates the other, False otherwise
        """
        return (
            self.objectives[0] >= other.objectives[0]
            and self.objectives[1] >= other.objectives[1]
            and (
                self.objectives[0] > other.objectives[0]
                or self.objectives[1] > other.objectives[1]
            )
        )
