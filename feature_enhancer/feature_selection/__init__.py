"""
Feature selection module using multi-objective optimization.
Provides NSGA-II based feature selection with various fitness functions.
"""

from .individual import Individual
from .crossover import (
    CrossoverOperator,
    SinglePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
)
from .mutation import MutationOperator, RandomBitFlip, UniformMutation, BlockMutation
from .fitness import (
    FitnessFunction,
    ErrorFitness,
    R2Fitness,
    SparsityFitness,
    CorrelationFitness,
    VarianceFitness,
    InformationGainFitness,
)
from .nsga2 import NSGA2
from .feature_selector import FeatureSelector

__all__ = [
    "Individual",
    "CrossoverOperator",
    "SinglePointCrossover",
    "TwoPointCrossover",
    "UniformCrossover",
    "MutationOperator",
    "RandomBitFlip",
    "UniformMutation",
    "BlockMutation",
    "FitnessFunction",
    "ErrorFitness",
    "R2Fitness",
    "SparsityFitness",
    "CorrelationFitness",
    "VarianceFitness",
    "InformationGainFitness",
    "MutualInformationFitness",
    "NSGA2",
    "FeatureSelector",
]
