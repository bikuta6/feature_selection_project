"""
Feature synthesis module using genetic programming.
Creates new features from original data using non-linear operations.
"""

from .individual import GPIndividual, Node
from .crossover import SubtreeCrossover, RandomCrossover, PointCrossover
from .mutation import (
    SubtreeMutation, RandomMutation, NodeMutation, 
    ParameterMutation, GrowMutation
)
from .feature_synthesis import SimpleGA, MultiFeatureGA

__all__ = [
    'Node',
    'GPIndividual',
    'SubtreeCrossover',
    'RandomCrossover',
    'PointCrossover',
    'SubtreeMutation',
    'RandomMutation',
    'NodeMutation',
    'ParameterMutation',
    'GrowMutation',
    'SimpleGA',
    'MultiFeatureGA'
]