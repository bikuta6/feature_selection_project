"""
Feature Enhancer Package

A comprehensive feature engineering toolkit that combines feature synthesis
using genetic programming with feature selection using NSGA-II optimization.
"""

from .feature_enhancer import FeatureEnhancer
from .feature_selection.feature_selector import FeatureSelector
from .feature_synthesis.feature_synthesis import SimpleGA, MultiFeatureGA
from .dataset_utils import DatasetLoader, print_dataset_summary

__version__ = "1.0.0"
__author__ = "Feature Enhancer Team"

__all__ = [
    "FeatureEnhancer",
    "FeatureSelector", 
    "SimpleGA",
    "MultiFeatureGA",
    "DatasetLoader",
    "print_dataset_summary"
]
