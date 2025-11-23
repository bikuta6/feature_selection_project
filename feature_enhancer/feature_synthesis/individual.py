import operator
from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np


class Node:
    """Represents a node in the genetic programming tree."""

    def __init__(self, value: Union[str, int, float], arity: int = 0):
        self.value = value
        self.arity = arity  # Number of children this node expects
        self.children: List["Node"] = []

    def is_terminal(self) -> bool:
        """Returns True if this is a terminal node (leaf)."""
        return self.arity == 0

    def add_child(self, child: "Node"):
        """Add a child node."""
        if len(self.children) < self.arity:
            self.children.append(child)

    def copy(self) -> "Node":
        """Create a deep copy of this node and its subtree."""
        new_node = Node(self.value, self.arity)
        new_node.children = [child.copy() for child in self.children]
        return new_node

    def evaluate(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the expression tree with given data."""
        if self.is_terminal():
            if isinstance(self.value, int):
                # Feature reference (column index)
                return data[:, self.value]
            else:
                # Constant
                return np.full(data.shape[0], self.value)
        else:
            # Function node
            child_values = [child.evaluate(data) for child in self.children]
            return self._apply_function(child_values)

    def _apply_function(self, child_values: List[np.ndarray]) -> np.ndarray:
        """Apply the function represented by this node."""
        if self.value == "+":
            return child_values[0] + child_values[1]
        elif self.value == "-":
            return child_values[0] - child_values[1]
        elif self.value == "*":
            return child_values[0] * child_values[1]
        elif self.value == "/":
            # Protected division
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.divide(child_values[0], child_values[1])
                result = np.where(np.isfinite(result), result, 1.0)
            return result
        elif self.value == "sin":
            return np.sin(child_values[0])
        elif self.value == "cos":
            return np.cos(child_values[0])
        elif self.value == "exp":
            # Protected exponential
            return np.exp(np.clip(child_values[0], -10, 10))
        elif self.value == "tanh":
            # Hyperbolic tangent
            return np.tanh(child_values[0])
        elif self.value == "abs":
            return np.abs(child_values[0])
        elif self.value == "neg":
            return -child_values[0]
        else:
            raise ValueError(f"Unknown function: {self.value}")

    def get_depth(self) -> int:
        """Get the depth of this subtree."""
        if self.is_terminal():
            return 1
        return 1 + max(child.get_depth() for child in self.children)

    def get_size(self) -> int:
        """Get the total number of nodes in this subtree."""
        if self.is_terminal():
            return 1
        return 1 + sum(child.get_size() for child in self.children)

    def get_all_nodes(self) -> List["Node"]:
        """Get all nodes in this subtree."""
        if self.is_terminal():
            return [self]
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def __str__(self) -> str:
        """String representation of the expression tree."""
        if self.is_terminal():
            if isinstance(self.value, int):
                return f"x{self.value}"
            else:
                return str(self.value)
        else:
            if len(self.children) == 1:
                return f"{self.value}({self.children[0]})"
            elif len(self.children) == 2:
                return f"({self.children[0]} {self.value} {self.children[1]})"
            else:
                child_strs = [str(child) for child in self.children]
                return f"{self.value}({', '.join(child_strs)})"


class GPIndividual:
    """Individual for genetic programming-based feature synthesis."""

    # Function set with their arities
    FUNCTION_SET = {
        "+": 2,
        "-": 2,
        "*": 2,
        "/": 2,  # Basic arithmetic
        "sin": 1,
        "cos": 1,  # Trigonometric
        "exp": 1,  # Exponential
        "tanh": 1,
        "abs": 1,  # Mathematical functions
        "neg": 1,  # Negation
    }

    def __init__(self, n_features: int, max_depth: int = 6):
        self.n_features = n_features
        self.feat_probs = np.array([1 / n_features] * n_features)
        self.max_depth = max_depth
        self.probabilities = np.array(
            [a**2 for a in self.FUNCTION_SET.values()], dtype=float
        )
        self.probabilities /= self.probabilities.sum().astype(float)
        self.tree = self._generate_random_tree(max_depth)

        # Ensure we don't have a single terminal node (feature reference or constant) as the entire tree
        while self.tree.is_terminal():
            self.tree = self._generate_random_tree(max_depth)

        self.fitness = 0.0

    def _generate_random_tree(self, max_depth: int, current_depth: int = 0) -> Node:
        """Generate a random expression tree."""
        # Terminal probability increases with depth
        terminal_prob = current_depth / max_depth if max_depth > 0 else 1.0
        if current_depth == 0:
            terminal_prob = 0.0  # Ensure root is not terminal

        if current_depth >= max_depth or (
            current_depth > 0 and np.random.random() < terminal_prob
        ):
            # Create terminal node
            if np.random.random() < 0.8:  # Feature reference
                feature_index = np.random.choice(self.n_features, p=self.feat_probs)
                # divide probability of selected feature by half
                self.feat_probs[feature_index] /= 2
                self.feat_probs /= self.feat_probs.sum()
                return Node(feature_index, 0)
            else:  # Constant
                return Node(np.random.uniform(-2, 2), 0)
        else:
            # Create function node
            func_name = np.random.choice(
                list(self.FUNCTION_SET.keys()), p=self.probabilities
            )
            arity = self.FUNCTION_SET[func_name]
            node = Node(func_name, arity)

            # Generate children
            for _ in range(arity):
                child = self._generate_random_tree(max_depth, current_depth + 1)
                node.add_child(child)

            return node

    def evaluate_feature(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the expression tree to create a new feature."""
        return self.tree.evaluate(data)

    def copy(self) -> "GPIndividual":
        """Create a copy of this individual."""
        new_individual = GPIndividual(self.n_features, self.max_depth)
        new_individual.tree = self.tree.copy()
        new_individual.fitness = self.fitness
        return new_individual

    def get_depth(self) -> int:
        """Get the depth of the expression tree."""
        return self.tree.get_depth()

    def get_size(self) -> int:
        """Get the size of the expression tree."""
        return self.tree.get_size()

    def __str__(self) -> str:
        """String representation of the individual."""
        return str(self.tree)
