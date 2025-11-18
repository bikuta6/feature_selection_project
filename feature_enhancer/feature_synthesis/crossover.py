import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from .individual import GPIndividual, Node


class CrossoverOperator(ABC):
    """Abstract base class for crossover operators in genetic programming."""

    @abstractmethod
    def __call__(
        self, parent1: GPIndividual, parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform crossover between two parents and return two offspring."""
        pass


class SubtreeCrossover(CrossoverOperator):
    """Subtree crossover for genetic programming trees."""

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def __call__(
        self, parent1: GPIndividual, parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform subtree crossover between two GP individuals."""
        # Create copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        # Get all nodes from both trees
        nodes1 = offspring1.tree.get_all_nodes()
        nodes2 = offspring2.tree.get_all_nodes()

        if len(nodes1) == 0 or len(nodes2) == 0:
            return offspring1, offspring2

        # Select random crossover points
        crossover_node1 = np.random.choice(nodes1)
        crossover_node2 = np.random.choice(nodes2)

        # Find parents of crossover nodes
        parent_node1 = self._find_parent(offspring1.tree, crossover_node1)
        parent_node2 = self._find_parent(offspring2.tree, crossover_node2)

        # Perform the swap
        if parent_node1 is None:
            # Crossover node is root - check if replacement would create single terminal tree
            replacement = crossover_node2.copy()
            if replacement.is_terminal():
                # Create a more complex tree
                func_name = np.random.choice(list(offspring1.FUNCTION_SET.keys()), p=offspring1.probabilities)
                arity = offspring1.FUNCTION_SET[func_name]
                complex_tree = Node(func_name, arity)
                complex_tree.add_child(replacement)
                for _ in range(arity - 1):
                    if np.random.random() < 0.5:
                        child = Node(np.random.randint(0, offspring1.n_features), 0)
                    else:
                        child = Node(np.random.uniform(-2, 2), 0)
                    complex_tree.add_child(child)
                offspring1.tree = complex_tree
            else:
                offspring1.tree = replacement
        else:
            # Replace child in parent
            for i, child in enumerate(parent_node1.children):
                if child is crossover_node1:
                    parent_node1.children[i] = crossover_node2.copy()
                    break

        if parent_node2 is None:
            # Crossover node is root - check if replacement would create single terminal tree
            replacement = crossover_node1.copy()
            if replacement.is_terminal():
                # Create a more complex tree
                func_name = np.random.choice(list(offspring2.FUNCTION_SET.keys()))
                arity = offspring2.FUNCTION_SET[func_name]
                complex_tree = Node(func_name, arity)
                complex_tree.add_child(replacement)
                for _ in range(arity - 1):
                    if np.random.random() < 0.5:
                        child = Node(np.random.randint(0, offspring2.n_features), 0)
                    else:
                        child = Node(np.random.uniform(-2, 2), 0)
                    complex_tree.add_child(child)
                offspring2.tree = complex_tree
            else:
                offspring2.tree = replacement
        else:
            # Replace child in parent
            for i, child in enumerate(parent_node2.children):
                if child is crossover_node2:
                    parent_node2.children[i] = crossover_node1.copy()
                    break

        # Ensure depth constraints
        self._enforce_depth_constraint(offspring1)
        self._enforce_depth_constraint(offspring2)

        return offspring1, offspring2

    def _find_parent(self, root: Node, target: Node) -> Node:
        """Find the parent node of the target node."""
        if root is target:
            return None

        for child in root.children:
            if child is target:
                return root
            parent = self._find_parent(child, target)
            if parent is not None:
                return parent
        return None

    def _enforce_depth_constraint(self, individual: GPIndividual):
        """Ensure the tree doesn't exceed maximum depth."""
        if individual.get_depth() > self.max_depth:
            # Replace with a simpler random tree
            individual.tree = individual._generate_random_tree(self.max_depth // 2)


class RandomCrossover(CrossoverOperator):
    """Random crossover that creates new random subtrees at crossover points."""

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth

    def __call__(
        self, parent1: GPIndividual, parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform random crossover between two GP individuals."""
        # Create copies of parents
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        # Get all nodes from both trees
        nodes1 = offspring1.tree.get_all_nodes()
        nodes2 = offspring2.tree.get_all_nodes()

        if len(nodes1) == 0 or len(nodes2) == 0:
            return offspring1, offspring2

        # Select random crossover points
        crossover_node1 = np.random.choice(nodes1)
        crossover_node2 = np.random.choice(nodes2)

        # Create new random subtrees
        new_subtree1 = offspring1._generate_random_tree(
            min(self.max_depth // 2, self.max_depth - crossover_node1.get_depth() + 1)
        )
        new_subtree2 = offspring2._generate_random_tree(
            min(self.max_depth // 2, self.max_depth - crossover_node2.get_depth() + 1)
        )

        # Find parents and replace subtrees
        parent_node1 = self._find_parent(offspring1.tree, crossover_node1)
        parent_node2 = self._find_parent(offspring2.tree, crossover_node2)

        if parent_node1 is None:
            # Check if new subtree would create single terminal tree
            if new_subtree1.is_terminal():
                func_name = np.random.choice(list(offspring1.FUNCTION_SET.keys()), p=offspring1.probabilities)
                arity = offspring1.FUNCTION_SET[func_name]
                complex_tree = Node(func_name, arity)
                complex_tree.add_child(new_subtree1)
                for _ in range(arity - 1):
                    if np.random.random() < 0.5:
                        child = Node(np.random.randint(0, offspring1.n_features), 0)
                    else:
                        child = Node(np.random.uniform(-2, 2), 0)
                    complex_tree.add_child(child)
                offspring1.tree = complex_tree
            else:
                offspring1.tree = new_subtree1
        else:
            for i, child in enumerate(parent_node1.children):
                if child is crossover_node1:
                    parent_node1.children[i] = new_subtree1
                    break

        if parent_node2 is None:
            # Check if new subtree would create single terminal tree
            if new_subtree2.is_terminal():
                func_name = np.random.choice(list(offspring2.FUNCTION_SET.keys()))
                arity = offspring2.FUNCTION_SET[func_name]
                complex_tree = Node(func_name, arity)
                complex_tree.add_child(new_subtree2)
                for _ in range(arity - 1):
                    if np.random.random() < 0.5:
                        child = Node(np.random.randint(0, offspring2.n_features), 0)
                    else:
                        child = Node(np.random.uniform(-2, 2), 0)
                    complex_tree.add_child(child)
                offspring2.tree = complex_tree
            else:
                offspring2.tree = new_subtree2
        else:
            for i, child in enumerate(parent_node2.children):
                if child is crossover_node2:
                    parent_node2.children[i] = new_subtree2
                    break

        return offspring1, offspring2

    def _find_parent(self, root: Node, target: Node) -> Node:
        """Find the parent node of the target node."""
        if root is target:
            return None

        for child in root.children:
            if child is target:
                return root
            parent = self._find_parent(child, target)
            if parent is not None:
                return parent
        return None


class PointCrossover(CrossoverOperator):
    """Point crossover that exchanges nodes at corresponding positions."""

    def __call__(
        self, parent1: GPIndividual, parent2: GPIndividual
    ) -> Tuple[GPIndividual, GPIndividual]:
        """Perform point crossover between two GP individuals."""
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()

        # Get all nodes from both trees
        nodes1 = offspring1.tree.get_all_nodes()
        nodes2 = offspring2.tree.get_all_nodes()

        if len(nodes1) == 0 or len(nodes2) == 0:
            return offspring1, offspring2

        # Determine the minimum number of nodes to consider
        min_nodes = min(len(nodes1), len(nodes2))

        # Select a random position up to the minimum
        if min_nodes > 1:
            position = np.random.randint(0, min_nodes)

            # Exchange nodes at the selected position
            if position < len(nodes1) and position < len(nodes2):
                # Store the values to swap
                temp_value = nodes1[position].value
                temp_arity = nodes1[position].arity

                # Swap values and arities
                nodes1[position].value = nodes2[position].value
                nodes1[position].arity = nodes2[position].arity

                nodes2[position].value = temp_value
                nodes2[position].arity = temp_arity

                # Adjust children if arity changed
                self._adjust_children(nodes1[position], parent1.n_features)
                self._adjust_children(nodes2[position], parent2.n_features)

        return offspring1, offspring2

    def _adjust_children(self, node: Node, n_features: int):
        """Adjust children count based on node arity."""
        current_children = len(node.children)
        required_children = node.arity

        if current_children > required_children:
            # Remove excess children
            node.children = node.children[:required_children]
        elif current_children < required_children:
            # Add missing children as random terminals
            for _ in range(required_children - current_children):
                if np.random.random() < 0.8:
                    child = Node(np.random.randint(0, n_features), 0)
                else:
                    child = Node(np.random.uniform(-5, 5), 0)
                node.children.append(child)
