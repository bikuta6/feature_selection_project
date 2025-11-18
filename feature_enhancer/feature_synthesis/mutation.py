import numpy as np
from abc import ABC, abstractmethod
from .individual import GPIndividual, Node


class MutationOperator(ABC):
    """Abstract base class for mutation operators in genetic programming."""

    def __init__(self, probability: float):
        self.probability = probability

    @abstractmethod
    def __call__(self, individual: GPIndividual) -> None:
        """Apply mutation to the individual."""
        pass


class SubtreeMutation(MutationOperator):
    """Subtree mutation that replaces a subtree with a new random subtree."""

    def __init__(self, probability: float, max_depth: int = 3):
        super().__init__(probability)
        self.max_depth = max_depth

    def __call__(self, individual: GPIndividual) -> None:
        """Apply subtree mutation to the individual."""
        if np.random.random() < self.probability:
            # Get all nodes in the tree
            nodes = individual.tree.get_all_nodes()

            if len(nodes) == 0:
                return

            # Select a random node to replace
            mutation_node = np.random.choice(nodes)

            # Generate a new random subtree
            new_subtree = individual._generate_random_tree(self.max_depth)

            # Ensure the new subtree is not a single terminal node (feature or constant)
            while new_subtree.is_terminal():
                new_subtree = individual._generate_random_tree(self.max_depth)

            # Find parent and replace the subtree
            parent_node = self._find_parent(individual.tree, mutation_node)

            if parent_node is None:
                # Mutation node is root - ensure we don't create single terminal tree
                if new_subtree.is_terminal():
                    # Force creation of a more complex tree
                    func_name = np.random.choice(list(individual.FUNCTION_SET.keys()), p=individual.probabilities)
                    arity = individual.FUNCTION_SET[func_name]
                    complex_tree = Node(func_name, arity)
                    complex_tree.add_child(new_subtree)
                    for _ in range(arity - 1):
                        if np.random.random() < 0.5:
                            child = Node(np.random.randint(0, individual.n_features), 0)
                        else:
                            child = Node(np.random.uniform(-2, 2), 0)
                        complex_tree.add_child(child)
                    individual.tree = complex_tree
                else:
                    individual.tree = new_subtree
            else:
                # Replace child in parent
                for i, child in enumerate(parent_node.children):
                    if child is mutation_node:
                        parent_node.children[i] = new_subtree
                        break

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


class NodeMutation(MutationOperator):
    """Node mutation that changes the value of a single node."""

    def __call__(self, individual: GPIndividual) -> None:
        """Apply node mutation to the individual."""
        if np.random.random() < self.probability:
            nodes = individual.tree.get_all_nodes()

            if len(nodes) == 0:
                return

            # If tree has only one terminal node (feature or constant), skip mutation
            if len(nodes) == 1 and nodes[0].is_terminal():
                return

            # Select a random node to mutate
            mutation_node = np.random.choice(nodes)

            if mutation_node.is_terminal():
                # Mutate terminal node
                if isinstance(mutation_node.value, int):
                    # Feature reference - change to another feature
                    mutation_node.value = np.random.randint(0, individual.n_features)
                else:
                    # Constant - change value
                    mutation_node.value = np.random.uniform(-2, 2)
            else:
                # Mutate function node - change to another function with same arity
                current_arity = mutation_node.arity
                compatible_functions = [
                    func
                    for func, arity in individual.FUNCTION_SET.items()
                    if arity == current_arity and func != mutation_node.value
                ]

                if compatible_functions:
                    mutation_node.value = np.random.choice(compatible_functions)


class RandomMutation(MutationOperator):
    """Random mutation that can perform various types of mutations."""

    def __init__(self, probability: float, max_depth: int = 3):
        super().__init__(probability)
        self.max_depth = max_depth

    def __call__(self, individual: GPIndividual) -> None:
        """Apply random mutation to the individual."""
        if np.random.random() < self.probability:
            mutation_type = np.random.choice(
                ["subtree", "node", "constant", "function"]
            )

            if mutation_type == "subtree":
                self._subtree_mutation(individual)
            elif mutation_type == "node":
                self._node_mutation(individual)
            elif mutation_type == "constant":
                self._constant_mutation(individual)
            elif mutation_type == "function":
                self._function_mutation(individual)

    def _subtree_mutation(self, individual: GPIndividual):
        """Replace a random subtree with a new one."""
        nodes = individual.tree.get_all_nodes()
        if len(nodes) == 0:
            return

        mutation_node = np.random.choice(nodes)
        new_subtree = individual._generate_random_tree(self.max_depth)

        # Ensure the new subtree is not a single terminal node (feature or constant)
        while new_subtree.is_terminal():
            new_subtree = individual._generate_random_tree(self.max_depth)

        parent_node = self._find_parent(individual.tree, mutation_node)

        if parent_node is None:
            # If replacing root, ensure we don't create single terminal tree
            if new_subtree.is_terminal():
                func_name = np.random.choice(list(individual.FUNCTION_SET.keys()), p=individual.probabilities)
                arity = individual.FUNCTION_SET[func_name]
                complex_tree = Node(func_name, arity)
                complex_tree.add_child(new_subtree)
                for _ in range(arity - 1):
                    if np.random.random() < 0.5:
                        child = Node(np.random.randint(0, individual.n_features), 0)
                    else:
                        child = Node(np.random.uniform(-2, 2), 0)
                    complex_tree.add_child(child)
                individual.tree = complex_tree
            else:
                individual.tree = new_subtree
        else:
            for i, child in enumerate(parent_node.children):
                if child is mutation_node:
                    parent_node.children[i] = new_subtree
                    break

    def _node_mutation(self, individual: GPIndividual):
        """Change a single node's value."""
        nodes = individual.tree.get_all_nodes()
        if len(nodes) == 0:
            return

        # If tree has only one terminal node (feature or constant), skip mutation
        if len(nodes) == 1 and nodes[0].is_terminal():
            return

        mutation_node = np.random.choice(nodes)

        if mutation_node.is_terminal():
            if isinstance(mutation_node.value, int):
                mutation_node.value = np.random.randint(0, individual.n_features)
            else:
                mutation_node.value = np.random.uniform(-2, 2)
        else:
            compatible_functions = [
                func
                for func, arity in individual.FUNCTION_SET.items()
                if arity == mutation_node.arity
            ]
            if compatible_functions:
                mutation_node.value = np.random.choice(compatible_functions)

    def _constant_mutation(self, individual: GPIndividual):
        """Mutate only constant terminal nodes."""
        nodes = individual.tree.get_all_nodes()
        constant_nodes = [
            node
            for node in nodes
            if node.is_terminal() and not isinstance(node.value, int)
        ]

        if constant_nodes:
            mutation_node = np.random.choice(constant_nodes)
            # Small perturbation to the constant
            perturbation = np.random.normal(0, 0.5)
            mutation_node.value = np.clip(mutation_node.value + perturbation, -10, 10)

    def _function_mutation(self, individual: GPIndividual):
        """Mutate only function nodes."""
        nodes = individual.tree.get_all_nodes()
        function_nodes = [node for node in nodes if not node.is_terminal()]

        if function_nodes:
            mutation_node = np.random.choice(function_nodes)
            current_arity = mutation_node.arity
            compatible_functions = [
                func
                for func, arity in individual.FUNCTION_SET.items()
                if arity == current_arity and func != mutation_node.value
            ]

            if compatible_functions:
                mutation_node.value = np.random.choice(compatible_functions)

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


class ParameterMutation(MutationOperator):
    """Parameter mutation that only changes constants and feature references."""

    def __call__(self, individual: GPIndividual) -> None:
        """Apply parameter mutation to the individual."""
        if np.random.random() < self.probability:
            nodes = individual.tree.get_all_nodes()
            terminal_nodes = [node for node in nodes if node.is_terminal()]

            if terminal_nodes:
                mutation_node = np.random.choice(terminal_nodes)

                if isinstance(mutation_node.value, int):
                    # Feature reference
                    mutation_node.value = np.random.randint(0, individual.n_features)
                else:
                    # Constant
                    # Gaussian perturbation
                    perturbation = np.random.normal(0, 1.0)
                    mutation_node.value = np.clip(
                        mutation_node.value + perturbation, -10, 10
                    )


class GrowMutation(MutationOperator):
    """Grow mutation that extends terminal nodes into subtrees."""

    def __init__(self, probability: float, max_depth: int = 2):
        super().__init__(probability)
        self.max_depth = max_depth

    def __call__(self, individual: GPIndividual) -> None:
        """Apply grow mutation to the individual."""
        if np.random.random() < self.probability:
            nodes = individual.tree.get_all_nodes()
            terminal_nodes = [node for node in nodes if node.is_terminal()]

            if terminal_nodes and individual.get_depth() < individual.max_depth:
                # Select a terminal node to grow
                growth_node = np.random.choice(terminal_nodes)

                # Choose a random function
                func_name = np.random.choice(list(individual.FUNCTION_SET.keys()), p=individual.probabilities)
                arity = individual.FUNCTION_SET[func_name]

                # Convert terminal to function
                old_value = growth_node.value
                growth_node.value = func_name
                growth_node.arity = arity

                # Add children
                growth_node.children = []
                for i in range(arity):
                    if i == 0:
                        # First child keeps the old terminal value
                        if isinstance(old_value, int):
                            child = Node(old_value, 0)
                        else:
                            child = Node(old_value, 0)
                    else:
                        # Other children are new random terminals
                        if np.random.random() < 0.8:
                            child = Node(np.random.randint(0, individual.n_features), 0)
                        else:
                            child = Node(np.random.uniform(-2, 2), 0)
                    growth_node.add_child(child)


class AdaptiveTreeMutation(MutationOperator):
    """Adaptive mutation that shifts from subtree to grow mutation using decay rate."""
    
    def __init__(self, probability: float, max_depth: int = 20, decay_rate: float = 0.95, 
                 reset_between_features: bool = True):
        super().__init__(probability)
        self.max_depth = max_depth
        self.decay_rate = decay_rate
        self.initial_subtree_prob = 0.9
        self.current_subtree_prob = self.initial_subtree_prob
        self.reset_between_features = reset_between_features
        
        # Create mutation operators
        self.subtree_mutation = SubtreeMutation(1.0, max_depth)
        self.grow_mutation = GrowMutation(1.0, max_depth)
        
    def reset(self):
        """Reset mutation probabilities to initial state."""
        if self.reset_between_features:
            self.current_subtree_prob = self.initial_subtree_prob
        
    def update_generation(self):
        """Update probabilities for next generation using decay."""
        self.current_subtree_prob *= self.decay_rate
        
    def __call__(self, individual: GPIndividual) -> None:
        """Apply adaptive mutation based on current probabilities."""
        if np.random.random() < self.probability:
            # Choose mutation type based on current probabilities
            if np.random.random() < self.current_subtree_prob:
                self.subtree_mutation(individual)
            else:
                self.grow_mutation(individual)
