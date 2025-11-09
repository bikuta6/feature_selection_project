import random

import numpy as np
from tqdm import tqdm

from .individual import Individual


class NSGA2:
    def __init__(
        self,
        population_size,
        n_features,
        fitness_functions,
        crossover_operator,
        mutation_operator,
        crossover_prob=0.9,
    ):
        """
        Initialize NSGA-II algorithm.

        Args:
            population_size: Population size (should be even)
            n_features: Number of features
            fitness_functions: List of fitness functions [obj1, obj2]
            crossover_operator: Crossover operator
            mutation_operator: Mutation operator
            crossover_prob: Crossover probability
        """
        self.population_size = population_size
        self.n_features = n_features
        self.fitness_functions = fitness_functions
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.crossover_prob = crossover_prob

        self.population = []
        self.generation = 0

    def initialize_population(self):
        """
        STEP 1: Population initialization
        Create random initial population respecting constraints
        """
        self.population = []
        for _ in range(self.population_size):
            individual = Individual(self.n_features)
            self.population.append(individual)
        self.population[0].chromosome = np.ones(
            self.n_features, dtype=bool
        )  # All features selected for one individual

    def evaluate_population(self, model, X_train, y_train, cv=3):
        """
        STEP 2: Objective evaluation
        Calculate fitness values for each individual using cross-validation
        """
        for individual in self.population:
            # Evaluate each objective function
            for i, fitness_func in enumerate(self.fitness_functions):
                individual.objectives[i] = fitness_func(
                    individual, model, X_train, y_train, cv
                )

    def fast_non_dominated_sort(self, population):
        """
        STEP 3: Fast non-dominated sorting
        Classify individuals into Pareto fronts

        Returns:
            fronts: List of fronts [[front0], [front1], ...]
        """
        # Initialize structures
        domination_count = [0] * len(population)  # How many dominate each individual
        dominated_solutions = [
            [] for _ in range(len(population))
        ]  # Who each individual dominates
        fronts = [[]]  # List of fronts

        # For each individual
        for i, p in enumerate(population):
            for j, q in enumerate(population):
                if i != j:
                    if p.dominates(q):
                        # p dominates q
                        dominated_solutions[i].append(j)
                    elif q.dominates(p):
                        # q dominates p
                        domination_count[i] += 1

            # If not dominated by anyone, belongs to first front
            if domination_count[i] == 0:
                p.rank = 0
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for i in fronts[front_index]:
                # For each solution dominated by i
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    # If j is no longer dominated by anyone
                    if domination_count[j] == 0:
                        population[j].rank = front_index + 1
                        next_front.append(j)

            front_index += 1
            fronts.append(next_front)

        # Remove last empty front
        if not fronts[-1]:
            fronts.pop()

        return fronts

    def calculate_crowding_distance(self, front_indices, population):
        """
        STEP 4: Crowding distance calculation
        Calculate diversity within each front
        """
        if len(front_indices) <= 2:
            # If 2 or fewer solutions, assign maximum distance
            for i in front_indices:
                population[i].crowding_distance = float("inf")
            return

        # Initialize distances
        for i in front_indices:
            population[i].crowding_distance = 0

        # For each objective
        n_objectives = len(self.fitness_functions)
        for obj in range(n_objectives):
            # Sort by objective value
            front_indices.sort(key=lambda x: population[x].objectives[obj])

            # Find objective range
            obj_min = population[front_indices[0]].objectives[obj]
            obj_max = population[front_indices[-1]].objectives[obj]
            obj_range = obj_max - obj_min

            # Assign infinite distance to extremes
            population[front_indices[0]].crowding_distance = float("inf")
            population[front_indices[-1]].crowding_distance = float("inf")

            # Calculate distance for intermediate solutions
            if obj_range > 0:
                for i in range(1, len(front_indices) - 1):
                    idx = front_indices[i]
                    prev_idx = front_indices[i - 1]
                    next_idx = front_indices[i + 1]

                    distance = (
                        population[next_idx].objectives[obj]
                        - population[prev_idx].objectives[obj]
                    ) / obj_range
                    population[idx].crowding_distance += distance

    def environmental_selection(self, combined_population):
        """
        STEP 5: Environmental selection
        Select the best N solutions using dominance and diversity
        """
        # Non-dominated sorting
        fronts = self.fast_non_dominated_sort(combined_population)

        # Calculate crowding distances for each front
        for front_indices in fronts:
            if len(front_indices) > 0:
                self.calculate_crowding_distance(front_indices, combined_population)

        # Select individuals for new population
        new_population = []

        for front_indices in fronts:
            if len(new_population) + len(front_indices) <= self.population_size:
                # Add entire front
                for i in front_indices:
                    new_population.append(combined_population[i])
            else:
                # Partial front - select by crowding distance
                remaining_slots = self.population_size - len(new_population)

                # Sort by crowding distance (descending)
                front_indices.sort(
                    key=lambda x: combined_population[x].crowding_distance, reverse=True
                )

                # Take the best ones
                for i in range(remaining_slots):
                    new_population.append(combined_population[front_indices[i]])
                break

        return new_population

    def tournament_selection(self, population, tournament_size=2):
        """
        STEP 6: Tournament selection
        Select parents using dominance and diversity comparison
        """
        # Select random individuals for tournament
        tournament = random.sample(population, tournament_size)

        # Find best according to NSGA-II criteria
        best = tournament[0]
        for individual in tournament[1:]:
            if self.compare_individuals(individual, best):
                best = individual

        return best

    def compare_individuals(self, ind1, ind2):
        """
        Compare two individuals according to NSGA-II criteria
        Returns True if ind1 is better than ind2
        """
        # First compare by rank (lower is better)
        if ind1.rank < ind2.rank:
            return True
        elif ind1.rank > ind2.rank:
            return False

        # If same rank, compare by crowding distance (higher is better)
        return ind1.crowding_distance > ind2.crowding_distance

    def create_offspring(self, model, X_train, y_train, cv=3):
        """
        STEP 7: Offspring creation
        Generate new population through crossover and mutation using cross-validation
        """
        offspring = []

        while len(offspring) < self.population_size:
            # Parent selection
            parent1 = self.tournament_selection(self.population)
            parent2 = self.tournament_selection(self.population)

            # Crossover
            if random.random() < self.crossover_prob:
                child1, child2 = self.crossover_operator(parent1, parent2)
            else:
                # No crossover, copy parents
                child1 = Individual(self.n_features)
                child2 = Individual(self.n_features)
                child1.chromosome = parent1.chromosome.copy()
                child2.chromosome = parent2.chromosome.copy()

            # Mutation
            self.mutation_operator(child1)
            self.mutation_operator(child2)

            offspring.extend([child1, child2])

        # Adjust to exact population size
        offspring = offspring[: self.population_size]

        # Evaluate offspring
        for individual in offspring:
            for i, fitness_func in enumerate(self.fitness_functions):
                individual.objectives[i] = fitness_func(
                    individual, model, X_train, y_train, cv
                )

        return offspring

    def evolve(self, model, X_train, y_train, generations, cv=3):
        """
        MAIN NSGA-II ALGORITHM using cross-validation

        Algorithm steps:
        1. Initialize population P(0)
        2. Evaluate P(0) using cross-validation
        3. For each generation t:
           a. Create offspring Q(t) from P(t)
           b. Combine R(t) = P(t) ∪ Q(t)
           c. Sort R(t) by non-dominance
           d. Calculate crowding distances
           e. Select P(t+1) from R(t)
        """
        # STEP 1: Initialization
        self.initialize_population()

        # STEP 2: Initial evaluation using cross-validation
        self.evaluate_population(model, X_train, y_train, cv)

        # Evolution through generations
        progress_bar = tqdm(
            range(generations),
            desc="NSGA-II Evolution",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for gen in progress_bar:
            progress_bar.set_description(f"Gen {gen + 1}/{generations}")

            # STEP 3: Create offspring using cross-validation
            offspring = self.create_offspring(model, X_train, y_train, cv)

            # STEP 4: Combine populations
            combined_population = self.population + offspring

            # STEP 5: Environmental selection (includes sorting and distances)
            self.population = self.environmental_selection(combined_population)

            self.generation = gen + 1

            # Optional statistics for progress bar
            if (gen + 1) % 5 == 0 or gen == 0:
                front_0 = [ind for ind in self.population if ind.rank == 0]
                if front_0:
                    best_obj1 = max(ind.objectives[0] for ind in front_0)
                    best_obj2 = max(ind.objectives[1] for ind in front_0)
                    progress_bar.set_postfix(
                        {
                            "Front0": len(front_0),
                            "Best_Obj1": f"{best_obj1:.3f}",
                            "Best_Obj2": f"{best_obj2:.3f}",
                        }
                    )

        progress_bar.close()

    def print_stats(self):
        """Print statistics of current generation"""
        front_0 = [ind for ind in self.population if ind.rank == 0]
        print(f"  Front 0: {len(front_0)} individuals")

        if front_0:
            obj1_vals = [ind.objectives[0] for ind in front_0]
            obj2_vals = [ind.objectives[1] for ind in front_0]
            print(f"  Obj1 - Min: {min(obj1_vals):.3f}, Max: {max(obj1_vals):.3f}")
            print(f"  Obj2 - Min: {min(obj2_vals):.3f}, Max: {max(obj2_vals):.3f}")

    def get_pareto_front(self):
        """Return Pareto front solutions (rank 0)"""
        return [ind for ind in self.population if ind.rank == 0]

    def get_best_individual(self, objective_weights=None):
        """
        Return best individual according to objective weights
        If no weights given, use most balanced solution from Pareto front
        """
        pareto_front = self.get_pareto_front()

        if not pareto_front:
            return None

        # Scale front objectives to [0,1]
        obj1_vals = [ind.objectives[0] for ind in pareto_front]
        obj2_vals = [ind.objectives[1] for ind in pareto_front]
        min_obj1, max_obj1 = min(obj1_vals), max(obj1_vals)
        diff_obj1 = max_obj1 - min_obj1 if max_obj1 > min_obj1 else 1.0
        min_obj2, max_obj2 = min(obj2_vals), max(obj2_vals)
        diff_obj2 = max_obj2 - min_obj2 if max_obj2 > min_obj2 else 1.0

        if objective_weights is None:
            # Select most balanced solution (sum of objectives)
            best = max(
                pareto_front,
                key=lambda x: ((x.objectives[0] - min_obj1) / diff_obj1)
                + ((x.objectives[1] - min_obj2) / diff_obj2),
            )
        else:
            # Use specified weights
            best = max(
                pareto_front,
                key=lambda x: (
                    objective_weights[0] * (x.objectives[0] - min_obj1) / diff_obj1
                    + objective_weights[1] * (x.objectives[1] - min_obj2) / diff_obj2
                ),
            )

        return best
