from multiprocessing import Pool, cpu_count
from typing import Callable, List, Optional

import numpy as np
import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from .crossover import CrossoverOperator, SubtreeCrossover
from .individual import GPIndividual
from .mutation import MutationOperator, SubtreeMutation


def _evaluate_fitness_worker(args):
    """
    Worker function for multiprocessing fitness evaluation.

    Args:
        args: Tuple containing (individual_data, X, y, predictor_model, cv, baseline_mae)

    Returns:
        Fitness value (float)
    """
    individual, X, y, predictor_model, cv = args

    try:
        # Generate synthesized feature for the whole dataset
        new_feature = individual.evaluate_feature(X)

        # Check for invalid values
        if not np.isfinite(new_feature).all():
            return 0.0

        # Create augmented feature matrix
        X_aug = np.column_stack([X, new_feature])

        # Evaluate model with augmented features using cross-validation (R2)
        try:
            scores = cross_val_score(predictor_model, X_aug, y, cv=cv, scoring="r2")
            r2 = np.mean(scores)
        except Exception:
            # Fallback: fit on entire data
            model = type(predictor_model)(**predictor_model.get_params())
            model.fit(X_aug, y)
            r2 = model.score(X_aug, y)

        # Calculate fitness from R2
        fitness = r2

        # Add parsimony pressure (prefer smaller trees)
        complexity_penalty = 0.001 * individual.get_size()
        fitness -= complexity_penalty

        return max(fitness, 0.0)  # Ensure non-negative fitness

    except Exception as e:
        # Return zero fitness for invalid individuals
        return 0.0


class SimpleGA:
    """
    Simple genetic algorithm for feature synthesis using genetic programming.
    Optimizes (1 - (mae / baseline_mae)) as maximization objective.
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 50,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        tournament_size: int = 3,
        max_depth: int = 6,
        crossover_operator: Optional[CrossoverOperator] = None,
        mutation_operator: Optional[MutationOperator] = None,
        elitism: bool = True,
        verbose: bool = True,
        use_multiprocessing: bool = False,
        n_jobs: int = 1,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        self.elitism = elitism
        self.verbose = verbose
        self.use_multiprocessing = use_multiprocessing
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()

        # Default operators
        self.crossover_operator = crossover_operator or SubtreeCrossover(max_depth)
        self.mutation_operator = mutation_operator or SubtreeMutation(
            mutation_prob, max_depth // 2
        )

        # Evolution tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.baseline_mae = None

    def evolve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 3,
        predictor_model: Optional[Callable] = None,
    ) -> GPIndividual:
        """
        Evolve a population to find the best feature synthesis expression.

        Args:
            X: Input features
            y: Target variable
            cv: Number of cross-validation folds to use for evaluation
            predictor_model: Model to use for prediction (default: RandomForestRegressor)

        Returns:
            Best individual found
        """
        if predictor_model is None:
            predictor_model = LinearRegression()

        # Calculate baseline MAE using cross-validation on the whole dataset
        try:
            scores = cross_val_score(
                predictor_model, X, y, cv=cv, scoring="neg_mean_absolute_error"
            )
            self.baseline_mae = -np.mean(scores)
        except Exception:
            # Fallback: fit on entire data and compute MAE on predictions
            baseline_model = type(predictor_model)(**predictor_model.get_params())
            baseline_model.fit(X, y)
            baseline_pred = baseline_model.predict(X)
            self.baseline_mae = mean_absolute_error(y, baseline_pred)

        if self.verbose:
            print(f"Baseline MAE: {self.baseline_mae:.6f}")
            if self.use_multiprocessing:
                print(f"Using multiprocessing with {self.n_jobs} processes")

        # Initialize population
        population = self._initialize_population(X.shape[1])

        # Evaluate initial population
        self._evaluate_population_fitness(population, X, y, predictor_model, cv)

        # Evolution loop
        progress_bar = tqdm.tqdm(
            range(self.max_generations),
            desc="Evolving Features",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for generation in progress_bar:
            progress_bar.set_description(f"Gen {generation + 1}/{self.max_generations}")
            # Track statistics
            fitnesses = [ind.fitness for ind in population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)

            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Update adaptive mutation if using it
            if hasattr(self.mutation_operator, 'update_generation'):
                self.mutation_operator.update_generation()

            # Update best individual
            best_idx = np.argmax(fitnesses)
            if (
                self.best_individual is None
                or population[best_idx].fitness > self.best_individual.fitness
            ):
                self.best_individual = population[best_idx].copy()

            # Create new population
            new_population = []

            # Elitism: keep best individual
            if self.elitism:
                new_population.append(population[best_idx].copy())

            # Generate offspring
            offspring_to_evaluate = []
            while (
                len(new_population) + len(offspring_to_evaluate) < self.population_size
            ):
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                if np.random.random() < self.crossover_prob:
                    offspring1, offspring2 = self.crossover_operator(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()

                # Mutation
                self.mutation_operator(offspring1)
                self.mutation_operator(offspring2)

                offspring_to_evaluate.extend([offspring1, offspring2])

            # Trim offspring to exact size needed
            offspring_to_evaluate = offspring_to_evaluate[
                : self.population_size - len(new_population)
            ]

            # Evaluate offspring fitness
            self._evaluate_population_fitness(
                offspring_to_evaluate, X, y, predictor_model, cv
            )

            # Add evaluated offspring to new population
            new_population.extend(offspring_to_evaluate)

            # Trim population to exact size
            population = new_population[: self.population_size]

        if self.verbose:
            print(
                f"Evolution completed. Best fitness: {self.best_individual.fitness:.6f}"
            )
            print(f"Best expression: {self.best_individual}")

        return self.best_individual

    def _initialize_population(self, n_features: int) -> List[GPIndividual]:
        """Initialize a random population."""
        population = []
        for _ in range(self.population_size):
            individual = GPIndividual(n_features, self.max_depth)
            population.append(individual)
        return population

    def _evaluate_population_fitness(
        self,
        population: List[GPIndividual],
        X: np.ndarray,
        y: np.ndarray,
        predictor_model: Callable,
        cv: int = 3,
    ):
        """
        Evaluate fitness for a population of individuals.

        Args:
            population: List of individuals to evaluate
            X: Input features
            y: Target variable
            predictor_model: Model to use for prediction
            cv: Number of cross-validation folds
        """
        if self.use_multiprocessing and len(population) > 1:
            # Prepare arguments for multiprocessing
            args_list = []
            for individual in population:
                args_list.append((individual, X, y, predictor_model, cv))

            # Use multiprocessing to evaluate fitness
            try:
                with Pool(processes=self.n_jobs) as pool:
                    fitnesses = pool.map(_evaluate_fitness_worker, args_list)

                # Assign fitness values back to individuals
                for individual, fitness in zip(population, fitnesses):
                    individual.fitness = fitness

            except Exception as e:
                if self.verbose:
                    print(
                        f"Multiprocessing failed ({e}), falling back to sequential evaluation"
                    )
                # Fallback to sequential evaluation
                for individual in population:
                    individual.fitness = self._evaluate_fitness(
                        individual, X, y, predictor_model, cv
                    )
        else:
            # Sequential evaluation
            for individual in population:
                individual.fitness = self._evaluate_fitness(
                    individual, X, y, predictor_model, cv
                )

    def _evaluate_fitness(
        self,
        individual: GPIndividual,
        X: np.ndarray,
        y: np.ndarray,
        predictor_model: Callable,
        cv: int = 3,
    ) -> float:
        """
        Evaluate fitness of an individual using R².

        Returns:
            Fitness value (higher is better)
        """
        try:
            # Generate synthesized feature for the whole dataset
            new_feature = individual.evaluate_feature(X)

            # Check for invalid values
            if not np.isfinite(new_feature).all():
                return 0.0

            # Create augmented feature matrix
            X_aug = np.column_stack([X, new_feature])

            # Evaluate model with augmented features using cross-validation (R2)
            try:
                scores = cross_val_score(predictor_model, X_aug, y, cv=cv, scoring="r2")
                r2 = np.mean(scores)
            except Exception:
                # Fallback: fit on entire data
                model = type(predictor_model)(**predictor_model.get_params())
                model.fit(X_aug, y)
                r2 = model.score(X_aug, y)

            # Calculate fitness from R2
            fitness = r2

            # Add parsimony pressure (prefer smaller trees)
            complexity_penalty = 0.001 * individual.get_size()
            fitness -= complexity_penalty

            return max(fitness, 0.0)  # Ensure non-negative fitness

        except Exception as e:
            # Return zero fitness for invalid individuals
            return 0.0

    def _tournament_selection(self, population: List[GPIndividual]) -> GPIndividual:
        """Select an individual using tournament selection."""
        tournament = np.random.choice(population, self.tournament_size, replace=False)
        return max(tournament, key=lambda ind: ind.fitness).copy()

    def get_statistics(self) -> dict:
        """Get evolution statistics."""
        return {
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "best_individual": self.best_individual,
            "baseline_mae": self.baseline_mae,
            "final_improvement": (
                self.best_individual.fitness if self.best_individual else 0.0
            ),
        }

    def create_enhanced_dataset(self, X: np.ndarray) -> np.ndarray:
        """
        Create an enhanced dataset by adding the synthesized feature.

        Args:
            X: Original feature matrix

        Returns:
            Enhanced feature matrix with synthesized feature
        """
        if self.best_individual is None:
            raise ValueError("No best individual found. Run evolve() first.")

        new_feature = self.best_individual.evaluate_feature(X)

        # Check for invalid values
        if not np.isfinite(new_feature).all():
            print(
                "Warning: Synthesized feature contains invalid values. Using original features."
            )
            return X

        return np.column_stack([X, new_feature])

    def __str__(self) -> str:
        """String representation of the MultiFeatureGA configuration."""
        return (
            f"MultiFeatureGA(\n"
            f"  population_size={self.population_size},\n"
            f"  max_generations={self.max_generations},\n"
            f"  crossover_prob={self.crossover_prob},\n"
            f"  mutation_prob={self.mutation_prob},\n"
            f"  tournament_size={self.tournament_size},\n"
            f"  max_depth={self.max_depth},\n"
            f"  crossover_operator={type(self.crossover_operator).__name__},\n"
            f"  mutation_operator={type(self.mutation_operator).__name__},\n"
            f"  elitism={self.elitism},\n"
            f"  use_multiprocessing={self.use_multiprocessing},\n"
            f"  n_jobs={self.n_jobs}\n"
            f")"
        )


class MultiFeatureGA(SimpleGA):
    """
    Extended GA that can evolve multiple features simultaneously.
    """

    def __init__(self, n_features_to_create: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.n_features_to_create = n_features_to_create
        self.best_features = []

    def evolve_multiple_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 3,
        predictor_model: Optional[Callable] = None,
    ) -> List[GPIndividual]:
        """
        Evolve multiple features sequentially.

        Returns:
            List of best individuals for each feature
        """
        if predictor_model is None:
            predictor_model = LinearRegression()

        current_X = X.copy()
        best_features = []

        for i in range(self.n_features_to_create):
            if self.verbose:
                print(f"\n=== Evolving feature {i + 1}/{self.n_features_to_create} ===")

            # Evolve one feature using cross-validation
            best_feature = self.evolve(current_X, y, cv, predictor_model)
            best_features.append(best_feature)

            # Add the new feature to the dataset for next iteration
            new_feature_values = best_feature.evaluate_feature(current_X)
            if np.isfinite(new_feature_values).all():
                current_X = np.column_stack([current_X, new_feature_values])
                if self.verbose:
                    print(f"Feature {i + 1} added: {best_feature}")
            else:
                if self.verbose:
                    print(f"Feature {i + 1} invalid, skipping")

        self.best_features = best_features
        return best_features

    def create_multi_enhanced_dataset(self, X: np.ndarray) -> np.ndarray:
        """
        Create dataset enhanced with multiple synthesized features.

        Args:
            X: Original feature matrix

        Returns:
            Enhanced feature matrix with all synthesized features
        """
        if not self.best_features:
            raise ValueError(
                "No features evolved. Run evolve_multiple_features() first."
            )

        enhanced_X = X.copy()

        for i, feature in enumerate(self.best_features):
            new_feature = feature.evaluate_feature(enhanced_X)
            if np.isfinite(new_feature).all():
                enhanced_X = np.column_stack([enhanced_X, new_feature])
            else:
                print(
                    f"Warning: Synthesized feature {i + 1} contains invalid values, skipping."
                )

        return enhanced_X

    def __str__(self) -> str:
        """String representation of the MultiFeatureGA configuration."""
        return (
            f"MultiFeatureGA(\n"
            f"  population_size={self.population_size},\n"
            f"  max_generations={self.max_generations},\n"
            f"  crossover_prob={self.crossover_prob},\n"
            f"  mutation_prob={self.mutation_prob},\n"
            f"  tournament_size={self.tournament_size},\n"
            f"  max_depth={self.max_depth},\n"
            f"  crossover_operator={type(self.crossover_operator).__name__},\n"
            f"  mutation_operator={type(self.mutation_operator).__name__},\n"
            f"  elitism={self.elitism},\n"
            f"  use_multiprocessing={self.use_multiprocessing},\n"
            f"  n_jobs={self.n_jobs},\n"
            f"  n_features_to_create={self.n_features_to_create}\n"
            f")"
        )
