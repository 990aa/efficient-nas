import numpy as np
from typing import List, Tuple, Dict, Any, Callable


class NSGA2:
    """NSGA-II multi-objective optimization algorithm implemented from scratch."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 60,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.3,
        objectives: List[str] = ["accuracy", "latency", "params"],
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.objectives = objectives
        self.num_objectives = len(objectives)

        # Optimization history
        self.history = {
            "generation": [],
            "population": [],
            "fitness": [],
            "pareto_front": [],
        }

    def fast_non_dominated_sort(self, fitness: np.ndarray) -> List[List[int]]:
        """
        Fast non-dominated sorting with O(M*N^2) complexity.

        Args:
            fitness: Array of shape (N, M) where N is population size, M is objectives

        Returns:
            List of fronts, where each front contains indices of individuals
        """
        N = fitness.shape[0]

        # Dominance relationships
        S = [[] for _ in range(N)]  # Individuals dominated by i
        n = np.zeros(N, dtype=int)  # Number of individuals that dominate i
        rank = np.zeros(N, dtype=int)

        fronts = [[]]

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                if self._dominates(fitness[i], fitness[j]):
                    S[i].append(j)
                elif self._dominates(fitness[j], fitness[i]):
                    n[i] += 1

            if n[i] == 0:
                rank[i] = 0
                fronts[0].append(i)

        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for individual_idx in fronts[i]:
                for dominated_idx in S[individual_idx]:
                    n[dominated_idx] -= 1
                    if n[dominated_idx] == 0:
                        rank[dominated_idx] = i + 1
                        next_front.append(dominated_idx)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 dominates obj2.

        For minimization: obj1 <= obj2 in all objectives and < in at least one
        For maximization: obj1 >= obj2 in all objectives and > in at least one
        """
        # Objectives: [accuracy (max), latency (min), params (min)]
        better_in_any = False

        for i, obj_name in enumerate(self.objectives):
            if obj_name == "accuracy":
                # Maximization
                if obj1[i] < obj2[i]:
                    return False
                if obj1[i] > obj2[i]:
                    better_in_any = True
            else:
                # Minimization (latency, params)
                if obj1[i] > obj2[i]:
                    return False
                if obj1[i] < obj2[i]:
                    better_in_any = True

        return better_in_any

    def crowding_distance(self, fitness: np.ndarray, front: List[int]) -> np.ndarray:
        """
        Calculate crowding distance for individuals in a Pareto front.

        Args:
            fitness: Fitness matrix of shape (N, M)
            front: List of indices in the current front

        Returns:
            Crowding distances for individuals in the front
        """
        if len(front) == 0:
            return np.array([])

        distances = np.zeros(len(front))
        front_fitness = fitness[front]
        num_objectives = front_fitness.shape[1]

        for obj_idx in range(num_objectives):
            # Sort by current objective
            sorted_indices = np.argsort(front_fitness[:, obj_idx])
            sorted_fitness = front_fitness[sorted_indices]

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Normalize objective range
            obj_range = sorted_fitness[-1, obj_idx] - sorted_fitness[0, obj_idx]
            if obj_range == 0:
                continue

            # Calculate distances for intermediate points
            for i in range(1, len(front) - 1):
                if self.objectives[obj_idx] == "accuracy":
                    # For maximization, neighbors are reversed
                    neighbor_diff = (
                        sorted_fitness[i - 1, obj_idx] - sorted_fitness[i + 1, obj_idx]
                    )
                else:
                    # For minimization
                    neighbor_diff = (
                        sorted_fitness[i + 1, obj_idx] - sorted_fitness[i - 1, obj_idx]
                    )

                distances[sorted_indices[i]] += abs(neighbor_diff) / obj_range

        return distances

    def binary_tournament_selection(
        self, population: List[Any], fitness: np.ndarray, fronts: List[List[int]]
    ) -> List[Any]:
        """
        Binary tournament selection based on Pareto rank and crowding distance.
        """
        selected = []
        N = len(population)

        # Precompute crowding distances for all fronts
        crowding_distances = np.zeros(N)
        for front in fronts:
            if len(front) > 0:
                front_distances = self.crowding_distance(fitness, front)
                for idx, distance in zip(front, front_distances):
                    crowding_distances[idx] = distance

        # Precompute ranks
        ranks = np.zeros(N, dtype=int)
        for rank_idx, front in enumerate(fronts):
            for individual_idx in front:
                ranks[individual_idx] = rank_idx

        for _ in range(N):
            # Randomly select two individuals
            idx1, idx2 = np.random.choice(N, 2, replace=False)

            # Select based on rank and crowding distance
            if ranks[idx1] < ranks[idx2]:
                selected.append(population[idx1])
            elif ranks[idx1] > ranks[idx2]:
                selected.append(population[idx2])
            else:
                # Same rank, select based on crowding distance
                if crowding_distances[idx1] > crowding_distances[idx2]:
                    selected.append(population[idx1])
                else:
                    selected.append(population[idx2])

        return selected

    def evolve(
        self, population: List[Any], evaluate_func: Callable
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Main evolution loop for NSGA-II.

        Args:
            population: Initial population of architectures
            evaluate_func: Function that evaluates architectures and returns fitness

        Returns:
            Final population and optimization history
        """
        print("Starting NSGA-II evolution...")

        # Evaluate initial population
        print("Evaluating initial population...")
        fitness = np.array([evaluate_func(ind) for ind in population])

        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")

            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(fitness)

            # Selection
            selected = self.binary_tournament_selection(population, fitness, fronts)

            # Create offspring through crossover and mutation
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]

                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self.crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    offspring.extend([child1, child2])

            # Evaluate offspring
            print(f"Evaluating {len(offspring)} offspring...")
            offspring_fitness = np.array([evaluate_func(ind) for ind in offspring])

            # Combine parent and offspring populations
            combined_population = population + offspring
            combined_fitness = np.vstack([fitness, offspring_fitness])

            # Select new population using NSGA-II selection
            new_population, new_fitness = self._select_new_population(
                combined_population, combined_fitness
            )

            population, fitness = new_population, new_fitness

            # Record history
            self._update_history(population, fitness, gen)

            # Print current Pareto front stats
            self._print_generation_stats(fitness, gen)

        return population, self.history

    def _select_new_population(
        self, population: List[Any], fitness: np.ndarray
    ) -> Tuple[List[Any], np.ndarray]:
        """Select new population using NSGA-II mechanism."""
        fronts = self.fast_non_dominated_sort(fitness)

        new_population = []
        new_fitness = []

        current_front = 0
        while len(new_population) + len(fronts[current_front]) <= self.population_size:
            # Add entire front
            for idx in fronts[current_front]:
                new_population.append(population[idx])
                new_fitness.append(fitness[idx])
            current_front += 1

        # If we need more individuals, use crowding distance
        if len(new_population) < self.population_size:
            remaining_slots = self.population_size - len(new_population)
            last_front = fronts[current_front]

            # Calculate crowding distances for the last front
            crowding_distances = self.crowding_distance(fitness, last_front)

            # Select individuals with highest crowding distance
            selected_indices = np.argsort(crowding_distances)[-remaining_slots:]

            for idx in selected_indices:
                new_population.append(population[last_front[idx]])
                new_fitness.append(fitness[last_front[idx]])

        return new_population, np.array(new_fitness)

    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover between two parent genomes."""
        # This will be implemented in genetic_operators.py
        from .genetic_operators import crossover_genomes

        return crossover_genomes(parent1, parent2)

    def mutate(self, individual: Dict) -> Dict:
        """Mutate an individual genome."""
        from .genetic_operators import mutate_genome

        return mutate_genome(individual, self.mutation_rate)

    def _update_history(
        self, population: List[Any], fitness: np.ndarray, generation: int
    ):
        """Update optimization history."""
        self.history["generation"].append(generation)
        self.history["population"].append(population.copy())
        self.history["fitness"].append(fitness.copy())

        # Find current Pareto front
        fronts = self.fast_non_dominated_sort(fitness)
        pareto_indices = fronts[0] if fronts else []
        pareto_front = {
            "individuals": [population[i] for i in pareto_indices],
            "fitness": fitness[pareto_indices],
        }
        self.history["pareto_front"].append(pareto_front)

    def _print_generation_stats(self, fitness: np.ndarray, generation: int):
        """Print statistics for current generation."""
        fronts = self.fast_non_dominated_sort(fitness)
        pareto_front = fronts[0] if fronts else []

        if len(pareto_front) > 0:
            pareto_fitness = fitness[pareto_front]

            print(f"  Pareto front size: {len(pareto_front)}")
            print(
                f"  Accuracy range: {pareto_fitness[:, 0].min():.3f} - {pareto_fitness[:, 0].max():.3f}"
            )
            print(
                f"  Latency range: {pareto_fitness[:, 1].min():.1f} - {pareto_fitness[:, 1].max():.1f} ms"
            )
            print(
                f"  Params range: {pareto_fitness[:, 2].min():.1f} - {pareto_fitness[:, 2].max():.1f}M"
            )
            print()


class PopulationManager:
    """Manages population and fitness evaluation."""

    def __init__(self, encoder, evaluator):
        self.encoder = encoder
        self.evaluator = evaluator

    def evaluate_population(
        self, population: List[Dict], generation: int, use_proxies: bool = True
    ) -> np.ndarray:
        """
        Evaluate population based on generation strategy.

        Args:
            population: List of architecture genomes
            generation: Current generation number
            use_proxies: Whether to use zero-cost proxies

        Returns:
            Fitness matrix of shape (N, 3) for [accuracy, latency, params]
        """
        fitness = np.zeros((len(population), 3))

        if generation < 30 and use_proxies:
            # Early generations: zero-cost proxies only
            print("Using zero-cost proxies for evaluation...")
            for i, individual in enumerate(population):
                scores = self.evaluator.evaluate_with_proxies(individual)
                fitness[i] = [
                    scores["accuracy_proxy"],
                    scores["latency"],
                    scores["params"],
                ]

        elif generation < 50:
            # Mid generations: quick training for top 20%
            print("Using quick training for top architectures...")
            proxy_scores = []
            for individual in population:
                scores = self.evaluator.evaluate_with_proxies(individual)
                proxy_scores.append(scores["accuracy_proxy"])

            # Select top 20% for quick training
            top_indices = np.argsort(proxy_scores)[-len(population) // 5 :]

            for i, individual in enumerate(population):
                if i in top_indices:
                    scores = self.evaluator.evaluate_quick_training(individual)
                    fitness[i] = [
                        scores["accuracy"],
                        scores["latency"],
                        scores["params"],
                    ]
                else:
                    scores = self.evaluator.evaluate_with_proxies(individual)
                    fitness[i] = [
                        scores["accuracy_proxy"],
                        scores["latency"],
                        scores["params"],
                    ]

        else:
            # Final generations: full training for top architectures
            print("Using full training for top architectures...")
            # Only evaluate a subset to save computation
            if len(population) > 10:
                # Use proxies to select top candidates
                proxy_scores = []
                for individual in population:
                    scores = self.evaluator.evaluate_with_proxies(individual)
                    proxy_scores.append(scores["accuracy_proxy"])

                top_indices = np.argsort(proxy_scores)[-10:]

                for i, individual in enumerate(population):
                    if i in top_indices:
                        scores = self.evaluator.evaluate_full_training(individual)
                        fitness[i] = [
                            scores["accuracy"],
                            scores["latency"],
                            scores["params"],
                        ]
                    else:
                        # For others, use proxy scores
                        scores = self.evaluator.evaluate_with_proxies(individual)
                        fitness[i] = [
                            scores["accuracy_proxy"],
                            scores["latency"],
                            scores["params"],
                        ]
            else:
                for i, individual in enumerate(population):
                    scores = self.evaluator.evaluate_full_training(individual)
                    fitness[i] = [
                        scores["accuracy"],
                        scores["latency"],
                        scores["params"],
                    ]

        return fitness
