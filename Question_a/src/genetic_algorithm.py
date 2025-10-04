import random
from typing import List, Callable, Tuple

# Generic Genetic Algorithm scaffold
class GeneticAlgorithm:
    def __init__(self,
                 population_size: int,
                 chromosome_length: int,
                 fitness_fn: Callable[[List[int]], float],
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.01,
                 tournament_size: int = 3,
                 maximize: bool = True,
                 seed: int | None = None):
        if seed is not None:
            random.seed(seed)
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.fitness_fn = fitness_fn
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.maximize = maximize
        self.population: List[List[int]] = []
        self.fitness_scores: List[float] = []

    def _random_chromosome(self) -> List[int]:
        return [random.randint(0, 1) for _ in range(self.chromosome_length)]

    def _init_population(self):
        self.population = [self._random_chromosome() for _ in range(self.population_size)]
        self.fitness_scores = [self.fitness_fn(ch) for ch in self.population]

    def _tournament_select(self) -> List[int]:
        competitors = random.sample(list(zip(self.population, self.fitness_scores)), k=self.tournament_size)
        competitors.sort(key=lambda x: x[1], reverse=self.maximize)
        return competitors[0][0][:]

    def _single_point_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate or self.chromosome_length < 2:
            return p1[:], p2[:]
        point = random.randint(1, self.chromosome_length - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def _mutate(self, chromosome: List[int]):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]

    def best(self) -> Tuple[List[int], float]:
        idx = max(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i]) if self.maximize else \
              min(range(len(self.fitness_scores)), key=lambda i: self.fitness_scores[i])
        return self.population[idx], self.fitness_scores[idx]

    def run(self, generations: int) -> Tuple[List[int], float, list]:
        self._init_population()
        history = []
        for g in range(generations):
            new_pop: List[List[int]] = []
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._single_point_crossover(p1, p2)
                self._mutate(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c1)
                self._mutate(c2)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)
            self.population = new_pop
            self.fitness_scores = [self.fitness_fn(ch) for ch in self.population]
            b_ch, b_fit = self.best()
            history.append(b_fit)
        # Explicitly unpack best solution to avoid starred-expression return (improves clarity)
        best_ch, best_fit = self.best()
        return best_ch, best_fit, history  # (chromosome, fitness, history)

# Example placeholder fitness function (count ones)
def sample_fitness(chromosome: List[int]) -> float:
    return sum(chromosome)

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=30, chromosome_length=20, fitness_fn=sample_fitness, seed=42)
    best_ch, best_fit, history = ga.run(generations=50)
    print("Best:", best_ch, best_fit)
