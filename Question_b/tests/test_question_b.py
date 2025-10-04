import sys
from pathlib import Path
import unittest

# Adjust path to import src modules for embedded python scenario
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from placeholder import weighted_plateau_fitness, max_weighted_plateau_score
from genetic_algorithm import GeneticAlgorithm
from beam_search import BeamSearch, bit_flip_expand

class TestQuestionB(unittest.TestCase):
    def test_weighted_plateau_fitness_penalty(self):
        length = 20
        # Construct a chromosome near optimal (all ones except one zero) triggering plateau penalty
        near_opt = [1]*(length-1) + [0]
        full_opt = [1]*length
        near_score = weighted_plateau_fitness(near_opt)
        full_score = weighted_plateau_fitness(full_opt)
        self.assertLess(full_score, max_weighted_plateau_score(length)+0.0001)  # sanity
        self.assertLess(near_score, full_score)
        self.assertGreater(near_score, 0)

    def test_ga_reaches_or_nears_optimum(self):
        length = 25
        ga = GeneticAlgorithm(population_size=40, chromosome_length=length, fitness_fn=weighted_plateau_fitness, seed=7)
        best_ch, best_fit, history = ga.run(generations=60)
        theoretical = max_weighted_plateau_score(length)
        # allow small gap if penalty plateau encountered
        self.assertGreaterEqual(best_fit, theoretical - 1.0)
        self.assertTrue(len(history) == 60)

    def test_beam_progress(self):
        import random
        length = 18
        start = [random.randint(0,1) for _ in range(length)]
        beam = BeamSearch(beam_width=4, expand_fn=bit_flip_expand, score_fn=weighted_plateau_fitness)
        best_state, best_score = beam.search(start, depth=25)
        self.assertGreater(best_score, weighted_plateau_fitness(start))

if __name__ == '__main__':
    unittest.main()
