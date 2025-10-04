import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from placeholder import fitness, brute_force_optimum, chromosome_length, is_feasible
from genetic_algorithm import GeneticAlgorithm
from beam_search import BeamSearchKnapsack

class TestQuestionC(unittest.TestCase):
    def test_bruteforce_optimum(self):
        best_val, chrom = brute_force_optimum()
        self.assertTrue(is_feasible(chrom))
        # Sanity: best value should be positive and chromosome has correct length
        self.assertGreater(best_val, 0)
        self.assertEqual(len(chrom), chromosome_length())

    def test_ga_approaches_optimum(self):
        optimal_val, _ = brute_force_optimum()
        ga = GeneticAlgorithm(population_size=80, chromosome_length=chromosome_length(), fitness_fn=fitness, seed=99)
        best_ch, best_fit, history = ga.run(generations=150)
        # Accept within 5% of optimum for stochastic run
        self.assertGreaterEqual(best_fit, 0.95 * optimal_val)
        self.assertEqual(len(history), 150)

    def test_beam_finds_feasible(self):
        optimal_val, _ = brute_force_optimum()
        beam = BeamSearchKnapsack(beam_width=32)
        best_state, best_score = beam.search(depth=chromosome_length()+5)
        self.assertTrue(is_feasible(best_state))
        self.assertGreaterEqual(best_score, 0.90 * optimal_val)

if __name__ == '__main__':
    unittest.main()