import unittest
import sys
from pathlib import Path

# Ensure the Question_a/src directory is on path when running with embedded Python
root = Path(__file__).resolve().parents[1]
src_dir = root / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from genetic_algorithm import GeneticAlgorithm, sample_fitness
from beam_search import BeamSearch, example_expand

class TestAlgorithms(unittest.TestCase):
    def test_ga_improves(self):
        ga = GeneticAlgorithm(population_size=20, chromosome_length=10, fitness_fn=sample_fitness, seed=1)
        best_ch, best_fit, history = ga.run(generations=20)
        # Best should be at least near chromosome_length due to maximize ones
        self.assertGreaterEqual(best_fit, 7)
        self.assertEqual(len(history), 20)

    def test_beam_search(self):
        start = [0]*8
        beam = BeamSearch(beam_width=2, expand_fn=example_expand, score_fn=sum)
        best_state, best_score = beam.search(start, depth=5)
        # Cannot exceed total length 8
        self.assertLessEqual(best_score, 8)
        self.assertGreater(best_score, 0)

if __name__ == '__main__':
    unittest.main()
