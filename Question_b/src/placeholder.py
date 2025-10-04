"""Question B Module

This module provides a concrete optimization setup for Question B.
Since the original problem statement text is not available here, we
use a slightly more complex variant of OneMax: a weighted bitstring
maximization with a plateau penalty region to make search dynamics
different from the plain OneMax used in Question A.

Fitness definition (Weighted Plateau OneMax):
  Let w_i = 1 + (i mod 7)/10  (periodic fractional weights)
  Raw score = sum(w_i * bit_i)
  If raw score is within 5% of max but not equal to max, subtract a
  small penalty (to create a deceptive plateau encouraging exploration).

This gives GA opportunity to use mutation/crossover to escape near-optimal
plateau, while a narrow Beam Search may get stuck exploring local flips.

The actual GA & Beam Search engines are duplicated in sibling files
`genetic_algorithm.py` and `beam_search.py` inside this folder to keep
Question B self-contained.
"""

from typing import List

def weighted_plateau_fitness(chromosome: List[int]) -> float:
	length = len(chromosome)
	weights = [1 + (i % 7)/10 for i in range(length)]
	raw = sum(w * b for w, b in zip(weights, chromosome))
	max_raw = sum(weights)  # when all bits = 1
	# Plateau: if within 5% of max but not max, apply small penalty
	if raw >= 0.95 * max_raw and raw < max_raw:
		raw -= 0.5  # penalty
	return raw

# Helper to compute theoretical max (used in tests)
def max_weighted_plateau_score(length: int) -> float:
	weights = [1 + (i % 7)/10 for i in range(length)]
	return sum(weights)
