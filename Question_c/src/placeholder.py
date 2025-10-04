"""Question C: 0/1 Knapsack Optimization (GA vs Beam Search)

We model a classic 0/1 Knapsack problem:
  - Each item i has value v_i and weight w_i.
  - Choose a subset of items maximizing total value subject to total weight <= capacity.

Dataset: A modest 15-item instance chosen so that the exact optimum can be brute-forced (2^15 = 32768).
This allows unit tests to validate solution quality.

Chromosome Representation (GA):
  Binary list of length N (1 = take item, 0 = skip item).

Fitness Function:
  If total_weight <= capacity: fitness = total_value
  Else: fitness = total_value - PENALTY * (total_weight - capacity)
  (A linear penalty keeps landscape smoother for GA.)

Beam Search Representation:
  Partial decisions (prefix bit list) expanded depth-wise: at depth d, we decide inclusion for item d.
  Score for partial state uses an optimistic bound = current value + sum of remaining items with best
  value/weight ratio (greedy filling) until capacity is exceeded; this guides beam retention.

Outputs:
  GA best solution value & chromosome, Beam search best feasible solution & value.

Assumption: Actual assignment's Question C requires a constrained optimization; if different, adapt
these scaffolds accordingly.
"""

from __future__ import annotations
from typing import List, Tuple

VALUES = [10, 5, 15, 7, 6, 18, 3, 12, 14, 9, 11, 8, 4, 13, 16]
WEIGHTS = [ 2, 3,  5, 7, 1,  4, 1,  6,  3, 5,  7, 2, 1,  4,  5]
CAPACITY = 25
PENALTY = 5  # penalty per unit overweight

def evaluate_solution(chrom: List[int]) -> Tuple[int, int]:
	value = sum(v for v, bit in zip(VALUES, chrom) if bit)
	weight = sum(w for w, bit in zip(WEIGHTS, chrom) if bit)
	return value, weight

def fitness(chrom: List[int]) -> float:
	value, weight = evaluate_solution(chrom)
	if weight <= CAPACITY:
		return float(value)
	return float(value - PENALTY * (weight - CAPACITY))

def is_feasible(chrom: List[int]) -> bool:
	return sum(w for w, b in zip(WEIGHTS, chrom) if b) <= CAPACITY

def brute_force_optimum() -> Tuple[int, List[int]]:
	n = len(VALUES)
	best_val = -1
	best_chrom: List[int] | None = None
	for mask in range(1 << n):
		chrom = [(mask >> i) & 1 for i in range(n)]
		val, weight = evaluate_solution(chrom)
		if weight <= CAPACITY and val > best_val:
			best_val = val
			best_chrom = chrom
	return best_val, best_chrom or [0]*n

def optimistic_bound(partial: List[int]) -> float:
	"""Compute optimistic bound for a partial decision vector.
	partial length k indicates decisions for first k items.
	Remaining items are greedily added by value/weight ratio until capacity.
	"""
	k = len(partial)
	chosen_value = sum(v for v, bit in zip(VALUES[:k], partial) if bit)
	chosen_weight = sum(w for w, bit in zip(WEIGHTS[:k], partial) if bit)
	remaining_capacity = CAPACITY - chosen_weight
	if remaining_capacity < 0:
		# Overweight partial: pessimistic; still allow but bound reduced
		return chosen_value - PENALTY * (-remaining_capacity)
	# Build list of remaining items with ratios
	items = []
	for i in range(k, len(VALUES)):
		items.append((VALUES[i]/WEIGHTS[i], VALUES[i], WEIGHTS[i]))
	items.sort(reverse=True)
	bound = chosen_value
	cap = remaining_capacity
	for ratio, val, wt in items:
		if wt <= cap:
			bound += val
			cap -= wt
		else:
			# fractional for bound only
			bound += ratio * cap
			break
	return bound

def chromosome_length() -> int:
	return len(VALUES)
