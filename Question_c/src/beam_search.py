from __future__ import annotations
from typing import List, Tuple, Callable
import heapq
from placeholder import optimistic_bound, chromosome_length, fitness, is_feasible

class BeamSearchKnapsack:
    def __init__(self, beam_width: int):
        self.beam_width = beam_width

    def expand(self, partial: List[int]) -> List[List[int]]:
        if len(partial) >= chromosome_length():
            return []
        return [partial + [0], partial + [1]]

    def search(self, depth: int) -> Tuple[List[int], float]:
        # Each state: (negative optimistic bound for max-heap via min-heap, partial_bits)
        start: List[int] = []
        beam: List[List[int]] = [start]
        best_full: Tuple[List[int], float] | None = None
        for _ in range(depth):
            candidates = []
            for state in beam:
                for nxt in self.expand(state):
                    bound = optimistic_bound(nxt)
                    heapq.heappush(candidates, (-bound, nxt))
            # prune
            new_beam: List[List[int]] = []
            while candidates and len(new_beam) < self.beam_width:
                _, st = heapq.heappop(candidates)
                new_beam.append(st)
            beam = new_beam
            # evaluate any complete states
            for st in beam:
                if len(st) == chromosome_length() and is_feasible(st):
                    fit = fitness(st)
                    if best_full is None or fit > best_full[1]:
                        best_full = (st, fit)
        # if best_full not found, attempt fallback by completing partial with zeros
        if best_full is None:
            for st in beam:
                if len(st) < chromosome_length():
                    filled = st + [0]*(chromosome_length()-len(st))
                    if is_feasible(filled):
                        fit = fitness(filled)
                        if best_full is None or fit > best_full[1]:
                            best_full = (filled, fit)
        return best_full if best_full else ([], 0.0)
