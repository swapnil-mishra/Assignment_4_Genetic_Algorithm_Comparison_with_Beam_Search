from __future__ import annotations
from typing import Callable, List, Tuple
import heapq

class BeamSearch:
    def __init__(self, beam_width: int, expand_fn: Callable[[List[int]], List[List[int]]], score_fn: Callable[[List[int]], float], maximize: bool = True):
        self.beam_width = beam_width
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.maximize = maximize

    def search(self, start: List[int], depth: int) -> Tuple[List[int], float]:
        beam = [start]
        best = (start, self.score_fn(start))
        for _ in range(depth):
            candidates: List[Tuple[float, List[int]]] = []
            for state in beam:
                for nxt in self.expand_fn(state):
                    score = self.score_fn(nxt)
                    if (self.maximize and score > best[1]) or (not self.maximize and score < best[1]):
                        best = (nxt, score)
                    heapq.heappush(candidates, ((-score) if self.maximize else score, nxt))
            new_beam = []
            while candidates and len(new_beam) < self.beam_width:
                score, st = heapq.heappop(candidates)
                new_beam.append(st)
            beam = new_beam
        return best

def bit_flip_expand(state: List[int]) -> List[List[int]]:
    out = []
    for i in range(len(state)):
        new_state = state[:]
        new_state[i] = 1 - new_state[i]
        out.append(new_state)
    return out
