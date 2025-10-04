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
            # pick top beam_width
            new_beam = []
            while candidates and len(new_beam) < self.beam_width:
                score, st = heapq.heappop(candidates)
                new_beam.append(st)
            beam = new_beam
        return best

# Example placeholder expand function flips each bit individually

def example_expand(state: List[int]) -> List[List[int]]:
    out = []
    for i in range(len(state)):
        new_state = state[:]
        new_state[i] = 1 - new_state[i]
        out.append(new_state)
    return out

if __name__ == "__main__":
    import random
    start = [random.randint(0,1) for _ in range(10)]
    bs = BeamSearch(beam_width=3, expand_fn=example_expand, score_fn=sum)
    best_state, best_score = bs.search(start, depth=10)
    print("Start:", start)
    print("Best:", best_state, best_score)
