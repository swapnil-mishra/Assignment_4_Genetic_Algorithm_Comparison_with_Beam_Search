import json  # still used temporarily if legacy cleanup needed
import time
from pathlib import Path
import sys
import os

# Add local src path for embedded / non-installed execution
SRC_DIR = Path(__file__).parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from genetic_algorithm import GeneticAlgorithm, sample_fitness
from beam_search import BeamSearch, example_expand

OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

def run_ga():
    ga = GeneticAlgorithm(population_size=40, chromosome_length=30, fitness_fn=sample_fitness, seed=42)
    t0 = time.time()
    best_ch, best_fit, history = ga.run(generations=60)
    elapsed = time.time() - t0
    return {
        'algorithm': 'GeneticAlgorithm',
        'best_fitness': best_fit,
        'best_chromosome': best_ch,
        'history': history,
        'time_sec': elapsed
    }

def run_beam():
    import random
    start = [random.randint(0,1) for _ in range(30)]
    beam = BeamSearch(beam_width=5, expand_fn=example_expand, score_fn=sum)
    t0 = time.time()
    best_state, best_score = beam.search(start, depth=30)
    elapsed = time.time() - t0
    return {
        'algorithm': 'BeamSearch',
        'best_fitness': best_score,
        'best_state': best_state,
        'time_sec': elapsed
    }

if __name__ == '__main__':
    ga_res = run_ga()
    beam_res = run_beam()

    # Prepare plain-text summary instead of JSON per user request
    lines = []
    lines.append('=== Question A: GA vs Beam Search Comparison ===')
    lines.append(f"GA Best Fitness: {ga_res['best_fitness']}")
    lines.append(f"GA Chromosome Length: {len(ga_res['best_chromosome'])}")
    lines.append(f"GA Time (s): {ga_res['time_sec']:.6f}")
    lines.append(f"GA Generations History Length: {len(ga_res['history'])}")
    lines.append(f"GA Early Progress (first 10 generations): {ga_res['history'][:10]}")
    lines.append('')
    lines.append(f"Beam Best Fitness: {beam_res['best_fitness']}")
    lines.append(f"Beam Time (s): {beam_res['time_sec']:.6f}")
    lines.append(f"Beam Start Length: {len(beam_res['best_state'])}")
    lines.append('')
    lines.append('Best GA Chromosome:')
    lines.append(''.join(str(b) for b in ga_res['best_chromosome']))
    lines.append('Best Beam State:')
    lines.append(''.join(str(b) for b in beam_res['best_state']))
    lines.append('')
    if ga_res['best_fitness'] == beam_res['best_fitness']:
        lines.append('Observation: Both algorithms reached the same optimal fitness.')
    else:
        lines.append('Observation: Fitness differs between GA and Beam search.')

    txt_path = OUT_DIR / 'comparison_results.txt'
    txt_path.write_text('\n'.join(lines))

    # Optional: remove legacy JSON file if it exists to avoid confusion
    legacy_json = OUT_DIR / 'comparison_results.json'
    if legacy_json.exists():
        try:
            os.remove(legacy_json)
        except OSError:
            pass

    print(f'Text results saved to {txt_path}')
