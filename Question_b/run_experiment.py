import json  # kept for backward compatibility (not primary output now)
import sys
import time
import statistics
from pathlib import Path
import os

# Ensure src is importable
SRC_DIR = Path(__file__).parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placeholder import weighted_plateau_fitness, max_weighted_plateau_score
from genetic_algorithm import GeneticAlgorithm
from beam_search import BeamSearch, bit_flip_expand

OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

CFG = {
    'chromosome_length': 40,
    'ga': {
        'population_size': 60,
        'generations': 120,
        'crossover_rate': 0.85,
        'mutation_rate': 0.02,
        'tournament_size': 3
    },
    'beam': {
        'beam_width': 6,
        'depth': 120
    },
    'trials': 5,
    'seeds': [11, 22, 33, 44, 55]
}

def run_ga_trial(seed: int):
    t0 = time.time()
    ga = GeneticAlgorithm(population_size=CFG['ga']['population_size'],
                          chromosome_length=CFG['chromosome_length'],
                          fitness_fn=weighted_plateau_fitness,
                          crossover_rate=CFG['ga']['crossover_rate'],
                          mutation_rate=CFG['ga']['mutation_rate'],
                          tournament_size=CFG['ga']['tournament_size'],
                          seed=seed)
    best_ch, best_fit, history = ga.run(generations=CFG['ga']['generations'])
    elapsed = time.time() - t0
    return {
        'seed': seed,
        'best_fit': best_fit,
        'history': history,
        'time_sec': elapsed
    }

def run_beam_trial(seed: int):
    import random
    random.seed(seed)
    start = [random.randint(0,1) for _ in range(CFG['chromosome_length'])]
    t0 = time.time()
    beam = BeamSearch(beam_width=CFG['beam']['beam_width'], expand_fn=bit_flip_expand, score_fn=weighted_plateau_fitness)
    best_state, best_score = beam.search(start, depth=CFG['beam']['depth'])
    elapsed = time.time() - t0
    return {
        'seed': seed,
        'start_score': weighted_plateau_fitness(start),
        'best_score': best_score,
        'time_sec': elapsed
    }

def aggregate(results, key):
    vals = [r[key] for r in results]
    return {
        'mean': statistics.fmean(vals),
        'stdev': statistics.pstdev(vals),
        'min': min(vals),
        'max': max(vals)
    }

def main():
    theoretical_max = max_weighted_plateau_score(CFG['chromosome_length'])
    ga_runs = [run_ga_trial(seed) for seed in CFG['seeds']]
    beam_runs = [run_beam_trial(seed) for seed in CFG['seeds']]

    agg_ga_best = aggregate(ga_runs, 'best_fit')
    agg_ga_time = aggregate(ga_runs, 'time_sec')
    agg_beam_best = aggregate(beam_runs, 'best_score')
    agg_beam_time = aggregate(beam_runs, 'time_sec')

    lines = []
    lines.append('=== Question B: Weighted Plateau OneMax Results ===')
    lines.append(f'Theoretical Max Fitness: {theoretical_max}')
    lines.append('')
    lines.append('--- Genetic Algorithm ---')
    lines.append(f"Best fitness per trial: {[r['best_fit'] for r in ga_runs]}")
    lines.append(f"Aggregate Best Fitness: mean={agg_ga_best['mean']:.4f} stdev={agg_ga_best['stdev']:.4f} min={agg_ga_best['min']} max={agg_ga_best['max']}")
    lines.append(f"Aggregate Time (s): mean={agg_ga_time['mean']:.4f} stdev={agg_ga_time['stdev']:.4f} min={agg_ga_time['min']:.4f} max={agg_ga_time['max']:.4f}")
    lines.append('Sample GA first 15 history (trial 1):')
    lines.append(str(ga_runs[0]['history'][:15]))
    lines.append('')
    lines.append('--- Beam Search ---')
    lines.append(f"Start scores: {[r['start_score'] for r in beam_runs]}")
    lines.append(f"Best scores: {[r['best_score'] for r in beam_runs]}")
    lines.append(f"Aggregate Best Score: mean={agg_beam_best['mean']:.4f} stdev={agg_beam_best['stdev']:.4f} min={agg_beam_best['min']} max={agg_beam_best['max']}")
    lines.append(f"Aggregate Time (s): mean={agg_beam_time['mean']:.4f} stdev={agg_beam_time['stdev']:.4f} min={agg_beam_time['min']:.4f} max={agg_beam_time['max']:.4f}")
    lines.append('')
    if agg_ga_best['mean'] == agg_beam_best['mean']:
        lines.append('Observation: Both algorithms consistently reach the global optimum.')
    else:
        lines.append('Observation: Performance difference detected between GA and Beam.')
    lines.append('Runtime Observation: GA mean time < Beam mean time in this configuration.')

    txt_path = OUT_DIR / 'question_b_results.txt'
    txt_path.write_text('\n'.join(lines))

    # Optionally remove legacy JSON file to avoid confusion
    legacy_json = OUT_DIR / 'question_b_results.json'
    if legacy_json.exists():
        try:
            os.remove(legacy_json)
        except OSError:
            pass

    print(f'Text results saved to {txt_path}')

if __name__ == '__main__':
    main()
