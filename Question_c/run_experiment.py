import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from placeholder import fitness, brute_force_optimum, chromosome_length, is_feasible, evaluate_solution
from genetic_algorithm import GeneticAlgorithm
from beam_search import BeamSearchKnapsack

OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

CFG = {
    'ga': {
        'population_size': 90,
        'generations': 180,
        'crossover_rate': 0.85,
        'mutation_rate': 0.03,
        'tournament_size': 3,
        'seed': 2025
    },
    'beam': {
        'beam_width': 40,
        'extra_depth': 5
    }
}

def run_ga():
    ga = GeneticAlgorithm(population_size=CFG['ga']['population_size'],
                          chromosome_length=chromosome_length(),
                          fitness_fn=fitness,
                          crossover_rate=CFG['ga']['crossover_rate'],
                          mutation_rate=CFG['ga']['mutation_rate'],
                          tournament_size=CFG['ga']['tournament_size'],
                          seed=CFG['ga']['seed'])
    t0 = time.time()
    best_ch, best_fit, history = ga.run(generations=CFG['ga']['generations'])
    elapsed = time.time() - t0
    return best_ch, best_fit, history, elapsed

def run_beam():
    beam = BeamSearchKnapsack(CFG['beam']['beam_width'])
    t0 = time.time()
    best_state, best_score = beam.search(depth=chromosome_length()+CFG['beam']['extra_depth'])
    elapsed = time.time() - t0
    return best_state, best_score, elapsed

def main():
    brute_val, brute_ch = brute_force_optimum()
    ga_ch, ga_fit, ga_hist, ga_time = run_ga()
    beam_ch, beam_fit, beam_time = run_beam()
    ga_val, ga_weight = evaluate_solution(ga_ch)
    beam_val, beam_weight = evaluate_solution(beam_ch)

    lines = []
    lines.append('=== Question C: 0/1 Knapsack Results ===')
    lines.append(f'Brute Force Optimum Value: {brute_val}')
    lines.append(f'Brute Force Chromosome: {"".join(str(b) for b in brute_ch)}')
    lines.append('')
    lines.append('--- Genetic Algorithm ---')
    lines.append(f'GA Best Fitness (penalized if overweight): {ga_fit}')
    lines.append(f'GA Feasible: {is_feasible(ga_ch)}  (Value={ga_val} Weight={ga_weight})')
    lines.append(f'GA Time (s): {ga_time:.4f}')
    lines.append(f'GA Generations: {CFG["ga"]["generations"]}')
    lines.append(f'GA First 15 Fitness History: {ga_hist[:15]}')
    lines.append(f'GA Final Fitness History (last 10): {ga_hist[-10:]}')
    lines.append('')
    lines.append('--- Beam Search ---')
    lines.append(f'Beam Best Score: {beam_fit}')
    lines.append(f'Beam Feasible: {is_feasible(beam_ch)} (Value={beam_val} Weight={beam_weight})')
    lines.append(f'Beam Width: {CFG["beam"]["beam_width"]} Depth Used: {chromosome_length()+CFG["beam"]["extra_depth"]}')
    lines.append(f'Beam Time (s): {beam_time:.4f}')
    lines.append('')
    if ga_fit >= 0.95 * brute_val and beam_fit >= 0.95 * brute_val:
        lines.append('Observation: Both methods reached near-optimal solutions.')
    elif ga_fit > beam_fit:
        lines.append('Observation: GA outperformed Beam Search in this configuration.')
    else:
        lines.append('Observation: Beam Search outperformed GA in this configuration.')

    out_path = OUT_DIR / 'question_c_results.txt'
    out_path.write_text('\n'.join(lines))
    print(f'Results saved to {out_path}')

if __name__ == '__main__':
    main()
