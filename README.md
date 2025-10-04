# Genetic Algorithm (GA) vs Beam Search: Comparative Study

Author Information  
Name: Swapnil Mishra  
College: Indian Institute of Technology, Patna  
Degree: Master Of Technology (M.Tech.)  
Stream: Computer Science Engineering (CSE)  
Semester: II  
Subject: Artificial Intelligence Lab  
Roll No.: 25s09res44  
Email Id: swapnil_25s09res44@iitp.ac.in

## Table of Contents
1. Introduction  
2. Problem Formulation  
3. Implementation Details  
4. Code Explanation  
5. Results & Observations  
6. Conclusion  
7. References  
8. Appendix (Source Code)  
9. How to Run  
10. Future Improvements

---

## 1. Introduction
This assignment compares two search/optimization paradigms—Genetic Algorithm (GA) and Beam Search—across three increasingly structured problems:
1. Binary OneMax (Question A) – maximize number of 1s in a bitstring.
2. Weighted Plateau Variant (Question B) – weighted OneMax with a deceptive near-optimal plateau.
3. 0/1 Knapsack (Question C) – constrained combinatorial optimization with capacity.

The goals were to: (a) implement both algorithms cleanly, (b) evaluate convergence behavior, runtime, optimality, and (c) reflect on diversity vs determinism (especially relevant to creative generation tasks).

## 2. Problem Formulation
### Question A – OneMax
Given a bitstring of length L, maximize:  
\( f(\mathbf{x}) = \sum_{i=1}^{L} x_i \), where \( x_i \in \{0,1\} \).

### Question B – Weighted Plateau OneMax
Bitstring length L=40. Weights: \( w_i = 1 + (i \bmod 7)/10 \). Raw score: \( S = \sum w_i x_i \).  
If \( 0.95 S_{max} \le S < S_{max} \), apply a penalty of 0.5 to create a plateau region. Objective: maximize penalized score.

### Question C – 0/1 Knapsack
Items (value, weight) lists of length 15. Capacity = 25. Maximize total value subject to weight constraint. Over-capacity solutions penalized linearly:  
\( f(\mathbf{x}) = V(\mathbf{x}) - \lambda (W(\mathbf{x}) - C)_+ \), with penalty \(\lambda = 5\).  
Brute force used to confirm the true optimum (value = 96).

## 3. Implementation Details
- Language: Python 3 (embedded distribution used here – no external deps)
- GA Components: tournament selection, single-point crossover, bit-flip mutation, population history tracking.
- Beam Search:
	- Q(A,B): Beam over neighborhood bit flips (exploitative local exploration).
	- Q(C): Constructive beam with optimistic fractional knapsack bound.
- Separation: Each question has independent `src/`, `tests/`, and `output/`.
- Outputs stored as `.txt` summaries (per later requirement change from JSON).

## 4. Code Explanation (High-Level)
| Component | File (Question A) | Description |
|-----------|------------------|-------------|
| GA Core | `Question_a/src/genetic_algorithm.py` | Generic GA scaffold (re-used conceptually in B, C) |
| Beam Search | `Question_a/src/beam_search.py` | Neighborhood beam for bit flip exploration |
| Experiment | `Question_a/run_experiment.py` | Runs GA & Beam, writes summary TXT |
| Tests | `Question_a/tests/test_algorithms.py` | Ensures improvement & correctness |

Question B defines a weighted plateau fitness (`placeholder.py`) and mirrors GA/Beam modules.  
Question C adds knapsack-specific logic and a beam with optimistic bound in `beam_search.py` plus brute force verification.

## 5. Results & Observations
### Summary Table
| Question | Problem | GA Best | Beam Best | Optimal Known | GA Time (s)* | Beam Time (s)* | Notes |
|----------|---------|---------|-----------|---------------|--------------|----------------|-------|
| A | OneMax (L=30) | 30 | 30 | 30 | ~0.015 | ~0.0016 | Both reach optimum quickly; GA converges ~gen 10 |
| B | Weighted Plateau (L=40) | 51.5 (all trials) | 51.5 (all trials) | 51.5 | Mean ~0.111 | Mean ~0.174 | GA faster; plateau did not trap either method |
| C | Knapsack (15 items) | 96 | 96 | 96 (brute force) | ~0.168 | ~0.0000 | Beam very fast (small depth); both optimal |
*Representative single-run or mean values from the generated output files.

### Question A Detail
Early GA fitness progression (first 10 gens): `[21, 24, 25, 27, 28, 28, 28, 28, 29, 30]` → rapid convergence, stable optimum thereafter. Beam Search reaches optimum quickly because OneMax has smooth gradient-like landscape.

### Question B Detail
Plateau penalty did not degrade performance: both algorithms reach the theoretical maximum (51.5) in every trial. GA exhibits lower runtime due to parallel exploitation; beam spends more time maintaining/pruning candidate sets. No fitness variance (stdev = 0) across trials for either algorithm under chosen parameters.

### Question C Detail
Both GA and Beam Search exactly match the brute-force optimum (value = 96). GA converges by generation ~8–10 (fitness stabilizes at 96). Beam’s constructive search with an admissible (optimistic) bound quickly isolates the best feasible set.

### Diversity vs Determinism (Cross-Cutting Insight)
Genetic Algorithm offers higher solution diversity (multiple distinct high-quality chromosomes per generation). Beam Search is more deterministic and exploitative—prefers highest-scoring (or highest-bound) branches, potentially pruning innovative but temporarily weaker candidates.

## 6. Conclusion
1. Both GA and Beam Search perform optimally on the selected benchmark problems.
2. GA demonstrates faster runtime than Beam Search on the plateau landscape (B) due to population-level improvements outweighing beam pruning overhead.
3. Beam Search can be extremely fast for structured constructive tasks (C) when a strong admissible bound exists.
4. GA is preferable where diversity, exploration, or multi-objective adaptation matters (e.g., creative / language generation tasks). Beam Search is preferable when deterministic, bounded, and consistent expansion is required.
5. Problem difficulty here is modest; more deceptive or large-scale landscapes would likely amplify differences (GA resilience vs Beam pruning sensitivity).

## 7. References
- Eiben & Smith, "Introduction to Evolutionary Computing".
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" (Beam search & heuristic search chapters).
- Common GA patterns (tournament selection, mutation) – standard literature.

## 8. Appendix (Source Code)
Structure:
```
Question_a/
	src/ (GA, Beam, fitness)
	tests/
	output/
Question_b/
	src/ (Weighted fitness + GA/Beam)
	tests/
	output/
Question_c/
	src/ (Knapsack + GA + Beam with bound)
	tests/
	output/
```
`Question_c/answer.txt` contains extended analytical discussion for part (c).

## 9. How to Run
Using the embedded Python in `pyembed/` (already configured in this workspace):
```powershell
# Run all tests
.\pyembed\python.exe -m unittest discover -s Question_a/tests -t .
.\pyembed\python.exe -m unittest discover -s Question_b/tests -t .
.\pyembed\python.exe -m unittest discover -s Question_c/tests -t .

# Run experiments (text outputs)
.\pyembed\python.exe Question_a\run_experiment.py
type Question_a\output\comparison_results.txt

.\pyembed\python.exe Question_b\run_experiment.py
type Question_b\output\question_b_results.txt

.\pyembed\python.exe Question_c\run_experiment.py
type Question_c\output\question_c_results.txt
```
If using a standard Python install instead:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt  # currently minimal
python -m unittest discover -s Question_a/tests -t .
```

## 10. Future Improvements
| Enhancement | Rationale |
|-------------|-----------|
| Shared core GA/Beam module | Remove duplication across questions |
| Elitism in GA | Guarantee monotonic best fitness retention |
| Diversity metrics (Hamming distance) | Quantify exploratory breadth |
| Parameter sweeps / plots | Visualize convergence vs mutation/beam width |
| Multi-objective GA (value & diversity) | Align with creative generation use cases |
| Logging & CSV exports | Easier post-hoc statistical analysis |

---
