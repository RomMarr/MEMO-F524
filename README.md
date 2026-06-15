# Machine Learning for Seismometer Placement Optimization

**Romain Markowitch** — Master's thesis, ULB 2025-2026.

Supervizor : Gianluca Bontempi

Co-supervizor : Pascal Tribel

## Repository structure

The thesis PDF and the interactive demo are available at the root of the repository.
All source code lives in `src/`.

## Getting started

`experiments.ipynb` is the main notebook. It walks through all the experiments
presented in the thesis: forward solver comparison, batch inversion accuracy,
noise robustness, runtime benchmark and interrogator placement optimization.
It also serves as the reference for how to use the M2 package.

## M2 folder

The code used throughout the thesis.

- `DP.py` — differentiable finite-difference forward solver built in PyTorch.
  Fully differentiable with respect to the epicenter coordinates, enabling
  gradient-based inversion without a precomputed dataset.
- `inverse_problem.py` — epicenter inversion via gradient-based misfit minimization.
- `placement.py` — bilevel interrogator placement optimization, combining
  gradient-based outer loop and iterated local search.
- `PINN/` — physics-informed neural network surrogate's architecture, PDE residual
  loss.
- `Utils/` — shared utilities : Ricker wavelet source and visualization helpers.

## M1 folder

`M1/` contains code written during the preparatory work course that preceded the
thesis. It covers early explorations of the problem and served as the foundation
for the approaches developed in M2. It is not maintained and may not run
out of the box.
