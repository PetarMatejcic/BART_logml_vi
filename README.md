# LogML-Weighted Variable Importance for BART

This repository contains the simulation experiments, analysis, and report for a
study of variable selection in Bayesian Additive Regression Trees (BART).

The project investigates whether standard BART variable inclusion proportions
can be improved by incorporating information about the contribution of the tree
structure associated with each split. It compares the resulting importance
measure with standard split-count variable inclusion proportions in a common
permutation-based feature-selection framework.

## Research question

Standard variable inclusion proportions count how often each predictor is used
in a splitting rule, giving every split equal weight. This study asks whether
feature selection can be improved by weighting those split occurrences
according to how strongly the corresponding subtree is supported by the data.

The experiments focus on two questions:

1. Does the logML-weighted importance improve recovery of truly relevant
   predictors relative to standard variable inclusion proportions?
2. Are the conclusions stable under moderate changes to selected BART
   hyperparameters?

## Method

For each internal node $v$, the fitted subtree rooted at that node is compared
with an alternative tree in which the subtree is replaced by a single terminal
node. Their difference in log marginal likelihood defines the gain
$\delta_v$, and the split receives weight

$$
w_v = \max(\delta_v, 0).
$$

These weights are accumulated by splitting variable and normalized to form a
relative importance measure. Standard and logML-weighted importance are then
used within the same permutation-based feature-selection procedure, so the
comparison isolates the effect of changing the importance statistic.

## Main result

Across the synthetic experiments, logML-weighted selection generally achieved
better feature-recovery performance than standard variable inclusion
proportions, with the clearest gains in the higher-dimensional
continuous-response settings.

The improvement was not universal: in one lower-dimensional continuous
setting, standard variable inclusion proportions performed better because the
logML-weighted method tended to underselect. The binary-response scenarios were
difficult for both approaches.

A separate hyperparameter-sensitivity study broadly preserved the main
conclusions. The full results and discussion are given in the report.


## Report

The full methodology, experimental design, results, and discussion are available in:

[**BART_logML_selection_report.pdf**](report/BART_logML_selection_report.pdf)


## Repository structure

- `Experiments/` — simulation drivers and data-generating functions.
- `results/` — committed repeat-level simulation outputs.
- `results/hyperparams_results/` — paired hyperparameter-sensitivity results.
- `results/export_tables/` — derived tables used in the report.
- `simulation_analysis.ipynb` — validation, aggregation, and analysis of the
  committed results.
- `scenario_analysis_helpers.py` and `sensitivity_analysis_helpers.py` —
  analysis utilities.
- `report/BART_logML_selection_report.pdf` — full methodology, results, and
  discussion.

The BART implementation used in the experiments is provided by
[GenBART](https://github.com/PetarMatejcic/GenBART).


## Installation

Python 3.10 or newer is recommended.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the version of [GenBART](https://github.com/PetarMatejcic/GenBART) used for these experiments directly from GitHub.

## Running the experiments

The reported experiments use 20 BART trees, 200 burn-in iterations, 1,000
retained posterior draws, 10 response permutations, and base seed 123. The main
experiments use 100 simulation repeats, while the hyperparameter-sensitivity
study uses 50 paired repeats.

The full model and selector configurations are defined explicitly in the
experiment scripts.

<details>
<summary>Commands for the main simulation study</summary>

```bash
# Scenarios 1-2: continuous predictors
python3 Experiments/run_experiments.py \
    --scenarios cc1 cc2 \
    --ns 500 \
    --ps 50 200 \
    --s2s 1 \
    --repeats 100 \
    --seed 123 \
    --n-processes 4

# Scenario 3: mixed predictors
python3 Experiments/run_experiments.py \
    --scenarios cm1 \
    --ns 500 1000 \
    --ps 50 200 \
    --s2s 1 10 \
    --repeats 100 \
    --seed 123 \
    --n-processes 4

# Scenario 4: fixed p = 84
python3 Experiments/run_experiments.py \
    --scenarios cm2 \
    --ns 500 1000 \
    --ps 84 \
    --s2s 1 10 \
    --repeats 100 \
    --seed 123 \
    --n-processes 4

# Scenario 5: binary response
python3 Experiments/run_experiments.py \
    --scenarios bm1 \
    --ns 500 1000 \
    --ps 50 200 \
    --s2s 1 \
    --repeats 100 \
    --seed 123 \
    --n-processes 4

# Scenario 6: binary response, fixed p = 84
python3 Experiments/run_experiments.py \
    --scenarios bm2 \
    --ns 500 1000 \
    --ps 84 \
    --s2s 1 \
    --repeats 100 \
    --seed 123 \
    --n-processes 4
```
</details>


<details>
<summary>Commands for the hyperparameter-sensitivity study</summary>

```bash
python3 Experiments/run_hyperparameter_experiments.py \
    --scenarios cc1 \
    --n 500 \
    --p 50 \
    --s2 1 \
    --repeats 50 \
    --seed 123 \
    --n-processes 4

python3 Experiments/run_hyperparameter_experiments.py \
    --scenarios cm2 \
    --n 500 \
    --p 84 \
    --s2 10 \
    --repeats 50 \
    --seed 123 \
    --n-processes 4

python3 Experiments/run_hyperparameter_experiments.py \
    --scenarios bm1 \
    --n 1000 \
    --p 50 \
    --s2 1 \
    --repeats 50 \
    --seed 123 \
    --n-processes 4
```
</details>

`--n-processes` controls only parallel execution and may be changed without
changing the intended experimental design.

The committed CSV files in `results/` contain the results used in the report,
including the repeat-level random seeds.

## Analysis workflow

The expensive BART simulations are stored as committed repeat-level CSV files;
`simulation_analysis.ipynb` validates and aggregates those results without
rerunning the simulations.

```text
Experiments/*.py
      ↓
results/*.csv
results/hyperparams_results/*.csv
      ↓
simulation_analysis.ipynb
      ↓
results/export_tables/*.csv
      ↓
report/BART_logML_selection_report.pdf
```
