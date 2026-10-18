# BART Feature Selection Simulation Experiments

This repository contains code, simulation results, and analysis notebooks for experiments evaluating BART feature selection across several test scenarios.

## Repository structure

* `Experiments/` — scripts and helper functions for running the main and hyperparameter experiments.
* `results/` — saved simulation outputs.
* `results/hyperparams_results/` — results from hyperparameter experiments.
* `results/export_tables/` — processed CSV tables used in the final report.
* `BVI_comparison.ipynb` — main notebook for comparing and analysing experiment results.
* `scenario_analysis_helpers.py` — helper functions for scenario-level analysis.
* `sensitivity_analysis_helpers.py` — helper functions for sensitivity analysis.

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

The reported experiments use:

- 20 BART trees in the reference model;
- 200 burn-in iterations and 1,000 retained posterior draws;
- 10 response permutations per feature-selection run;
- one BART fit per observed or permuted response;
- permutation threshold level \(\alpha=0.05\);
- 100 simulation repeats for the main experiments;
- 50 paired repeats for the hyperparameter-sensitivity experiments;
- base seed 123.

The full model and selector configuration is defined explicitly in the
experiment scripts.

### Main simulation study

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

### Hyperparameter-sensitivity study

The sensitivity study varies one of \(m\), \(\beta\), or \(k\) at a time around
the reference configuration.

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

`--n-processes` controls only parallel execution and may be changed without
changing the intended experimental design.

The committed CSV files in `results/` contain the results used in the report,
including the repeat-level random seeds.

## Analysis

The saved experiment results are analysed in `BVI_comparison.ipynb`.

The main simulation outputs are stored in `results/`, hyperparameter experiment outputs in `results/hyperparams_results/`, and the processed CSV tables used in the final report are stored in `results/export_tables/`.

## Report

The accompanying report is available in this repository as [`BART_logML_selection_report.pdf`](report/BART_logML_selection_report.pdf). The report describes the methodology, experiments, and interpretation of the results, while this repository contains the code and data used to produce them.

