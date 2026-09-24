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

Run the main simulation experiments with:

```bash
python Experiments/run_experiments.py
```

Run the hyperparameter experiments with:

```bash
python Experiments/run_hyperparameter_experiments.py
```

## Analysis

The saved experiment results are analysed in `BVI_comparison.ipynb`.

The main simulation outputs are stored in `results/`, hyperparameter experiment outputs in `results/hyperparams_results/`, and the processed CSV tables used in the final report are stored in `results/export_tables/`.
