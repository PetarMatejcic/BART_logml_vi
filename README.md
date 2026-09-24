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

## Running the experiments

The main experiments can be run with:

```bash
python Experiments/run_experiments.py
```

Hyperparameter experiments can be run with:

```bash
python Experiments/run_hyperparameter_experiments.py
```

## Analysis

The saved experiment results are analysed in `BVI_comparison.ipynb`. The processed tables used in the report are available in `results/export_tables/`.
