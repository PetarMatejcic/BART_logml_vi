#!/usr/bin/env python3
"""
Run a parameter grid for one or more simulation scenarios.

Each data-generating function must return:

    X, y, truth

where ``truth`` is a one-dimensional Boolean mask with one entry per column
of X.

Examples
--------
Command line:

    python run_experiments.py \
        --scenarios cc1 cc2 \
        --ns 100 500 \
        --ps 10 50 \
        --s2s 1 5 \
        --repeats 20 \
        --seed 123

Python:

    results = run_experiments(
        scenarios=["cc1", "cc2"],
        ns=[100, 500],
        ps=[10, 50],
        s2s=[1.0, 5.0],
        repeats=20,
        base_seed=123,
    )

The returned dictionary maps each scenario name to its repeat-level results DataFrame.
Each DataFrame is also saved as a separate CSV file.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import test_functions as data
from runners import run_scenario, summarise_results
from genbart import RegBart, ProbitBart


# ---------------------------------------------------------------------------
# EDITABLE CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    datagen: Callable[[int, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]]
    model_key: str


@dataclass(frozen=True)
class ModelSpec:
    model: Any
    model_params: dict[str, Any]


# Add, remove, or rename scenarios here.
SCENARIOS: dict[str, ScenarioSpec] = {
    "cc0": ScenarioSpec(data.datagen_cc0, "continuous"),
    "cc1": ScenarioSpec(data.datagen_cc1, "continuous"),
    "cc2": ScenarioSpec(data.datagen_cc2, "continuous"),
    "cm1": ScenarioSpec(data.datagen_cm1, "continuous"),
    "cm2": ScenarioSpec(data.datagen_cm2, "continuous"),
    "bm1": ScenarioSpec(data.datagen_bm1, "binary"),
    "bm2": ScenarioSpec(data.datagen_bm2, "binary"),
}


MODEL_CONFIGS: dict[str, ModelSpec] = {
    "continuous": ModelSpec(
        model=RegBart,
        model_params={"m": 20},
    ),
    "binary": ModelSpec(
        model=ProbitBart,
        model_params={"m": 20},
    ),
}


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

METRIC_COLUMNS = tuple(
    f"{metric}_{importance}_{method}"
    for importance in ("raw", "logl")
    for method in ("local", "global_max", "global_se")
    for metric in ("precision", "recall", "f1", "r_miss", "n_selected")
)


def _stable_seed(
    base_seed: int,
    scenario: str,
    n: int,
    p: int,
    s2: float,
    repeat: int,
) -> int:
    """Create a stable seed that does not depend on run order."""
    value = f"{base_seed}|{scenario}|{n}|{p}|{s2:.17g}|{repeat}"
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _validate_truth(truth: Any, number_of_features: int) -> np.ndarray:
    truth_array = np.asarray(truth)

    if truth_array.ndim != 1:
        raise ValueError("truth must be a one-dimensional Boolean mask.")

    if truth_array.size != number_of_features:
        raise ValueError(
            f"truth has length {truth_array.size}, but X has "
            f"{number_of_features} columns."
        )

    return truth_array.astype(bool, copy=False)


def _run_one_repeat(
    scenario_spec: ScenarioSpec,
    model_spec: ModelSpec,
    n: int,
    p: int,
    s2: float,
    seed: int,
) -> pd.Series:
    """
    Run one repeat through the existing run_scenario function.

    The adapter removes truth before run_scenario calls the data generator,
    while retaining truth for summarise_results.
    """
    truth_holder: list[np.ndarray] = []

    def datagen_adapter(
        adapter_n: int,
        adapter_p: int,
        adapter_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        generated = scenario_spec.datagen(adapter_n, adapter_p, adapter_s)

        if not isinstance(generated, tuple) or len(generated) != 3:
            raise ValueError(
                "The data-generating function must return exactly "
                "(X, y, truth)."
            )

        X, y, truth = generated
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional; got shape {X.shape}.")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows, but y has length {y.shape[0]}."
            )

        truth_holder.append(_validate_truth(truth, X.shape[1]))
        return X, y

    model_params = dict(model_spec.model_params)

    raw, logl = run_scenario(
        datagen=datagen_adapter,
        n=n,
        p=p,
        s2=s2,
        model=model_spec.model,
        model_params=model_params,
        random_state=seed,
    )

    if len(truth_holder) != 1:
        raise RuntimeError(
            "Expected the data-generating function to be called exactly once."
        )

    return summarise_results(raw, logl, truth_holder[0])


def _run_repeat_job(
    scenario_name: str,
    n: int,
    p: int,
    s2: float,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    """Run one complete simulation repeat.

    This function is top-level so it can be executed in a worker process.
    """
    scenario_spec = SCENARIOS[scenario_name]
    model_spec = MODEL_CONFIGS[scenario_spec.model_key]

    _set_seed(seed)

    repeat_info = {
        "scenario": scenario_name,
        "model_key": scenario_spec.model_key,
        "n": int(n),
        "p": int(p),
        "s2": float(s2),
        "repeat": int(repeat),
        "seed": int(seed),
    }

    try:
        result = _run_one_repeat(
            scenario_spec=scenario_spec,
            model_spec=model_spec,
            n=int(n),
            p=int(p),
            s2=float(s2),
            seed=int(seed),
        )

        return {
            **repeat_info,
            **result.to_dict(),
            "status": "complete",
            "error": "",
        }

    except Exception as error:
        return {
            **repeat_info,
            **_failed_repeat_row(error),
        }


def _failed_repeat_row(
    error: Exception,
) -> dict[str, Any]:
    """Create empty metric values for one failed simulation repeat."""
    row: dict[str, Any] = {
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }

    for metric in METRIC_COLUMNS:
        row[metric] = np.nan

    return row


def _next_output_path(output_dir: Path, scenario: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    first_path = output_dir / f"{scenario}.csv"
    if not first_path.exists():
        return first_path

    version = 1
    while True:
        candidate = output_dir / f"{scenario}_{version:03d}.csv"
        if not candidate.exists():
            return candidate
        version += 1


def run_experiments(
    scenarios: Iterable[str],
    ns: Iterable[int],
    ps: Iterable[int],
    s2s: Iterable[float],
    repeats: int,
    base_seed: int = 0,
    output_dir: str | Path = "results",
    n_processes: int = 1,
) -> dict[str, pd.DataFrame]:
    """
    Run all requested parameter combinations and save one CSV per scenario.

    Invalid combinations are recorded as failed rows and produce warnings.
    Other combinations continue running.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")

    scenario_names = list(dict.fromkeys(scenarios))
    n_values = list(ns)
    p_values = list(ps)
    s2_values = list(s2s)
    output_dir = Path(output_dir)

    if not scenario_names:
        raise ValueError("At least one scenario must be selected.")
    if not n_values or not p_values or not s2_values:
        raise ValueError("ns, ps, and s2s must all contain at least one value.")

    runnable_scenario_names = [
        scenario_name
        for scenario_name in scenario_names
        if (
            scenario_name in SCENARIOS
            and SCENARIOS[scenario_name].model_key in MODEL_CONFIGS
        )
    ]

    number_of_combinations = (
        len(n_values)
        * len(p_values)
        * len(s2_values)
    )

    total_repeat_jobs = (
        len(runnable_scenario_names)
        * number_of_combinations
        * repeats
    )

    outputs: dict[str, pd.DataFrame] = {}

    executor = (
        ProcessPoolExecutor(max_workers=n_processes)
        if n_processes > 1
        else None
    )

    try:
        with tqdm(
            total=total_repeat_jobs,
            desc="Experiments",
            unit="repeat",
        ) as progress:

            scenario_position = 0

            for scenario_name in scenario_names:
                if scenario_name not in SCENARIOS:
                    warnings.warn(
                        f"Unknown scenario {scenario_name!r}; skipping it.",
                        stacklevel=2,
                    )
                    continue

                scenario_spec = SCENARIOS[scenario_name]

                if scenario_spec.model_key not in MODEL_CONFIGS:
                    warnings.warn(
                        f"Scenario {scenario_name!r} refers to missing model config "
                        f"{scenario_spec.model_key!r}; skipping it.",
                        stacklevel=2,
                    )
                    continue

                model_spec = MODEL_CONFIGS[scenario_spec.model_key]
                if model_spec.model is None:
                    raise ValueError(
                        f"MODEL_CONFIGS[{scenario_spec.model_key!r}].model is None. "
                        "Set the continuous and binary models in the editable "
                        "configuration section."
                    )

                scenario_position += 1

                rows: list[dict[str, Any]] = []

                parameter_grid = itertools.product(
                    n_values,
                    p_values,
                    s2_values,
                )

                for n, p, s2 in parameter_grid:

                    progress.set_postfix(
                        scenario=f"{scenario_name} "
                                f"({scenario_position}/{len(runnable_scenario_names)})",
                        n=int(n),
                        p=int(p),
                        s2=float(s2),
                        refresh=True,
                    )

                    jobs = []

                    for repeat in range(repeats):
                        seed = _stable_seed(
                            base_seed=base_seed,
                            scenario=scenario_name,
                            n=int(n),
                            p=int(p),
                            s2=float(s2),
                            repeat=repeat,
                        )

                        jobs.append(
                            (
                                scenario_name,
                                int(n),
                                int(p),
                                float(s2),
                                repeat,
                                seed,
                            )
                        )

                    combination_rows = []

                    if executor is None:
                        # Serial execution: update immediately after every repeat.
                        for job in jobs:
                            row = _run_repeat_job(*job)
                            combination_rows.append(row)
                            progress.update(1)

                    else:
                        # Parallel execution: update whenever any worker finishes.
                        futures = [
                            executor.submit(
                                _run_repeat_job,
                                *job,
                            )
                            for job in jobs
                        ]

                        for future in as_completed(futures):
                            row = future.result()
                            combination_rows.append(row)
                            progress.update(1)

                    for row in combination_rows:
                        if row["status"] == "failed":
                            warnings.warn(
                                "Failed repeat "
                                f"scenario={row['scenario']!r}, "
                                f"n={row['n']}, "
                                f"p={row['p']}, "
                                f"s2={row['s2']}, "
                                f"repeat={row['repeat']}: "
                                f"{row['error']}",
                                stacklevel=2,
                            )

                        rows.append(row)

                scenario_df = pd.DataFrame(rows)

                scenario_df = (
                    scenario_df
                    .sort_values(
                        ["n", "p", "s2", "repeat"],
                        kind="stable",
                    )
                    .reset_index(drop=True)
                )
                output_path = _next_output_path(output_dir, scenario_name)
                scenario_df.to_csv(output_path, index=False)
                scenario_df.attrs["output_path"] = str(output_path)
                outputs[scenario_name] = scenario_df

                tqdm.write(f"Saved {scenario_name}: {output_path}")

    finally:
        if executor is not None:
            executor.shutdown()

    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected scenarios over all combinations of n, p, and s2."
        )
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help=(
            "Scenario names to run, or 'all'. Available: "
            + ", ".join(SCENARIOS)
        ),
    )
    parser.add_argument("--ns", nargs="+", type=int, required=True)
    parser.add_argument("--ps", nargs="+", type=int, required=True)
    parser.add_argument("--s2s", nargs="+", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help=(
            "Number of simulation repeats to run concurrently. "
            "Default: 1 (serial execution)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    selected_scenarios = (
        list(SCENARIOS)
        if args.scenarios == ["all"]
        else args.scenarios
    )

    run_experiments(
        scenarios=selected_scenarios,
        ns=args.ns,
        ps=args.ps,
        s2s=args.s2s,
        repeats=args.repeats,
        base_seed=args.seed,
        output_dir=args.output_dir,
        n_processes=args.n_processes,
    )


if __name__ == "__main__":
    main()
