#!/usr/bin/env python3
"""
Run a paired BART hyperparameter-sensitivity experiment.

For each requested scenario and simulation repeat, the same deterministic
simulated dataset is evaluated under seven fixed BART configurations:

    reference:  m=20, alpha=0.95, beta=2, k=2
    m_low:      m=10, alpha=0.95, beta=2, k=2
    m_high:     m=50, alpha=0.95, beta=2, k=2
    beta_low:   m=20, alpha=0.95, beta=1, k=2
    beta_high:  m=20, alpha=0.95, beta=3, k=2
    k_low:      m=20, alpha=0.95, beta=2, k=1
    k_high:     m=20, alpha=0.95, beta=2, k=3

The repeat seed does not depend on the hyperparameter configuration. Therefore
all seven configurations within a given (scenario, repeat) use the same
simulated data and the same BartVariableSelector random_state.

Each completed (scenario, repeat, configuration) is one output row. One CSV is
saved per scenario.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import test_functions as data
from genbart import ProbitBart, RegBart
from runners import run_scenario, summarise_results


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


@dataclass(frozen=True)
class HyperparameterSpec:
    config_id: str
    varied_parameter: str
    m: int
    alpha: float
    beta: float
    k: float

    def model_params(self) -> dict[str, Any]:
        return {
            "m": self.m,
            "alpha": self.alpha,
            "beta": self.beta,
            "k": self.k,
        }


SCENARIOS: dict[str, ScenarioSpec] = {
    "cc0": ScenarioSpec(data.datagen_cc0, "continuous"),
    "cc1": ScenarioSpec(data.datagen_cc1, "continuous"),
    "cc2": ScenarioSpec(data.datagen_cc2, "continuous"),
    "cm1": ScenarioSpec(data.datagen_cm1, "continuous"),
    "cm2": ScenarioSpec(data.datagen_cm2, "continuous"),
    "bm1": ScenarioSpec(data.datagen_bm1, "binary"),
    "bm2": ScenarioSpec(data.datagen_bm2, "binary"),
}


# Put fixed, non-sensitivity-study model parameters here if needed.
# m, alpha, beta, and k are supplied by HYPERPARAMETER_CONFIGS below.
MODEL_CONFIGS: dict[str, ModelSpec] = {
    "continuous": ModelSpec(
        model=RegBart,
        model_params={},
    ),
    "binary": ModelSpec(
        model=ProbitBart,
        model_params={},
    ),
}


HYPERPARAMETER_CONFIGS: tuple[HyperparameterSpec, ...] = (
    HyperparameterSpec("reference", "reference", 20, 0.95, 2, 2),
    HyperparameterSpec("m_low", "m", 10, 0.95, 2, 2),
    HyperparameterSpec("m_high", "m", 50, 0.95, 2, 2),
    HyperparameterSpec("beta_low", "beta", 20, 0.95, 1, 2),
    HyperparameterSpec("beta_high", "beta", 20, 0.95, 3, 2),
    HyperparameterSpec("k_low", "k", 20, 0.95, 2, 1),
    HyperparameterSpec("k_high", "k", 20, 0.95, 2, 3),
)

CONFIG_BY_ID = {
    config.config_id: config
    for config in HYPERPARAMETER_CONFIGS
}

CONFIG_ORDER = {
    config.config_id: index
    for index, config in enumerate(HYPERPARAMETER_CONFIGS)
}


# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

METRIC_COLUMNS = tuple(
    f"{metric}_{importance}_{method}"
    for importance in ("raw", "logml")
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
    """Create a repeat seed independent of execution/configuration order."""
    value = f"{base_seed}|{scenario}|{n}|{p}|{s2:.17g}|{repeat}"
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _set_seed(seed: int) -> None:
    """Set global RNG state used by the simulation data generators."""
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


def _run_one_configuration(
    scenario_spec: ScenarioSpec,
    model_spec: ModelSpec,
    config: HyperparameterSpec,
    n: int,
    p: int,
    s2: float,
    seed: int,
) -> pd.Series:
    """Evaluate one BART configuration on one deterministic simulation repeat."""
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
    model_params.update(config.model_params())

    raw, logml = run_scenario(
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

    return summarise_results(raw, logml, truth_holder[0])


def _failed_configuration_row(error: Exception) -> dict[str, Any]:
    """Create empty metric values for one failed configuration job."""
    row: dict[str, Any] = {
        "status": "failed",
        "error": f"{type(error).__name__}: {error}",
    }

    for metric in METRIC_COLUMNS:
        row[metric] = np.nan

    return row


def _run_configuration_job(
    scenario_name: str,
    n: int,
    p: int,
    s2: float,
    repeat: int,
    seed: int,
    config_id: str,
) -> dict[str, Any]:
    """Run one (scenario, repeat, hyperparameter configuration) job."""
    scenario_spec = SCENARIOS[scenario_name]
    model_spec = MODEL_CONFIGS[scenario_spec.model_key]
    config = CONFIG_BY_ID[config_id]

    # The same repeat seed is intentionally reused for every hyperparameter
    # configuration belonging to this scenario/repeat.
    _set_seed(seed)

    row_info = {
        "scenario": scenario_name,
        "model_key": scenario_spec.model_key,
        "n": int(n),
        "p": int(p),
        "s2": float(s2),
        "repeat": int(repeat),
        "seed": int(seed),
        "config_id": config.config_id,
        "varied_parameter": config.varied_parameter,
        "m": int(config.m),
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "k": float(config.k),
    }

    try:
        result = _run_one_configuration(
            scenario_spec=scenario_spec,
            model_spec=model_spec,
            config=config,
            n=int(n),
            p=int(p),
            s2=float(s2),
            seed=int(seed),
        )

        return {
            **row_info,
            **result.to_dict(),
            "status": "complete",
            "error": "",
        }

    except Exception as error:
        return {
            **row_info,
            **_failed_configuration_row(error),
        }


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


def run_hyperparameter_experiments(
    scenarios: Iterable[str],
    n: int,
    p: int,
    s2: float,
    repeats: int,
    base_seed: int = 0,
    output_dir: str | Path = "hyperparameter_results",
    n_processes: int = 1,
) -> dict[str, pd.DataFrame]:
    """Run the fixed seven-configuration hyperparameter sensitivity study."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    if p < 1:
        raise ValueError("p must be at least 1.")

    if s2 < 0:
        raise ValueError("s2 must be nonnegative.")

    scenario_names = list(dict.fromkeys(scenarios))
    output_dir = Path(output_dir)

    if not scenario_names:
        raise ValueError("At least one scenario must be selected.")

    runnable_scenario_names = [
        scenario_name
        for scenario_name in scenario_names
        if (
            scenario_name in SCENARIOS
            and SCENARIOS[scenario_name].model_key in MODEL_CONFIGS
        )
    ]

    total_jobs = (
        len(runnable_scenario_names)
        * repeats
        * len(HYPERPARAMETER_CONFIGS)
    )

    outputs: dict[str, pd.DataFrame] = {}

    executor = (
        ProcessPoolExecutor(max_workers=n_processes)
        if n_processes > 1
        else None
    )

    try:
        with tqdm(
            total=total_jobs,
            desc="Hyperparameter experiments",
            unit="configuration",
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
                        f"MODEL_CONFIGS[{scenario_spec.model_key!r}].model is None."
                    )

                scenario_position += 1

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

                    for config in HYPERPARAMETER_CONFIGS:
                        jobs.append(
                            (
                                scenario_name,
                                int(n),
                                int(p),
                                float(s2),
                                repeat,
                                seed,
                                config.config_id,
                            )
                        )

                rows: list[dict[str, Any]] = []

                if executor is None:
                    for job in jobs:
                        row = _run_configuration_job(*job)
                        rows.append(row)

                        progress.set_postfix(
                            scenario=(
                                f"{scenario_name} "
                                f"({scenario_position}/"
                                f"{len(runnable_scenario_names)})"
                            ),
                            repeat=row["repeat"],
                            config=row["config_id"],
                            refresh=False,
                        )
                        progress.update(1)

                else:
                    futures = [
                        executor.submit(
                            _run_configuration_job,
                            *job,
                        )
                        for job in jobs
                    ]

                    for future in as_completed(futures):
                        row = future.result()
                        rows.append(row)

                        progress.set_postfix(
                            scenario=(
                                f"{scenario_name} "
                                f"({scenario_position}/"
                                f"{len(runnable_scenario_names)})"
                            ),
                            repeat=row["repeat"],
                            config=row["config_id"],
                            refresh=False,
                        )
                        progress.update(1)

                for row in rows:
                    if row["status"] == "failed":
                        warnings.warn(
                            "Failed hyperparameter configuration "
                            f"scenario={row['scenario']!r}, "
                            f"repeat={row['repeat']}, "
                            f"config={row['config_id']!r}: "
                            f"{row['error']}",
                            stacklevel=2,
                        )

                scenario_df = pd.DataFrame(rows)
                scenario_df["_config_order"] = (
                    scenario_df["config_id"].map(CONFIG_ORDER)
                )

                scenario_df = (
                    scenario_df
                    .sort_values(
                        ["repeat", "_config_order"],
                        kind="stable",
                    )
                    .drop(columns="_config_order")
                    .reset_index(drop=True)
                )

                output_path = _next_output_path(
                    output_dir,
                    scenario_name,
                )

                scenario_df.to_csv(
                    output_path,
                    index=False,
                )

                scenario_df.attrs["output_path"] = str(output_path)
                outputs[scenario_name] = scenario_df

                tqdm.write(
                    f"Saved {scenario_name}: {output_path}"
                )

    finally:
        if executor is not None:
            executor.shutdown()

    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired BART hyperparameter-sensitivity experiments "
            "for selected simulation scenarios."
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

    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--s2", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="hyperparameter_results",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help=(
            "Number of configuration jobs to run concurrently. "
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

    run_hyperparameter_experiments(
        scenarios=selected_scenarios,
        n=args.n,
        p=args.p,
        s2=args.s2,
        repeats=args.repeats,
        base_seed=args.seed,
        output_dir=args.output_dir,
        n_processes=args.n_processes,
    )


if __name__ == "__main__":
    main()
