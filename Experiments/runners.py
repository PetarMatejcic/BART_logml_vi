import numpy as np
import pandas as pd

from genbart import BartVariableSelector

def run_scenario(
    datagen,
    n: int,
    p: int,
    s2: float,
    model,
    model_params,
    repeats: int = 1,
    random_state: int | None = None,
):
    s = np.sqrt(s2)

    results_raw = []
    results_logl = []

    for r in range(repeats):
        X, y = datagen(n, p, s)

        selector = BartVariableSelector(
            model,
            model_params,
            n_permutations=10,
            n_repeats=1,
            importance_kind="raw",
            random_state=random_state,
        )

        selector.fit(X, y)

        vi_result_raw = selector.get_result("raw")
        vi_result_logl = selector.get_result("logml")

        results_raw.append({
            method: vi_result_raw.selected_mask(method)
            for method in ("local", "global_max", "global_se")
        })

        results_logl.append({
            method: vi_result_logl.selected_mask(method)
            for method in ("local", "global_max", "global_se")
        })

    return results_raw, results_logl

def precision(pred, truth):
    tp = np.sum(pred & truth)
    fp = np.sum(pred & ~truth)
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall(pred, truth):
    tp = np.sum(pred & truth)
    fn = np.sum(~pred & truth)
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def f1_score(pred, truth):
    p = precision(pred, truth)
    r = recall(pred, truth)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def r_miss(pred, truth):
    return int(np.any(truth & ~pred))


def n_selected(pred):
    return int(np.sum(pred))


def summarise_results(raw, logl, truth):
    rows = []

    for raw_result, logl_result in zip(raw, logl):
        row = {}

        for method, pred in raw_result.items():
            row[f"precision_raw_{method}"] = precision(pred, truth)
            row[f"recall_raw_{method}"] = recall(pred, truth)
            row[f"f1_raw_{method}"] = f1_score(pred, truth)
            row[f"r_miss_raw_{method}"] = r_miss(pred, truth)
            row[f"n_selected_raw_{method}"] = n_selected(pred)

        for method, pred in logl_result.items():
            row[f"precision_logl_{method}"] = precision(pred, truth)
            row[f"recall_logl_{method}"] = recall(pred, truth)
            row[f"f1_logl_{method}"] = f1_score(pred, truth)
            row[f"r_miss_logl_{method}"] = r_miss(pred, truth)
            row[f"n_selected_logl_{method}"] = n_selected(pred)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = "repeat"

    return df