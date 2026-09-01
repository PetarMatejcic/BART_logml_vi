import numpy as np
import pandas as pd

from genbart import BartVariableSelector

def run_scenario(datagen, n: int, p: int, s2: float, model, model_params, repeats: int = 1):
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
        )

        selector.fit(X, y)

        vi_result_raw = selector.get_result("raw")
        vi_result_logl = selector.get_result("logml")

        results_raw.append(
            np.vstack(
            (
                vi_result_raw.importance,
                vi_result_raw.thresholds()
            )
            )
        )

        results_logl.append(
            np.vstack(
            (
                vi_result_logl.importance,
                vi_result_logl.thresholds()
            )
            )
        )

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

def statistics(preds, truth):
    prec = np.array(
        [precision(p, truth) for p in preds]
    )
    rec = np.array(
        [recall(p, truth) for p in preds]
    )
    f1 = np.array(
        [f1_score(p, truth) for p in preds]
    )

    return prec, rec, f1


def summarise_results(raw, logl, truth):
    raw_masks = [r[0, :] > r[1, :]
                 for r in raw]
    logl_masks = [l[0, :] > l[1, :]
                  for l in logl]

    prec_raw, rec_raw, f1_raw = statistics(raw_masks, truth)
    prec_logl, rec_logl, f1_logl = statistics(logl_masks, truth)

    df = pd.DataFrame({
        "precision_raw": prec_raw,
        "recall_raw": rec_raw,
        "f1_raw": f1_raw,
        "precision_logl": prec_logl,
        "recall_logl": rec_logl,
        "f1_logl": f1_logl,
    })

    df.index.name = "repeat"

    return df