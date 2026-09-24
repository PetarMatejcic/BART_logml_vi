import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown


def _as_set(x):
    if pd.isna(x):
        return set()
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return set(x)


def _jaccard_distance(a, b):
    u = a | b
    return 0.0 if not u else 1 - len(a & b) / len(u)


def aggregate_hyperparameter_results(df):
    data = df[df["status"].eq("complete")].copy() if "status" in df else df.copy()

    groups = [
        "scenario", "model_key", "n", "p", "s2", "config_id",
        "varied_parameter", "m", "alpha", "beta", "k"
    ]

    methods = [
        "raw_local", "raw_global_max", "raw_global_se",
        "logml_local", "logml_global_max", "logml_global_se"
    ]

    g = data.groupby(groups, dropna=False, sort=False)
    out = g.size().rename("n_repeats").reset_index()

    for method in methods:
        for metric in ["precision", "recall", "f1", "n_selected"]:
            col = f"{metric}_{method}"
            x = g[col].agg(["mean", "sem"]).rename(
                columns={
                    "mean": f"{col}_mean",
                    "sem": f"{col}_se",
                }
            )
            out = out.merge(x.reset_index(), on=groups)

        col = f"r_miss_{method}"
        x = g[col].apply(lambda s: (s == 1).mean()).rename(f"{col}_rate")
        out = out.merge(x.reset_index(), on=groups)

    return out


def display_hyperparameter_summary(summary_df, method, output_path=None):
    if method not in {"local", "global_se", "global_max"}:
        raise ValueError(
            "method must be 'local', 'global_se', or 'global_max'."
        )

    ref = summary_df[summary_df["config_id"].eq("reference")]
    blocks = []

    for param in ["m", "beta", "k"]:
        d_param = pd.concat([
            summary_df[summary_df["varied_parameter"].eq(param)],
            ref,
        ])

        d_param = d_param.sort_values(param).copy()
        d_param["Parameter"] = {
            "m": "m",
            "beta": "β",
            "k": "k",
        }[param]

        d_param["Value"] = d_param[param]
        d_param["_reference"] = d_param["config_id"].eq("reference")

        blocks.append(d_param)

    d = pd.concat(blocks, ignore_index=True)

    # Use temporary flat column names while constructing/styling.
    out = pd.DataFrame({
        "Parameter": d["Parameter"],
        "Value": d["Value"].map(lambda x: f"{x:g}"),
    })

    styles = pd.DataFrame("", index=d.index, columns=[
        "Parameter",
        "Value",
        "Precision Raw",
        "Precision LogML",
        "Recall Raw",
        "Recall LogML",
        "F1 Raw",
        "F1 LogML",
        "r_miss Raw",
        "r_miss LogML",
        "Selected Raw",
        "Selected LogML",
    ])

    # Pale blue text for the better outcome.
    better_style = "color: #6fa6c9; font-weight: 600"

    # Higher is better.
    for metric, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]:
        raw = f"{metric}_raw_{method}"
        logml = f"{metric}_logml_{method}"

        out[f"{label} Raw"] = [
            f"{mean:.2f} ({se:.2f})"
            for mean, se in zip(
                d[raw + "_mean"],
                d[raw + "_se"],
            )
        ]

        out[f"{label} LogML"] = [
            f"{mean:.2f} ({se:.2f})"
            for mean, se in zip(
                d[logml + "_mean"],
                d[logml + "_se"],
            )
        ]

        styles.loc[
            d[raw + "_mean"] > d[logml + "_mean"],
            f"{label} Raw",
        ] = better_style

        styles.loc[
            d[logml + "_mean"] > d[raw + "_mean"],
            f"{label} LogML",
        ] = better_style

    # Lower r_miss is better.
    raw = f"r_miss_raw_{method}_rate"
    logml = f"r_miss_logml_{method}_rate"

    out["r_miss Raw"] = d[raw].map(lambda x: f"{x:.0%}")
    out["r_miss LogML"] = d[logml].map(lambda x: f"{x:.0%}")

    styles.loc[
        d[raw] < d[logml],
        "r_miss Raw",
    ] = better_style

    styles.loc[
        d[logml] < d[raw],
        "r_miss LogML",
    ] = better_style

    # Number selected is descriptive, so do not highlight either method.
    for weight, name in [
        ("raw", "Raw"),
        ("logml", "LogML"),
    ]:
        col = f"n_selected_{weight}_{method}"

        out[f"Selected {name}"] = [
            f"{mean:.2f} ({se:.2f})"
            for mean, se in zip(
                d[col + "_mean"],
                d[col + "_se"],
            )
        ]

    # Bold the reference hyperparameter value.
    styles.loc[d["_reference"], "Value"] = "font-weight: bold"

    # Keep CSV output simple with one header row.
    if output_path is not None:
        out.to_csv(output_path, index=False)

    # Two-level headers for notebook display.
    multi_columns = pd.MultiIndex.from_tuples([
        ("", "Parameter"),
        ("", "Value"),
        ("Precision", "Raw"),
        ("Precision", "LogML"),
        ("Recall", "Raw"),
        ("Recall", "LogML"),
        ("F1", "Raw"),
        ("F1", "LogML"),
        ("r_miss", "Raw"),
        ("r_miss", "LogML"),
        ("Selected", "Raw"),
        ("Selected", "LogML"),
    ])

    out.columns = multi_columns
    styles.columns = multi_columns

    styler = (
        out.style
        .apply(lambda _: styles, axis=None)
        .hide(axis="index")
        .set_table_styles([
            # Start of each metric block.
            {
                "selector": "th.col2, td.col2",
                "props": [("border-left", "1px solid rgba(140,140,140,0.45)")],
            },
            {
                "selector": "th.col4, td.col4",
                "props": [("border-left", "1px solid rgba(140,140,140,0.45)")],
            },
            {
                "selector": "th.col6, td.col6",
                "props": [("border-left", "1px solid rgba(140,140,140,0.45)")],
            },
            {
                "selector": "th.col8, td.col8",
                "props": [("border-left", "1px solid rgba(140,140,140,0.45)")],
            },
            {
                "selector": "th.col10, td.col10",
                "props": [("border-left", "1px solid rgba(140,140,140,0.45)")],
            },
        ])
    )

    display(styler)


def aggregate_paired_hyperparameter_results(df):
    data = df[df["status"].eq("complete")].copy() if "status" in df else df.copy()
    keys = ["scenario", "model_key", "n", "p", "s2", "repeat", "seed"]
    stats = ["precision", "recall", "f1", "r_miss", "n_selected"]
    methods = ["raw_local", "raw_global_se", "raw_global_max",
               "logml_local", "logml_global_se", "logml_global_max"]

    cols = [f"{s}_{m}" for s in stats for m in methods]
    set_cols = [f"selected_{m}" for m in methods]

    ref = data[data["config_id"].eq("reference")][
        keys + ["m", "beta", "k"] + cols + set_cols
    ]
    alt = data[~data["config_id"].eq("reference")].copy()
    x = alt.merge(ref, on=keys, suffixes=("", "_ref"), validate="many_to_one")

    x["value"] = x.apply(lambda r: r[r["varied_parameter"]], axis=1)
    x["reference_value"] = x.apply(
        lambda r: r[f'{r["varied_parameter"]}_ref'], axis=1)

    truth = x["true_variables"].map(_as_set)

    for c in cols:
        x[f"delta_{c}"] = x[c] - x[f"{c}_ref"]

    for m in methods:
        alt_sets = x[f"selected_{m}"].map(_as_set)
        ref_sets = x[f"selected_{m}_ref"].map(_as_set)

        x[f"delta_tp_{m}"] = [
            len(a & t) - len(r & t) for a, r, t in zip(alt_sets, ref_sets, truth)
        ]
        x[f"delta_fp_{m}"] = [
            len(a - t) - len(r - t) for a, r, t in zip(alt_sets, ref_sets, truth)
        ]
        x[f"jaccard_{m}"] = [
            _jaccard_distance(a, r) for a, r in zip(alt_sets, ref_sets)
        ]

    groups = ["scenario", "model_key", "n", "p", "s2", "config_id",
              "varied_parameter", "value", "reference_value"]
    g = x.groupby(groups, dropna=False, sort=False)
    out = g.size().rename("n_pairs").reset_index()

    summary_cols = [c for c in x if c.startswith(("delta_", "jaccard_"))]
    for c in summary_cols:
        z = g[c].agg(["mean", "sem"]).rename(
            columns={"mean": f"{c}_mean", "sem": f"{c}_se"})
        out = out.merge(z.reset_index(), on=groups)

    for m in methods:
        c = f"n_selected_{m}_ref"
        z = g[c].mean().rename(f"reference_n_selected_{m}").reset_index()
        out = out.merge(z, on=groups)

    return out


def display_paired_summary(
    paired_df,
    method,
    output_path=None,
):
    if method not in {"local", "global_se", "global_max"}:
        raise ValueError(
            "method must be 'local', 'global_se', or 'global_max'."
        )

    d = paired_df.copy()

    d["Parameter"] = d["varied_parameter"].map({
        "m": "m",
        "beta": "β",
        "k": "k",
    })

    d["Value"] = d["value"].map(lambda x: f"{x:g}")

    d["_ord"] = d["varied_parameter"].map({
        "m": 0,
        "beta": 1,
        "k": 2,
    })

    d = (
        d.sort_values(["_ord", "value"])
        .reset_index(drop=True)
    )

    out = pd.DataFrame({
        "Parameter": d["Parameter"],
        "Value": d["Value"],
    })

    # Metrics shown in the notebook.
    for metric, label in [
        ("precision", "Δ Precision"),
        ("recall", "Δ Recall"),
        ("f1", "Δ F1"),
        ("n_selected", "Δ Selected"),
    ]:
        for weight, name in [
            ("raw", "Raw"),
            ("logml", "LogML"),
        ]:
            col = f"delta_{metric}_{weight}_{method}"

            out[f"{label} {name}"] = [
                f"{mean:+.2f} ({se:.2f})"
                for mean, se in zip(
                    d[col + "_mean"],
                    d[col + "_se"],
                )
            ]

    for weight, name in [
        ("raw", "Raw"),
        ("logml", "LogML"),
    ]:
        col = f"jaccard_{weight}_{method}"

        out[f"Jaccard {name}"] = [
            f"{mean:.2f} ({se:.2f})"
            for mean, se in zip(
                d[col + "_mean"],
                d[col + "_se"],
            )
        ]

    # Full CSV includes diagnostic TP / FP changes.
    export_out = out.copy()

    for stat, label in [
        ("delta_tp", "Δ TP"),
        ("delta_fp", "Δ FP"),
    ]:
        for weight, name in [
            ("raw", "Raw"),
            ("logml", "LogML"),
        ]:
            col = f"{stat}_{weight}_{method}"

            export_out[f"{label} {name}"] = [
                f"{mean:+.2f} ({se:.2f})"
                for mean, se in zip(
                    d[col + "_mean"],
                    d[col + "_se"],
                )
            ]

    if output_path is not None:
        export_out.to_csv(
            output_path,
            index=False,
        )

    # Notebook display remains compact.
    out.columns = pd.MultiIndex.from_tuples([
        ("", "Parameter"),
        ("", "Value"),
        ("Δ Precision", "Raw"),
        ("Δ Precision", "LogML"),
        ("Δ Recall", "Raw"),
        ("Δ Recall", "LogML"),
        ("Δ F1", "Raw"),
        ("Δ F1", "LogML"),
        ("Δ Selected", "Raw"),
        ("Δ Selected", "LogML"),
        ("Jaccard", "Raw"),
        ("Jaccard", "LogML"),
    ])

    styler = (
        out.style
        .hide(axis="index")
        .set_table_styles([
            {
                "selector": "th.col2, td.col2",
                "props": [
                    (
                        "border-left",
                        "1px solid rgba(140,140,140,0.45)",
                    )
                ],
            },
            {
                "selector": "th.col4, td.col4",
                "props": [
                    (
                        "border-left",
                        "1px solid rgba(140,140,140,0.45)",
                    )
                ],
            },
            {
                "selector": "th.col6, td.col6",
                "props": [
                    (
                        "border-left",
                        "1px solid rgba(140,140,140,0.45)",
                    )
                ],
            },
            {
                "selector": "th.col8, td.col8",
                "props": [
                    (
                        "border-left",
                        "1px solid rgba(140,140,140,0.45)",
                    )
                ],
            },
            {
                "selector": "th.col10, td.col10",
                "props": [
                    (
                        "border-left",
                        "1px solid rgba(140,140,140,0.45)",
                    )
                ],
            },
        ])
    )

    display(styler)


def aggregate_hyperparameter_sensitivity(df):
    data = (
        df[df["status"].eq("complete")].copy()
        if "status" in df
        else df.copy()
    )

    keys = [
        "scenario",
        "model_key",
        "n",
        "p",
        "s2",
        "repeat",
        "seed",
    ]

    stats = [
        "precision",
        "recall",
        "f1",
        "r_miss",
        "n_selected",
    ]

    methods = [
        "local",
        "global_se",
        "global_max",
    ]

    ref = data[data["config_id"].eq("reference")]

    x = data[~data["config_id"].eq("reference")].merge(
        ref,
        on=keys,
        suffixes=("", "_ref"),
        validate="many_to_one",
    )

    x["value"] = x.apply(
        lambda r: r[r["varied_parameter"]],
        axis=1,
    )

    x["reference_value"] = x.apply(
        lambda r: r[f'{r["varied_parameter"]}_ref'],
        axis=1,
    )

    # Standard metric sensitivities.
    for stat in stats:
        for method in methods:
            for weight in ["raw", "logml"]:
                col = f"{stat}_{weight}_{method}"

                x[f"abs_delta_{col}"] = (
                    x[col] - x[f"{col}_ref"]
                ).abs()

            x[f"sens_diff_{stat}_{method}"] = (
                x[f"abs_delta_{stat}_logml_{method}"]
                - x[f"abs_delta_{stat}_raw_{method}"]
            )

    # True/false-positive sensitivity and selected-set sensitivity.
    truth = x["true_variables"].map(_as_set)

    for method in methods:
        for weight in ["raw", "logml"]:
            alt_sets = x[f"selected_{weight}_{method}"].map(_as_set)
            ref_sets = x[f"selected_{weight}_{method}_ref"].map(_as_set)

            tp_alt = np.array([
                len(a & t)
                for a, t in zip(alt_sets, truth)
            ])

            tp_ref = np.array([
                len(r & t)
                for r, t in zip(ref_sets, truth)
            ])

            fp_alt = np.array([
                len(a - t)
                for a, t in zip(alt_sets, truth)
            ])

            fp_ref = np.array([
                len(r - t)
                for r, t in zip(ref_sets, truth)
            ])

            x[f"abs_delta_tp_{weight}_{method}"] = np.abs(
                tp_alt - tp_ref
            )

            x[f"abs_delta_fp_{weight}_{method}"] = np.abs(
                fp_alt - fp_ref
            )

            x[f"jaccard_{weight}_{method}"] = [
                _jaccard_distance(a, r)
                for a, r in zip(alt_sets, ref_sets)
            ]

        x[f"sens_diff_tp_{method}"] = (
            x[f"abs_delta_tp_logml_{method}"]
            - x[f"abs_delta_tp_raw_{method}"]
        )

        x[f"sens_diff_fp_{method}"] = (
            x[f"abs_delta_fp_logml_{method}"]
            - x[f"abs_delta_fp_raw_{method}"]
        )

        x[f"jaccard_diff_{method}"] = (
            x[f"jaccard_logml_{method}"]
            - x[f"jaccard_raw_{method}"]
        )

    groups = [
        "scenario",
        "model_key",
        "n",
        "p",
        "s2",
        "config_id",
        "varied_parameter",
        "value",
        "reference_value",
    ]

    g = x.groupby(
        groups,
        dropna=False,
        sort=False,
    )

    out = (
        g.size()
        .rename("n_pairs")
        .reset_index()
    )

    cols = [
        c for c in x
        if c.startswith(
            (
                "abs_delta_",
                "sens_diff_",
                "jaccard_",
            )
        )
    ]

    for col in cols:
        z = (
            g[col]
            .agg(["mean", "sem"])
            .rename(
                columns={
                    "mean": f"{col}_mean",
                    "sem": f"{col}_se",
                }
            )
        )

        out = out.merge(
            z.reset_index(),
            on=groups,
        )

    return out


def display_sensitivity_comparison(
    sensitivity_df,
    method,
    output_path=None,
):
    if method not in {"local", "global_se", "global_max"}:
        raise ValueError(
            "method must be 'local', 'global_se', or 'global_max'."
        )

    d = sensitivity_df.copy()

    d["Parameter"] = d["varied_parameter"].map({
        "m": "m",
        "beta": "β",
        "k": "k",
    })

    d["Value"] = d["value"].map(lambda x: f"{x:g}")

    d["_ord"] = d["varied_parameter"].map({
        "m": 0,
        "beta": 1,
        "k": 2,
    })

    d = (
        d.sort_values(["_ord", "value"])
        .reset_index(drop=True)
    )

    # -----------------------------
    # Notebook display
    # -----------------------------
    out = pd.DataFrame({
        "Parameter": d["Parameter"],
        "Value": d["Value"],
    })

    for metric, label in [
        ("precision", "|Δ Precision|"),
        ("recall", "|Δ Recall|"),
        ("f1", "|Δ F1|"),
        ("n_selected", "|Δ Selected|"),
    ]:
        for weight, name in [
            ("raw", "Raw"),
            ("logml", "LogML"),
        ]:
            col = f"abs_delta_{metric}_{weight}_{method}"

            out[f"{label} {name}"] = [
                f"{mean:.2f} ({se:.2f})"
                for mean, se in zip(
                    d[col + "_mean"],
                    d[col + "_se"],
                )
            ]

    # Jaccard is already a nonnegative distance, so no |Δ| notation needed.
    for weight, name in [
        ("raw", "Raw"),
        ("logml", "LogML"),
    ]:
        col = f"jaccard_{weight}_{method}"

        out[f"Jaccard {name}"] = [
            f"{mean:.2f} ({se:.2f})"
            for mean, se in zip(
                d[col + "_mean"],
                d[col + "_se"],
            )
        ]

    # -----------------------------
    # CSV export
    # -----------------------------
    # Start with everything shown in the notebook.
    export_out = out.copy()

    # Add TP / FP diagnostics to the CSV only.
    for stat, label in [
        ("tp", "|Δ TP|"),
        ("fp", "|Δ FP|"),
    ]:
        for weight, name in [
            ("raw", "Raw"),
            ("logml", "LogML"),
        ]:
            col = f"abs_delta_{stat}_{weight}_{method}"

            export_out[f"{label} {name}"] = [
                f"{mean:.2f} ({se:.2f})"
                for mean, se in zip(
                    d[col + "_mean"],
                    d[col + "_se"],
                )
            ]

    if output_path is not None:
        export_out.to_csv(
            output_path,
            index=False,
        )

    # -----------------------------
    # Grouped notebook headers
    # -----------------------------
    out.columns = pd.MultiIndex.from_tuples([
        ("", "Parameter"),
        ("", "Value"),
        ("|Δ Precision|", "Raw"),
        ("|Δ Precision|", "LogML"),
        ("|Δ Recall|", "Raw"),
        ("|Δ Recall|", "LogML"),
        ("|Δ F1|", "Raw"),
        ("|Δ F1|", "LogML"),
        ("|Δ Selected|", "Raw"),
        ("|Δ Selected|", "LogML"),
        ("Jaccard", "Raw"),
        ("Jaccard", "LogML"),
    ])

    styler = (
        out.style
        .hide(axis="index")
        .set_table_styles([
            {
                "selector": "th.col2, td.col2",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col4, td.col4",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col6, td.col6",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col8, td.col8",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col10, td.col10",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
        ])
    )

    display(styler)

    display(Markdown(
        "*Values are mean absolute changes from the reference configuration; "
        "parentheses contain SEs. Jaccard distance measures change in the "
        "selected-variable set, with 0 indicating identical sets.*"
    ))


def plot_hyperparameter_sensitivity(
    sensitivity_df,
    method,
    output_path=None,
):
    if method not in {"local", "global_se", "global_max"}:
        raise ValueError("Invalid method.")

    params = [
        ("m", r"$m$"),
        ("beta", r"$\beta$"),
        ("k", r"$k$"),
    ]

    metrics = [
        ("recall", r"$|\Delta|$ Recall", 1),
        ("f1", r"$|\Delta|$ F1", 1),
        ("r_miss", r"$|\Delta|$ $r_{\rm miss}$ (pp)", 100),
        ("n_selected", r"$|\Delta|$ \# selected", 1),
        ("jaccard", "Jaccard dist.", 1),
    ]

    fig, axes = plt.subplots(
        5,
        3,
        figsize=(8.2, 7.2),
        dpi=180,
        sharey="row",
    )

    for j, (param, title) in enumerate(params):
        d = sensitivity_df[
            sensitivity_df["varied_parameter"].eq(param)
        ].sort_values("value")

        ref = float(d["reference_value"].iloc[0])

        for i, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[i, j]

            for weight, label, marker, linestyle in [
                ("raw", "Raw", "o", "-"),
                ("logml", "LogML", "s", "--"),
            ]:
                if metric == "jaccard":
                    col = f"jaccard_{weight}_{method}"
                else:
                    col = f"abs_delta_{metric}_{weight}_{method}"

                x = np.r_[
                    d["value"].to_numpy(float),
                    ref,
                ]

                y = np.r_[
                    d[col + "_mean"].to_numpy(float) * scale,
                    0.0,
                ]

                se = np.r_[
                    d[col + "_se"].to_numpy(float) * scale,
                    0.0,
                ]

                order = np.argsort(x)

                ax.errorbar(
                    x[order],
                    y[order],
                    yerr=se[order],
                    marker=marker,
                    linestyle=linestyle,
                    color="black",
                    markerfacecolor="white",
                    markersize=3.8,
                    linewidth=0.9,
                    capsize=2,
                    label=label,
                )

            ticks = sorted(
                set(d["value"].tolist() + [ref])
            )
            ax.set_xticks(ticks)

            for tick, value in zip(
                ax.get_xticklabels(),
                ticks,
            ):
                if value == ref:
                    tick.set_fontweight("bold")

            ax.set_ylim(bottom=0)

            ax.grid(
                axis="y",
                color="0.9",
                linewidth=0.5,
            )

            ax.tick_params(
                axis="both",
                labelsize=7,
            )

            if i == 0:
                ax.set_title(
                    title,
                    fontsize=10,
                )

            if j == 0:
                ax.set_ylabel(
                    ylabel,
                    fontsize=8,
                )

            if i == len(metrics) - 1:
                ax.set_xlabel(
                    "Hyperparameter value",
                    fontsize=8,
                )

    # Put the legend unobtrusively in the upper-right panel.
    axes[0, 2].legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        borderaxespad=0.2,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=400,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)

    display(Markdown(
        "*Raw: solid line with open circles; LogML: dashed line with open "
        "squares. Error bars show ±1 SE. Bold x-axis ticks indicate the "
        "reference hyperparameter value.*"
    ))