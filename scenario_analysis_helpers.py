import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display, Markdown


def aggregate_experiment_results(df):
    """
    Aggregate experiment results across repeats.

    Parameters
    ----------
    df : pandas.DataFrame
        Experiment results containing one row per repeat.
    """

    group_cols = ["scenario", "model_key", "n", "p", "s2"]

    # If status is available, only aggregate successfully completed runs.
    data = df.copy()
    if "status" in data.columns:
        data = data[data["status"] == "complete"].copy()

    methods = [
        "raw_local",
        "raw_global_max",
        "raw_global_se",
        "logml_local",
        "logml_global_max",
        "logml_global_se",
    ]

    grouped = data.groupby(group_cols, dropna=False, sort=False)

    # Start summary table with the number of completed repeats.
    summary_df = (
        grouped.size()
        .rename("n_repeats")
        .reset_index()
    )

    for method in methods:
        for metric in ["precision", "recall", "f1"]:
            col = f"{metric}_{method}"

            means = grouped[col].mean()
            ses = grouped[col].sem()

            metric_summary = pd.DataFrame({
                f"{col}_mean": means,
                f"{col}_se": ses,
            }).reset_index()

            summary_df = summary_df.merge(
                metric_summary,
                on=group_cols,
                how="left",
            )

        # r_miss is binary, so its mean is the proportion of repeats
        # for which r_miss == 1.
        r_miss_col = f"r_miss_{method}"

        r_miss = (
            grouped[r_miss_col]
            .apply(lambda x: (x == 1).mean())
            .rename(f"{r_miss_col}_rate")
            .reset_index()
        )

        summary_df = summary_df.merge(
            r_miss,
            on=group_cols,
            how="left",
        )

    # Build long-format n_selected frequency table.
    n_selected_parts = []

    for method in methods:
        col = f"n_selected_{method}"

        counts = (
            data
            .groupby(group_cols + [col], dropna=False)
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={col: "n_selected"})
        )

        counts["method"] = method

        counts = counts[
            group_cols
            + ["method", "n_selected", "count"]
        ]

        n_selected_parts.append(counts)

    n_selected_df = pd.concat(
        n_selected_parts,
        ignore_index=True,
    )

    n_selected_df = n_selected_df.sort_values(
        group_cols + ["method", "n_selected"],
        ignore_index=True,
    )

    return summary_df, n_selected_df


def display_summary_table(summary_df, method, output_path=None):
    if method not in {"local", "global_max", "global_se"}:
        raise ValueError(
            "method must be 'local', 'global_max', or 'global_se'."
        )

    method_labels = {
        "local": "Local threshold",
        "global_se": "Global-SE threshold",
        "global_max": "Global-max threshold",
    }

    df = summary_df.copy()

    # Build flat table first so CSV output remains simple.
    out = pd.DataFrame({
        "n": df["n"],
        "p": df["p"],
        "s2": df["s2"].map(lambda x: f"{x:g}"),
    })

    flat_columns = [
        "n", "p", "s2",
        "Precision Raw", "Precision LogML",
        "Recall Raw", "Recall LogML",
        "F1 Raw", "F1 LogML",
        "r_miss Raw", "r_miss LogML",
    ]

    styles = pd.DataFrame(
        "",
        index=df.index,
        columns=flat_columns,
    )

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
            f"{mean:.3f} ({se:.2f})"
            for mean, se in zip(
                df[raw + "_mean"],
                df[raw + "_se"],
            )
        ]

        out[f"{label} LogML"] = [
            f"{mean:.3f} ({se:.2f})"
            for mean, se in zip(
                df[logml + "_mean"],
                df[logml + "_se"],
            )
        ]

        styles.loc[
            df[raw + "_mean"] > df[logml + "_mean"],
            f"{label} Raw",
        ] = better_style

        styles.loc[
            df[logml + "_mean"] > df[raw + "_mean"],
            f"{label} LogML",
        ] = better_style

    # Lower r_miss is better.
    raw = f"r_miss_raw_{method}_rate"
    logml = f"r_miss_logml_{method}_rate"

    out["r_miss Raw"] = df[raw].map(lambda x: f"{x:.0%}")
    out["r_miss LogML"] = df[logml].map(lambda x: f"{x:.0%}")

    styles.loc[
        df[raw] < df[logml],
        "r_miss Raw",
    ] = better_style

    styles.loc[
        df[logml] < df[raw],
        "r_miss LogML",
    ] = better_style

    # Keep exported CSV with one simple header row.
    if output_path is not None:
        out.to_csv(output_path, index=False)

    # Group Raw and LogML under each metric for notebook display.
    multi_columns = pd.MultiIndex.from_tuples([
        ("", "n"),
        ("", "p"),
        ("", "s2"),
        ("Precision", "Raw"),
        ("Precision", "LogML"),
        ("Recall", "Raw"),
        ("Recall", "LogML"),
        ("F1", "Raw"),
        ("F1", "LogML"),
        ("r_miss", "Raw"),
        ("r_miss", "LogML"),
    ])

    out.columns = multi_columns
    styles.columns = multi_columns

    styler = (
        out.style
        .apply(lambda _: styles, axis=None)
        .hide(axis="index")
        .set_table_styles([
            # Faint separator at the start of each metric block.
            {
                "selector": "th.col3, td.col3",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col5, td.col5",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col7, td.col7",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
            {
                "selector": "th.col9, td.col9",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
        ])
    )

    display(Markdown(f"#### {method_labels[method]}"))
    display(styler)


def display_n_selected_summary(
    n_selected_df,
    true_n_selected,
    output_path=None,
):
    rows = []

    for (n, p, s2, method), d in n_selected_df.groupby(
        ["n", "p", "s2", "method"]
    ):
        total = d["count"].sum()

        mean = (
            d["n_selected"] * d["count"]
        ).sum() / total

        exact = (
            d.loc[
                d["n_selected"] == true_n_selected,
                "count",
            ].sum()
            / total
        )

        under = (
            d.loc[
                d["n_selected"] < true_n_selected,
                "count",
            ].sum()
            / total
        )

        over = (
            d.loc[
                d["n_selected"] > true_n_selected,
                "count",
            ].sum()
            / total
        )

        excess = (
            np.maximum(
                d["n_selected"] - true_n_selected,
                0,
            )
            * d["count"]
        ).sum() / total

        weight, threshold = method.split("_", 1)

        rows.append([
            n,
            p,
            s2,
            threshold,
            weight,
            mean,
            exact,
            under,
            over,
            excess,
        ])

    x = pd.DataFrame(
        rows,
        columns=[
            "n",
            "p",
            "s2",
            "method",
            "weight",
            "mean",
            "exact",
            "under",
            "over",
            "excess",
        ],
    )

    labels = {
        "local": "Local",
        "global_se": "Global SE",
        "global_max": "Global Max",
    }

    method_order = {
        "Global SE": 0,
        "Local": 1,
        "Global Max": 2,
    }

    table_rows = []

    for keys, d in x.groupby(
        ["n", "p", "s2", "method"],
        sort=False,
    ):
        vals = {
            row["weight"]: row
            for _, row in d.iterrows()
        }

        table_rows.append([
            keys[0],
            keys[1],
            keys[2],
            labels[keys[3]],

            vals["raw"]["mean"],
            vals["logml"]["mean"],

            vals["raw"]["exact"],
            vals["logml"]["exact"],

            vals["raw"]["under"],
            vals["logml"]["under"],

            vals["raw"]["over"],
            vals["logml"]["over"],

            vals["raw"]["excess"],
            vals["logml"]["excess"],
        ])

    out = pd.DataFrame(
        table_rows,
        columns=[
            "n",
            "p",
            "s2",
            "Method",
            "Mean selected Raw",
            "Mean selected LogML",
            "Exact Raw",
            "Exact LogML",
            "Under Raw",
            "Under LogML",
            "Over Raw",
            "Over LogML",
            "Mean excess Raw",
            "Mean excess LogML",
        ],
    )

    # Put Global SE first within each scenario variant.
    out["_method_order"] = out["Method"].map(method_order)

    out = (
        out
        .sort_values(["n", "p", "s2", "_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    # Identify the first row of each new scenario variant.
    group_start = (
        out[["n", "p", "s2"]]
        .ne(out[["n", "p", "s2"]].shift())
        .any(axis=1)
    )
    group_start.iloc[0] = False

    # Flat formatted table for CSV export.
    export_out = out.copy()

    export_out["s2"] = export_out["s2"].map(lambda x: f"{x:g}")

    for col in [
        "Mean selected Raw",
        "Mean selected LogML",
        "Mean excess Raw",
        "Mean excess LogML",
    ]:
        export_out[col] = export_out[col].map(lambda x: f"{x:.2f}")

    for col in [
        "Exact Raw",
        "Exact LogML",
        "Under Raw",
        "Under LogML",
        "Over Raw",
        "Over LogML",
    ]:
        export_out[col] = export_out[col].map(lambda x: f"{x:.0%}")

    if output_path is not None:
        export_out.to_csv(output_path, index=False)

    # Notebook display.
    display_out = export_out.copy()

    display_out.columns = pd.MultiIndex.from_tuples([
        ("", "n"),
        ("", "p"),
        ("", "s2"),
        ("", "Method"),
        ("Mean selected", "Raw"),
        ("Mean selected", "LogML"),
        ("Exact", "Raw"),
        ("Exact", "LogML"),
        ("Under", "Raw"),
        ("Under", "LogML"),
        ("Over", "Raw"),
        ("Over", "LogML"),
        ("Mean excess", "Raw"),
        ("Mean excess", "LogML"),
    ])

    # Horizontal dividers between scenario variants.
    row_styles = pd.DataFrame(
        "",
        index=display_out.index,
        columns=display_out.columns,
    )

    for idx in display_out.index[group_start]:
        row_styles.loc[idx, :] = (
            "border-top: 1.5px solid rgba(180,180,180,0.65)"
        )

    styler = (
        display_out.style
        .apply(lambda _: row_styles, axis=None)
        .hide(axis="index")
        .set_table_styles([
            # Vertical metric separators.
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
            {
                "selector": "th.col12, td.col12",
                "props": [
                    ("border-left", "1px solid rgba(140,140,140,0.45)")
                ],
            },
        ])
    )

    display(Markdown("#### Number of selected variables"))
    display(styler)