from __future__ import annotations

# Centralized header mapping for all outputs
HEADER_MAP = {
    "metric": "Metric",
    "basis": "Basis",
    "log": "Log",
    "rho": "Rho",
    "p": "P",
    "RelCI_50": "RelCI_50",
    "RelCI_250": "RelCI_250",
    "RelCI_500": "RelCI_500",
    "plateau_n": "Plateau_n",
    "improvement": "Improvement",
    # Add more if needed
}


def read_master_csv(path: str) -> pd.DataFrame:
    """Read a master_table CSV and normalize headers to canonical column names.

    This maps friendly output headers (from HEADER_MAP) back to the canonical
    internal column names used throughout the code (e.g., 'Metric' -> 'metric').
    """
    df = pd.read_csv(path)
    inv_header_map = {v: k for k, v in HEADER_MAP.items()}
    try:
        return df.rename(columns=inv_header_map)
    except Exception:
        return df


import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------


def _escape_latex(text: Any) -> str:
    """Escape underscores in text, preserving \textit{...} wrappers."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text)
    if s.startswith("\\textit{") and s.endswith("}"):
        return s
    return s.replace("_", "\\_")


def _validate_improvement_inputs(
    improvement_per_log_df: Optional[pd.DataFrame],
    improvement_summary_df: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Light schema checks + type normalization."""
    ipl = improvement_per_log_df
    ims = improvement_summary_df

    if ipl is not None:
        required = {"log", "metric", "improved"}
        missing = required - set(ipl.columns)
        if missing:
            raise ValueError(f"improvement_per_log_df missing columns: {missing}")
        # normalize dtypes
        ipl = ipl.copy()
        ipl["log"] = ipl["log"].astype(str)
        ipl["metric"] = ipl["metric"].astype(str)
        # coerce to bool (allow 0/1, 'True'/'False', etc.)
        ipl["improved"] = ipl["improved"].astype(bool)

    if ims is not None:
        required = {"metric", "share_improved"}
        missing = required - set(ims.columns)
        if missing:
            raise ValueError(f"improvement_summary_df missing columns: {missing}")
        ims = ims.copy()
        ims["metric"] = ims["metric"].astype(str)
        # numeric share in [0,1]
        ims["share_improved"] = pd.to_numeric(ims["share_improved"], errors="coerce")

    return ipl, ims


# ----------------------------
# 1) Build & save CSV (returns DataFrame too)
# ----------------------------


def build_master_table_csv(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,  # rows=metrics, cols=logs -> rho
    pval_df: pd.DataFrame,  # rows=metrics, cols=logs -> p
    plateau_df: pd.DataFrame,  # rows=metrics, cols=logs -> plateau n (or NaN)
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: Optional[List[int]] = None,
    measure_basis_map: Optional[Dict[str, str]] = None,
    pretty_plateau: bool = True,
    improvement_per_log_df: Optional[
        pd.DataFrame
    ] = None,  # cols: log, metric, improved (bool)
    improvement_summary_df: Optional[
        pd.DataFrame
    ] = None,  # cols: metric, share_improved
) -> Tuple[str, pd.DataFrame]:
    """
    Assemble the master table and write CSV.
    Returns (csv_path, table_df_with_mean_rows).

    - Integrates per-log 'improved' if provided (merged by log+metric).
    - Integrates mean-row 'share_improved' if provided (merged by metric).
    - Writes both into a single CSV column 'improvement':
        * Per-log rows -> "Yes" if improved else ""
        * Mean rows    -> share_improved formatted to 4 decimals
    """
    if ref_sizes is None:
        ref_sizes = [50, 250, 500]

    improvement_per_log_df, improvement_summary_df = _validate_improvement_inputs(
        improvement_per_log_df, improvement_summary_df
    )

    rows = []

    # Collect union of all metric columns if not provided
    if metric_columns is None:
        all_metrics: set[str] = set()
        for df in measures_per_log.values():
            id_cols = {"sample_size", "sample_id"}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update([c for c in numeric_cols if c not in id_cols])
        metric_columns = sorted(all_metrics)

    # Sort helpers
    metric_order = {metric: idx for idx, metric in enumerate(metric_columns)}

    logs_sorted = sorted(measures_per_log.keys())

    for log in logs_sorted:
        rel_df = sample_ci_rel_width_per_log.get(log)
        if rel_df is None:
            raise ValueError(f"Missing relative CI width DF for log '{log}'.")
        rel_df = rel_df.set_index("sample_size").sort_index()

        metric_cols = [m for m in metric_columns if m in measures_per_log[log].columns]
        if not metric_cols:
            continue

        for m in metric_cols:
            rho = np.nan
            pval = np.nan
            if (m in corr_df.index) and (log in corr_df.columns):
                rho = corr_df.at[m, log]
            if (m in pval_df.index) and (log in pval_df.columns):
                pval = pval_df.at[m, log]

            plateau = np.nan
            if (m in plateau_df.index) and (log in plateau_df.columns):
                plateau = plateau_df.at[m, log]

            relci = {}
            for n in ref_sizes:
                val = rel_df[m].get(n, np.nan) if m in rel_df.columns else np.nan
                relci[f"RelCI_{n}"] = val

            basis = measure_basis_map.get(m, "—") if measure_basis_map else "—"

            row = {
                "metric": m,
                "basis": basis,
                "log": str(log),
                "rho": rho,
                "p": pval,
                **relci,
                "plateau_n": (
                    "---"
                    if (pretty_plateau and (not np.isfinite(plateau)))
                    else plateau
                ),
            }

            # Merge per-log improvement flag if provided
            if improvement_per_log_df is not None:
                sub = improvement_per_log_df[
                    (improvement_per_log_df["log"] == str(log))
                    & (improvement_per_log_df["metric"] == str(m))
                ]
                if not sub.empty:
                    row["improved"] = bool(sub.iloc[0]["improved"])

            rows.append(row)

    if not rows:
        raise ValueError("No rows produced; check inputs and metric names.")

    # Build DataFrame in requested order
    ref_cols = [f"RelCI_{n}" for n in ref_sizes]
    base_cols = ["metric", "basis", "log", "rho", "p"] + ref_cols + ["plateau_n"]
    # keep originals internally (not necessarily in CSV)
    if improvement_per_log_df is not None and any(("improved" in r) for r in rows):
        base_cols.append("improved")

    table_df = pd.DataFrame(rows)
    table_df = table_df[[c for c in base_cols if c in table_df.columns]]

    # ---- Mean rows per (metric, basis) ----
    numeric_cols = ["rho", "p"] + [c for c in ref_cols if c in table_df.columns]
    grp = table_df.groupby(["metric", "basis"], dropna=False)
    means = grp[numeric_cols].mean(numeric_only=True).reset_index()
    means.insert(2, "log", "")  # blank identifier on mean row
    means["_is_mean"] = True
    means["plateau_n"] = "---"

    # Merge share_improved on mean rows only
    if improvement_summary_df is not None:
        means = means.merge(
            improvement_summary_df[["metric", "share_improved"]],
            on="metric",
            how="left",
        )
        # carry the column in per-log rows too (will be blank there)
        table_df["share_improved"] = np.nan

    # Per-log 'improved' should not appear on mean rows
    means["improved"] = np.nan

    # Combine
    table_df["_is_mean"] = False
    table_df = pd.concat([table_df, means], ignore_index=True, sort=False)

    # ---- Create combined 'improvement' column ----
    # Only create if improvement_per_log_df is provided
    if improvement_per_log_df is not None:
        # Per-log: "Yes" if improved else ""
        if "improved" in table_df.columns:
            table_df["improvement"] = table_df["improved"].apply(
                lambda v: "Yes" if (isinstance(v, (bool, np.bool_)) and v) else ""
            )
        else:
            table_df["improvement"] = ""

        # Mean rows: write share_improved with 4 decimals (overwriting per-log text in those rows)
        if "share_improved" in table_df.columns:
            mean_mask = table_df["_is_mean"] == True
            table_df.loc[mean_mask, "improvement"] = table_df.loc[
                mean_mask, "share_improved"
            ].apply(lambda x: ("" if pd.isna(x) else f"{float(x):.4f}"))

    # Sorting: metric (as-is) → log (as-is); mean rows last
    table_df["_metric_order"] = table_df["metric"].map(
        lambda x: metric_order.get(x, len(metric_columns))
    )
    table_df["_log_order"] = table_df["_is_mean"].astype(int)

    table_df = (
        table_df.sort_values(["_metric_order", "_log_order", "log", "metric"])
        .drop(columns=["_metric_order", "_log_order"])
        .reset_index(drop=True)
    )

    # Final CSV order: include only the combined column (not the separate ones)
    cols_order = ["metric", "basis", "log", "rho", "p"] + ref_cols + ["plateau_n"]
    # Only include improvement column if improvement_per_log_df is provided
    if improvement_per_log_df is not None:
        cols_order.append("improvement")
    cols_order_csv = [c for c in cols_order if c in table_df.columns]
    # keep helper flag at end if present (for downstream LaTeX)
    if "_is_mean" in table_df.columns:
        cols_order_csv = cols_order_csv + ["_is_mean"]
    # (We keep 'improved' and 'share_improved' in the DataFrame but omit from CSV to avoid duplication)

    # Prepare CSV copy so we can format and suppress repeats without mutating the
    # internal DataFrame used for LaTeX rendering.
    csv_df = table_df[cols_order_csv].copy()

    # For CSV output keep raw numeric values (no rounding/trimming) so downstream
    # tools can consume the full precision. Do not format numeric columns here.

    # For CSV output we keep metric names repeated on every line (machine-readable)

    # ---- Apply central header mapping for CSV output ----
    csv_df.columns = [HEADER_MAP.get(c, c) for c in csv_df.columns]
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    csv_df.to_csv(out_csv_path, index=False)

    # Return path and the original (unformatted) table slice for downstream LaTeX rendering
    return out_csv_path, table_df[cols_order_csv]


# ----------------------------
# 2) Render LaTeX
# ----------------------------


def render_master_table_latex(
    table_df: pd.DataFrame,
    out_tex_path: str,
    caption: str = "Assessment of Measures before Applying Remedies",
    label: str = "tab:master_table",
) -> str:
    """
    Pretty LaTeX rendering for the master table DataFrame produced by build_master_table_csv(...).
    - Suppresses repeats for 'log' and 'basis' within (metric,basis) blocks
    - Mean rows italicized; metric/basis/log blanked on mean rows
    - Always 4 decimals for numeric; no scientific notation
    - Escapes underscores in headers and cell content
    - Wrapped in a full table environment
    """
    disp = table_df.copy()

    # Suppress repeats for 'log' and 'basis' within (metric,basis) blocks
    suppress_cols = [c for c in ["log", "basis"] if c in disp.columns]
    last_seen: Dict[str, Tuple[Any, Any]] = {c: (None, None) for c in suppress_cols}
    if suppress_cols:
        for idx, row in disp.iterrows():
            group_key = (row["metric"], row["basis"])
            for c in suppress_cols:
                prev_val, prev_key = last_seen[c]
                if prev_key != group_key:
                    prev_val = None
                if row[c] == prev_val:
                    disp.at[idx, c] = ""
                last_seen[c] = (row[c], group_key)

    # Mean row: blank metric/basis/log completely
    mean_mask = (
        disp["_is_mean"] == True
        if "_is_mean" in disp.columns
        else pd.Series(False, index=disp.index)
    )
    for col in ["metric", "basis", "log"]:
        if col in disp.columns:
            disp.loc[mean_mask, col] = ""

    # ---- Fixed 4-decimal formatting for numerics ----
    ref_cols = [c for c in disp.columns if c.startswith("RelCI_")]
    numeric_cols_present = [
        c for c in disp.columns if c in (["rho", "p"] + ref_cols + ["share_improved"])
    ]

    def fmt_num(x: Any) -> str:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return _escape_latex(x)

    for col in numeric_cols_present:
        disp[col] = disp[col].map(fmt_num)

    # Format 'improved' (bool) → Yes / ""
    # Note: 'improvement' column (if present) is handled generically below as a text column
    if "improved" in disp.columns:
        disp["improved"] = disp["improved"].map(
            lambda v: (
                "Yes"
                if (isinstance(v, (bool, np.bool_)) and v)
                else (
                    ""
                    if (v is None or (isinstance(v, float) and pd.isna(v)))
                    else str(v)
                )
            )
        )
        # blank 'improved' in mean rows explicitly
        disp.loc[mean_mask, "improved"] = ""

    # Escape underscores in text columns (including 'improvement' if present)
    for col in disp.columns:
        if col in numeric_cols_present or col in {"_is_mean"}:
            continue
        disp[col] = disp[col].map(_escape_latex)

    # Bold rho and p if p < 0.05 (including mean rows, even if italicized)
    if "p" in disp.columns and "p" in table_df.columns:
        p_numeric = pd.to_numeric(table_df["p"], errors="coerce")
        sig_mask = p_numeric.apply(lambda v: False if pd.isna(v) else (float(v) < 0.05))
        for idx, is_sig in sig_mask.items():
            if not is_sig:
                continue
            for col in ("rho", "p"):
                if col in disp.columns:
                    val = disp.at[idx, col]
                    if val not in (None, "", np.nan):
                        disp.at[idx, col] = f"\\textbf{{{val}}}"

    # Italicize entire mean row (after formatting/escaping and bolding)
    for col in disp.columns:
        if col == "_is_mean":
            continue
        disp.loc[mean_mask, col] = disp.loc[mean_mask, col].apply(
            lambda s: (
                s
                if (isinstance(s, str) and s.startswith("\\textit{"))
                or s in (None, "", np.nan)
                else f"\\textit{{{s}}}"
            )
        )

    # Drop helper
    if "_is_mean" in disp.columns:
        disp = disp.drop(columns=["_is_mean"])

    # Apply central header mapping and escape for LaTeX
    disp.columns = [_escape_latex(HEADER_MAP.get(c, c)) for c in disp.columns]

    # Build LaTeX tabular
    tabular_str = disp.to_latex(index=False, escape=False, na_rep="")

    # Wrap with table env
    os.makedirs(os.path.dirname(out_tex_path) or ".", exist_ok=True)
    table_env = (
        "\\begin{table}[Ht]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        "\\tiny\n"
        f"{tabular_str}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)

    return out_tex_path


# ----------------------------
# Convenience wrapper
# ----------------------------


def create_master_table(
    measures_per_log: Dict[str, pd.DataFrame],
    sample_ci_rel_width_per_log: Dict[str, pd.DataFrame],
    corr_df: pd.DataFrame,
    pval_df: pd.DataFrame,
    plateau_df: pd.DataFrame,
    out_csv_path: str,
    metric_columns: Optional[List[str]] = None,
    ref_sizes: Optional[List[int]] = None,
    measure_basis_map: Optional[Dict[str, str]] = None,
    pretty_plateau: bool = True,
    caption: str = "Assessment of Measures before Applying Remedies",
    label: str = "tab:master_table",
    improvement_per_log_df: Optional[pd.DataFrame] = None,
    improvement_summary_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, str]:
    """
    Build CSV first, then render LaTeX via the helper.
    """
    csv_path, table_df = build_master_table_csv(
        measures_per_log=measures_per_log,
        sample_ci_rel_width_per_log=sample_ci_rel_width_per_log,
        corr_df=corr_df,
        pval_df=pval_df,
        plateau_df=plateau_df,
        out_csv_path=out_csv_path,
        metric_columns=metric_columns,
        ref_sizes=ref_sizes,
        measure_basis_map=measure_basis_map,
        pretty_plateau=pretty_plateau,
        improvement_per_log_df=improvement_per_log_df,
        improvement_summary_df=improvement_summary_df,
    )

    out_tex_path = os.path.splitext(csv_path)[0] + ".tex"
    render_master_table_latex(
        table_df=table_df,
        out_tex_path=out_tex_path,
        caption=caption,
        label=label,
    )
    return csv_path, out_tex_path


def write_means_only_table_from_master_csv(
    master_csv_path: str,
    means_csv_path: str,
    means_tex_path: str,
    caption: str = "Assessment (Means Across Logs)",
    label: str = "tab:master_table_means",
) -> None:
    """
    Create a compact 'means-only' table from the full master CSV produced earlier.

    - Keeps 'metric' and 'basis'
    - Drops the 'log' column entirely
    - Includes only rows where _is_mean == True
    - Formats all numeric values with exactly 4 decimals (no scientific notation)
    - Escapes underscores in headers and cell contents
    - Wraps LaTeX tabular in a full table environment
    """
    df = read_master_csv(master_csv_path)

    # Guard: must have the mean-flag column
    if "_is_mean" not in df.columns:
        raise ValueError(
            "Master CSV is missing '_is_mean'. Please generate with the full function first."
        )

    # Map capitalized headers back to canonical names for processing
    inv_header_map = {v: k for k, v in HEADER_MAP.items()}
    df = df.rename(columns=inv_header_map)

    # Filter to mean rows only
    m = df[df["_is_mean"] == True].copy()

    # Keep columns: metric, basis, (drop log), then the numeric/result columns in original order
    # Identify likely result columns
    result_cols = [
        c for c in m.columns if c not in {"metric", "basis", "log", "_is_mean"}
    ]
    # Rebuild ordered columns
    cols = ["metric", "basis"] + result_cols
    m = m[cols]

    # ---- Format values: always 4 decimals for numeric, no exponent ----
    def _fmt_num(x: Any) -> str:
        if pd.isna(x):
            return ""
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    numeric_cols = [c for c in m.columns if pd.api.types.is_numeric_dtype(m[c])]
    for c in numeric_cols:
        m[c] = m[c].map(_fmt_num)

    # ---- Bold rho and p in means LaTeX if p < 0.05 ----
    if "p" in m.columns:
        # use numeric p values before formatting where possible
        p_numeric = pd.to_numeric(
            m["p"].map(lambda x: x if x != "" else np.nan), errors="coerce"
        )
        sig_mask = p_numeric.apply(lambda v: False if pd.isna(v) else (float(v) < 0.05))
        for idx, is_sig in sig_mask.items():
            if not is_sig:
                continue
            for col in ("rho", "p"):
                if col in m.columns:
                    val = m.at[idx, col]
                    if val not in (None, "", np.nan):
                        # val is already a formatted string (4 decimals) — wrap with \textbf{}
                        m.at[idx, col] = f"\\textbf{{{val}}}"

    # ---- Escape underscores in headers and in string cells ----
    # Escape cell contents in non-numeric columns
    for c in m.columns:
        if c not in numeric_cols:
            m[c] = m[c].map(_escape_latex)

    # Apply central header mapping and escape for LaTeX
    m.columns = [_escape_latex(HEADER_MAP.get(c, c)) for c in m.columns]

    # ---- Save CSV (plain, un-italicized, same column order) ----
    # Read the master CSV (headers are likely mapped to HEADER_MAP); map back
    # to canonical names for processing, then re-apply HEADER_MAP for output.
    csv_plain = read_master_csv(master_csv_path)
    csv_means = csv_plain[csv_plain["_is_mean"] == True].copy()
    csv_means = csv_means.drop(columns=["log", "_is_mean"], errors="ignore")
    csv_cols = ["metric", "basis"] + [
        c for c in csv_means.columns if c not in {"metric", "basis"}
    ]
    csv_means = csv_means[csv_cols]
    # Re-apply header mapping so the output CSV has the friendly headers
    csv_means.columns = [HEADER_MAP.get(c, c) for c in csv_means.columns]
    csv_means.to_csv(means_csv_path, index=False)

    # ---- Build LaTeX table ----
    tabular_str = m.to_latex(index=False, escape=False, na_rep="")
    table_env = (
        "\\begin{table}[Ht]\n"
        "\\centering\n"
        f"\\caption{{{_escape_latex(caption)}}}\n"
        "\\tiny\n"
        f"{tabular_str}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    with open(means_tex_path, "w", encoding="utf-8") as f:
        f.write(table_env)
