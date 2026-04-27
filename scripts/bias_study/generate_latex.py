#!/usr/bin/env python
"""Generate tab_summary_*.tex and tab_full_*.tex from bias_study master.csv (per scenario).

Reads master.csv only; does not run the bias study pipeline.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils import constants
from utils.latex_table_generation import (
    _escape_latex,
    _format_num,
    _format_pvalue,
)

from .yaml_input_handling import load_scenarios_yaml

# Metric / dimension ordering (same as rest of project)
_ALL_METRICS = constants.ALL_METRIC_NAMES
_DIM_ORDER = {d: i for i, d in enumerate(constants.DIMENSIONS_ORDER)}
_METRIC_SHORT = constants.METRIC_NAMES_TO_LATEX_MAP


def _relci_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"^RelCI \d+$", str(c))]
    return sorted(cols, key=lambda c: int(str(c).replace("RelCI ", "")))


def _relci_header_cell(col_name: str) -> str:
    m = re.search(r"\d+", str(col_name))
    return f"Rel.\\ CI {m.group()}" if m else str(col_name)


def _strip_conv_avg_suffix(text: str) -> str:
    """Drop ' (avg. …)' and similar tails from plateau / conv. column text."""
    return re.sub(r"\s*\(avg.*", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def _effect_size_label_from_rho(rho_val: Any) -> str:
    """Cohen-style correlation magnitude labels based on absolute rho."""
    try:
        rho_abs = abs(float(rho_val))
    except (TypeError, ValueError):
        return "---"
    if rho_abs >= 0.5:
        return "large"
    if rho_abs >= 0.3:
        return "medium"
    if rho_abs >= 0.1:
        return "small"
    return "---"


def _metric_order(metric: str) -> int:
    try:
        return _ALL_METRICS.index(metric)
    except ValueError:
        return 999


def _sort_mean_block(df_mean: pd.DataFrame) -> pd.DataFrame:
    df = df_mean.copy()
    df["_do"] = df["Dimension"].map(lambda d: _DIM_ORDER.get(d, 99))
    df["_mo"] = df["Metric"].map(_metric_order)
    return df.sort_values(["_do", "_mo"]).drop(columns=["_do", "_mo"])


def _sort_log_block(df_log: pd.DataFrame) -> pd.DataFrame:
    df = df_log.copy()
    df["_do"] = df["Dimension"].map(lambda d: _DIM_ORDER.get(d, 99))
    df["_mo"] = df["Metric"].map(_metric_order)
    df["_lo"] = df["Log"]
    return df.sort_values(["_do", "_mo", "_lo"]).drop(columns=["_do", "_mo", "_lo"])


def _caption_descriptor(scenario_cfg: Dict[str, Any], scenario_key: str) -> str:
    """Phrase inserted as '... for <descriptor> data' in captions."""
    raw = scenario_cfg.get("latex_data_descriptor")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    # Fallback: human-readable from scenario key
    return scenario_key.replace("_", " ")


def _summary_caption(descriptor: str) -> str:
    return f"Summarized diagnostic results for {descriptor} data"


def _full_caption(descriptor: str) -> str:
    cap = descriptor[0].upper() + descriptor[1:] if descriptor else "Diagnostic"
    return f"{cap} data results table, breakdown per log"


def _build_summary_table(
    df_mean: pd.DataFrame,
    relci_cols: Sequence[str],
    rho_col: str,
) -> Tuple[str, str]:
    """Return (colspec, body rows joined by newline, including rules and header)."""
    n_rel = len(relci_cols)
    # Columns: Dimension, Measure, rho, ES, PR, RelCIs  -> ll + c*(3+n_rel)
    colspec = "ll" + "c" * (3 + n_rel)

    rel_headers = " & ".join(_relci_header_cell(c) for c in relci_cols)
    m_var = len(relci_cols)
    row1 = (
        f" & & \\multicolumn{{2}}{{c}}{{Corr.}} & Conv. & \\multicolumn{{{m_var}}}{{c}}{{Var.}} \\\\"
    )
    row2 = f"Dimension & Measure & $\\rho$ & ES & PR & {rel_headers} \\\\"

    lines: List[str] = ["\\toprule", row1, row2, "\\midrule"]

    prev_dim: Optional[str] = None
    for _, row in df_mean.iterrows():
        dim_s = str(row.get("Dimension", ""))
        if prev_dim is not None and dim_s != prev_dim:
            lines.append("\\midrule")
        metric = row.get("Metric", "")
        dim_cell = "" if prev_dim is not None and dim_s == prev_dim else _escape_latex(dim_s)
        prev_dim = dim_s

        short = _METRIC_SHORT.get(metric)
        if short is None:
            short = _escape_latex(str(metric))

        rho = _format_num(row.get(rho_col), 4)
        es = _effect_size_label_from_rho(row.get(rho_col))
        pr_raw = row.get("Plateau Reached", row.get("Plateau Found", ""))
        if pr_raw is not None and str(pr_raw) != "":
            pr_s = _strip_conv_avg_suffix(str(pr_raw))
            pr = _escape_latex(pr_s) if pr_s else "—"
        else:
            pr = "—"

        rel_cells = " & ".join(_format_num(row.get(c), 4) for c in relci_cols)
        lines.append(
            f"{dim_cell} & {short} & {rho} & {es} & {pr} & {rel_cells} \\\\"
        )

    lines.append("\\bottomrule")
    return colspec, "\n".join(lines)


def _measure_display_latex(metric: str) -> str:
    short = _METRIC_SHORT.get(metric)
    if short is None:
        return _escape_latex(str(metric))
    return short


def _build_full_longtable(
    df_log: pd.DataFrame,
    relci_cols: Sequence[str],
    rho_col: str,
    p_col: str,
    caption_main: str,
    caption_cont: str,
    label: str,
) -> str:
    n_rel = len(relci_cols)
    total_cols = 6 + n_rel  # Dim, Meas, Log, rho, p, PR, Rel...
    last_col = total_cols
    rel_headers = " & ".join(_relci_header_cell(c) for c in relci_cols)

    col_spec_inner = "lll" + "cc" + "l" + "c" * n_rel

    header_line = f"Dimension & Measure & Log & $\\rho$ & $p$ & PR & {rel_headers} \\\\"

    parts: List[str] = [
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\setlength{\extrarowheight}{0pt}",
        r"\renewcommand{\arraystretch}{1}",
        r"\tiny",
        "",
        f"\\begin{{longtable}}{{{col_spec_inner}}}",
        f"\\caption{{{_escape_latex(caption_main)}}}",
        f"\\label{{{label}}} \\\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endfirsthead",
        "",
        f"\\caption[]{{{_escape_latex(caption_cont)}}} \\\\",
        r"\toprule",
        header_line,
        r"\midrule",
        r"\endhead",
        "",
        r"\bottomrule",
        r"\endfoot",
    ]

    prev_dim: Optional[str] = None
    prev_meas: Optional[str] = None
    for _, row in df_log.iterrows():
        dim_str = str(row.get("Dimension", ""))
        metric = str(row.get("Metric", ""))
        meas_display = _measure_display_latex(metric)

        if prev_dim is not None:
            if dim_str != prev_dim:
                parts.append(f"\\cmidrule(lr){{1-{last_col}}}")
            elif meas_display != prev_meas:
                parts.append(f"\\cmidrule(lr){{2-{last_col}}}")

        dim_cell = "" if dim_str == prev_dim else _escape_latex(dim_str)
        meas_cell = "" if meas_display == prev_meas else meas_display

        prev_dim = dim_str
        prev_meas = meas_display

        log_name = _escape_latex(str(row.get("Log", "")))
        rho = _format_num(row.get(rho_col), 4)
        pval = _format_pvalue(row.get(p_col))
        pr_raw = row.get("Plateau Reached", row.get("Plateau Found", ""))
        pr = _escape_latex(pr_raw) if pr_raw is not None and str(pr_raw) != "" else "—"
        rel_cells = " & ".join(_format_num(row.get(c), 4) for c in relci_cols)

        parts.append(
            f"{dim_cell} & {meas_cell} & {log_name} & {rho} & {pval} & {pr} & {rel_cells} \\\\"
        )

    parts.extend(["", r"\bottomrule", r"\end{longtable}"])
    return "\n".join(parts)


def _summary_below_tabular_legend(
    correlation: str,
    relci_cols: Sequence[str],
) -> str:
    nums: List[str] = []
    for c in relci_cols:
        m = re.search(r"\d+", str(c))
        if m:
            nums.append(m.group())
    relci_join = "/".join(nums)
    sizes_join = ", ".join(nums)
    rho_name = "Spearman" if correlation == "Spearman" else "Pearson"
    return f"""{{\\centering
\\vspace{{0.5em}}
$\\rho$: {rho_name} correlation \\quad
\\textit{{ES}}: Effect size~\\cite{{Cohen2009StatisticalPowerAnalysis}} \\quad
($|\\rho|\\in[0.1,0.3)$ small, $|\\rho|\\in[0.3,0.5)$ medium, $|\\rho|\\geq 0.5$ large) \\quad
PR: Plateau reached (X/Y logs) \\
Rel.\\ CI {relci_join}: 95\\% relative CI at window sizes {sizes_join}
}}

"""


def _wrap_summary_tabular(
    colspec: str,
    body: str,
    caption: str,
    label: str,
    *,
    correlation: str,
    relci_cols: Sequence[str],
) -> str:
    legend = _summary_below_tabular_legend(correlation, relci_cols)
    return f"""\\begin{{table}}[ht]
\\centering
\\setlength{{\\tabcolsep}}{{3pt}}

\\caption{{{caption}}}
\\label{{{label}}}
\\begin{{tabular}}{{{colspec}}}
{body}
\\end{{tabular}}

{legend}\\end{{table}}
"""


def generate_latex_for_scenario(
    scenario_key: str,
    scenario_cfg: Dict[str, Any],
    *,
    results_root: Path,
    correlation: str = "Spearman",
) -> Tuple[Path, Path]:
    """Write summary and full LaTeX files; return paths."""
    master_path = results_root / scenario_key / "master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master.csv: {master_path}")

    df = pd.read_csv(master_path)
    relci_cols = _relci_columns(df)
    if not relci_cols:
        raise ValueError(f"No RelCI columns in {master_path}")

    rho_col = f"{correlation} Rho"
    p_col = f"{correlation} P"
    if rho_col not in df.columns or p_col not in df.columns:
        raise ValueError(f"Expected columns {rho_col}, {p_col} in master.csv")

    df_mean = df[df["Log"] == "MEAN"].copy()
    if df_mean.empty:
        raise ValueError(f"No MEAN rows in {master_path}")
    df_mean = _sort_mean_block(df_mean)

    df_log = df[df["Log"] != "MEAN"].copy()
    if df_log.empty:
        raise ValueError(f"No per-log rows in {master_path}")
    df_log = _sort_log_block(df_log)

    descriptor = _caption_descriptor(scenario_cfg, scenario_key)
    cap_sum = _summary_caption(descriptor)
    cap_full_main = _full_caption(descriptor)
    cap_full_cont = cap_full_main + " (continued)"

    out_dir = results_root / scenario_key / "latex"
    out_dir.mkdir(parents=True, exist_ok=True)

    colspec, summary_body = _build_summary_table(df_mean, relci_cols, rho_col)
    summary_tex = _wrap_summary_tabular(
        colspec,
        summary_body,
        _escape_latex(cap_sum),
        f"tab:summarized_master_means_{scenario_key}",
        correlation=correlation,
        relci_cols=relci_cols,
    )
    p_summary = out_dir / f"tab_summary_{scenario_key}.tex"
    p_summary.write_text(summary_tex, encoding="utf-8")

    full_tex = _build_full_longtable(
        df_log,
        relci_cols,
        rho_col,
        p_col,
        cap_full_main,
        cap_full_cont,
        f"tab:summarized_master_full_{scenario_key}",
    )
    p_full = out_dir / f"tab_full_{scenario_key}.tex"
    p_full.write_text(full_tex, encoding="utf-8")

    return p_summary, p_full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tab_summary_*.tex and tab_full_*.tex from bias_study master.csv files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=constants.BIAS_STUDY_RESULTS_DIR,
        help=f"Root containing <scenario>/master.csv (default: {constants.BIAS_STUDY_RESULTS_DIR})",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Scenario names. Default: all keys in scenarios.yaml that exist under results-dir.",
    )
    parser.add_argument(
        "--correlation",
        choices=("Spearman", "Pearson"),
        default="Spearman",
        help="Which correlation columns to use in the tables.",
    )
    args = parser.parse_args()

    scenarios_yaml = load_scenarios_yaml()
    if args.scenarios is None:
        names = list(scenarios_yaml.keys())
    else:
        names = list(args.scenarios)

    for name in names:
        if name not in scenarios_yaml:
            raise SystemExit(f"Unknown scenario {name!r}. Keys: {list(scenarios_yaml.keys())}")
        cfg = scenarios_yaml[name]
        try:
            ps, pf = generate_latex_for_scenario(
                name,
                cfg,
                results_root=args.results_dir.resolve(),
                correlation=args.correlation,
            )
            print(f"Wrote {ps}")
            print(f"Wrote {pf}")
        except FileNotFoundError as e:
            print(f"[skip {name}] {e}")
        except ValueError as e:
            print(f"[skip {name}] {e}")


if __name__ == "__main__":
    main()
