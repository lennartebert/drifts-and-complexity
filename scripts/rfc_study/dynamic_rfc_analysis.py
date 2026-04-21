"""Dynamic preferential-attachment (PA) analysis from attachment CSV inputs.

This implements the measurement procedure from:

  Preferential Attachment in Online Networks: Measurement and Explanations
  (Kunegis, Blattner, and Moser, 2013; Web Science / arXiv:1303.6271v1)

Attachment rows use columns: attachment_time, attachment_index, node_id.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from rfc_shared import load_attachments, parse_dataset_inputs

DEFAULT_LAMBDA_REG = 0.1


def _split_old_new(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split edges into E1 (old) and E\\E1 (new) by temporal order."""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")
    split_idx = int(np.floor(len(df) * split_ratio))
    old_df = df.iloc[:split_idx].copy()
    new_df = df.iloc[split_idx:].copy()
    return old_df, new_df


def _compute_d1_d2(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Compute d1 (degree in E1) and d2 (attachments after t1) for nodes in E1 only.

    Restricts to nodes that appear at least once before t1; nodes that first appear
    after t1 are excluded (no row with d1=0 from purely-new nodes).
    """
    d1 = old_df["node_id"].value_counts()
    d2 = new_df["node_id"].value_counts()
    all_nodes = pd.Index(d1.index)
    result = pd.DataFrame({"node_id": all_nodes})
    result["d1"] = result["node_id"].map(d1).fillna(0).astype(float)
    result["d2"] = result["node_id"].map(d2).fillna(0).astype(float)
    return result


def _fit_alpha_beta_least_squares(d_df: pd.DataFrame, lambda_reg: float) -> Tuple[float, float, float]:
    """Eq. (2) least squares: y ~ alpha + beta*log(1+d1) with y = log(lambda + d2)."""
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be > 0")
    x = np.log1p(d_df["d1"].values)
    y = np.log(lambda_reg + d_df["d2"].values)
    design = np.column_stack([np.ones_like(x), x])
    params, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    alpha = float(params[0])
    beta = float(params[1])
    residual = alpha + beta * x - y
    epsilon = float(np.exp(np.sqrt(np.mean(residual**2))))
    return alpha, beta, epsilon


def _estimate_power_law_exponent(d_df: pd.DataFrame, xmin: float = 1.0) -> float:
    """Estimate power-law exponent with robust MLE form."""
    degree = d_df["d1"].values + d_df["d2"].values
    tail = degree[degree >= xmin]
    if tail.size < 2:
        return float("nan")
    log_terms = np.log(tail / xmin)
    denom = np.sum(log_terms)
    if denom <= 0:
        return float("nan")
    return float(1.0 + tail.size / denom)


def _make_scatter_plot(d_df: pd.DataFrame, alpha: float, beta: float, lambda_reg: float, output_path: Path) -> None:
    """Log-log fit plot with grouped boxplots by regularized degree."""
    x_raw = 1.0 + d_df["d1"].values
    y_raw = lambda_reg + d_df["d2"].values
    order = np.argsort(x_raw)
    x_sorted = x_raw[order]
    y_fit_sorted = np.exp(alpha) * (x_sorted**beta)

    unique_x = np.unique(x_raw)
    grouped_x_positions: List[float] = []
    grouped_y_values: List[np.ndarray] = []
    if unique_x.size <= 200:
        for x_val in unique_x:
            vals = y_raw[x_raw == x_val]
            if vals.size > 0:
                grouped_x_positions.append(float(x_val))
                grouped_y_values.append(vals)
    else:
        quantile_edges = np.unique(np.quantile(x_raw, np.linspace(0, 1, 121)))
        for i in range(len(quantile_edges) - 1):
            lo = quantile_edges[i]
            hi = quantile_edges[i + 1]
            if i == len(quantile_edges) - 2:
                mask = (x_raw >= lo) & (x_raw <= hi)
            else:
                mask = (x_raw >= lo) & (x_raw < hi)
            vals = y_raw[mask]
            if vals.size > 0:
                x_mid = float(np.sqrt(max(lo, 1e-12) * max(hi, 1e-12)))
                grouped_x_positions.append(x_mid)
                grouped_y_values.append(vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    if grouped_y_values:
        widths = [max(pos * 0.08, 0.03) for pos in grouped_x_positions]
        ax.boxplot(
            grouped_y_values,
            positions=grouped_x_positions,
            widths=widths,
            showmeans=False,
            manage_ticks=False,
            patch_artist=True,
            boxprops={"facecolor": "#a6cee3", "alpha": 0.45, "edgecolor": "#1f78b4"},
            whiskerprops={"color": "#1f78b4", "alpha": 0.8},
            capprops={"color": "#1f78b4", "alpha": 0.8},
            medianprops={"color": "#08306b", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.2, "markeredgecolor": "#666666"},
        )
    ax.plot(
        x_sorted,
        y_fit_sorted,
        color="red",
        linewidth=2,
        label=f"fit: exp(alpha)*x^beta, exp(alpha)={np.exp(alpha):.6f}, beta={beta:.6f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Regularized degree at t1: 1 + d1(u)")
    ax.set_ylabel("Regularized new edges after t1: lambda + d2(u)")
    ax.grid(True, alpha=0.3)
    box_proxy = Patch(
        facecolor="#a6cee3",
        edgecolor="#1f78b4",
        alpha=0.45,
        label="Boxplot per degree bin (box=IQR, center line=median)",
    )
    fit_handle = ax.lines[-1]
    ax.legend(handles=[box_proxy, fit_handle], loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_figure_beta_vs_epsilon(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create figure: beta vs epsilon."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary_df["beta"], summary_df["epsilon"], alpha=0.8)
    for _, row in summary_df.iterrows():
        ax.text(row["beta"], row["epsilon"], str(row["dataset"]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Preferential attachment exponent (beta)")
    ax.set_ylabel("Root-mean-square logarithmic error (epsilon)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_figure_beta_vs_power_law(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create figure: beta vs estimated power-law exponent."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = summary_df[np.isfinite(summary_df["estimated_power_law_exponent"])]
    ax.scatter(valid["estimated_power_law_exponent"], valid["beta"], alpha=0.8)
    for _, row in valid.iterrows():
        ax.text(
            row["estimated_power_law_exponent"],
            row["beta"],
            str(row["dataset"]),
            fontsize=8,
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("Estimated power law exponent")
    ax.set_ylabel("Preferential attachment exponent (beta)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI for dynamic PA analysis."""
    parser = argparse.ArgumentParser(description="Dynamic preferential-attachment analysis from attachment CSV files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Dataset/attachments pairs: <dataset>=<attachments_csv_path>",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results/rfc_study)",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default="dynamic_pa_analysis",
        help="Subfolder name under the output root for this run (e.g. same as concept or a batch label)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.75,
        help="Temporal split ratio for t1 (default: 0.75, Kunegis et al. 2013)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=DEFAULT_LAMBDA_REG,
        help="Regularization lambda (default: 0.1, Kunegis et al. 2013)",
    )
    return parser.parse_args()


def main() -> None:
    """Run dynamic analysis for selected attachment inputs."""
    args = parse_args()
    output_root = Path(args.output_dir) if args.output_dir else Path("results") / "rfc_study"
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root = output_root / args.analysis_name
    analysis_root.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    summary_rows: List[Dict[str, float]] = []
    for dataset_name, attachments_path in input_pairs:
        if not attachments_path.exists():
            print(f"Skipping {dataset_name}: missing {attachments_path}")
            continue

        print(f"\nProcessing {dataset_name}...")
        try:
            df = load_attachments(attachments_path)
        except ValueError as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue
        old_df, new_df = _split_old_new(df, split_ratio=args.split_ratio)
        d_df = _compute_d1_d2(old_df, new_df)
        alpha, beta, epsilon = _fit_alpha_beta_least_squares(d_df, lambda_reg=args.lambda_reg)
        estimated_power_law_exponent = _estimate_power_law_exponent(d_df, xmin=1.0)

        dataset_dir = analysis_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        d_df.to_csv(dataset_dir / "d1_d2_table.csv", index=False)
        _make_scatter_plot(
            d_df,
            alpha=alpha,
            beta=beta,
            lambda_reg=args.lambda_reg,
            output_path=dataset_dir / "dynamic_fit.png",
        )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "num_edges_total": float(len(df)),
                "num_edges_old": float(len(old_df)),
                "num_edges_new": float(len(new_df)),
                "num_nodes": float(d_df["node_id"].nunique()),
                "split_ratio": float(args.split_ratio),
                "lambda_reg": float(args.lambda_reg),
                "alpha": float(alpha),
                "beta": float(beta),
                "epsilon": float(epsilon),
                "estimated_power_law_exponent": float(estimated_power_law_exponent),
            }
        )
        print(
            f"  beta={beta:.6f}, epsilon={epsilon:.6f}, "
            f"estimated_power_law_exponent={estimated_power_law_exponent:.6f}"
        )
        print(f"  Saved: {dataset_dir}")

    if not summary_rows:
        print("\nNo datasets processed.")
        raise SystemExit(1)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = analysis_root / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _make_figure_beta_vs_epsilon(summary_df, analysis_root / "figure_beta_vs_epsilon.png")
    _make_figure_beta_vs_power_law(summary_df, analysis_root / "figure_beta_vs_power_law_exponent.png")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
