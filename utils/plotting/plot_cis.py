from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def plot_sample_cis(
    analysis_df: pd.DataFrame,
    plot_grid: Sequence[Sequence[Sequence[str]]],
    plot_row_groups: Sequence[str],
    out_dir: Path,
) -> None:
    """Plot mean values and sample confidence intervals in a fixed grid layout.

    The function expects `analysis_df` in the format produced by
    `compute_analysis_for_metrics`, i.e. with one row per
    (Sample Size, Metric) and at least the columns:
    ``"Sample Size"``, ``"Metric"``, ``"Mean Value"``,
    ``"Sample CI Low"``, and ``"Sample CI High"``.

    The `plot_grid` argument controls which metrics are drawn in which subplot.
    It is a 2D structure where each entry is a list of metric names to plot in
    that panel (an empty list leaves the panel blank).
    `plot_row_groups` assigns a group label to every row in `plot_grid`; a group
    label is drawn once above the first row of each group.
    """
    # Global style configuration for this figure
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    # Reset index if Sample Size / Metric are stored in the index (MultiIndex or named index).
    if isinstance(analysis_df.index, pd.MultiIndex) or any(
        name is not None for name in analysis_df.index.names
    ):
        analysis_df = analysis_df.reset_index()

    required_columns = {
        "Sample Size",
        "Metric",
        "Mean Value",
        "Sample CI Low",
        "Sample CI High",
    }
    missing = required_columns - set(analysis_df.columns)
    if missing:
        raise ValueError(f"analysis_df is missing required columns: {missing}")

    n_rows = len(plot_grid)
    if n_rows == 0:
        return

    if len(plot_row_groups) != n_rows:
        raise ValueError(
            "plot_row_groups must have the same number of entries as plot_grid rows."
        )

    row_lengths = {len(row) for row in plot_grid}
    if len(row_lengths) != 1:
        raise ValueError("All rows in plot_grid must have the same length.")
    n_cols = row_lengths.pop()

    # Pre-compute per-metric data frames.
    metric_data: dict[str, pd.DataFrame] = {}
    for metric, df_metric in analysis_df.groupby("Metric"):
        metric_data[str(metric)] = df_metric.sort_values("Sample Size")

    # Create figure and axes (slightly reduced height and width per subplot).
    figsize = (3.1 * n_cols, 1.7 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Draw all panels according to the grid.
    for row_idx, row in enumerate(plot_grid):
        for col_idx, metrics_in_cell in enumerate(row):
            ax = axes[row_idx][col_idx]

            if not metrics_in_cell:
                ax.axis("off")
                continue

            for metric_name in metrics_in_cell:
                data = metric_data.get(metric_name)
                if data is None or data.empty:
                    continue

                x = data["Sample Size"].values
                y = data["Mean Value"].values
                y_low = data["Sample CI Low"].values
                y_high = data["Sample CI High"].values

                # Mean line
                ax.plot(x, y, color="C0", linestyle="-", linewidth=2)
                # Shaded CI band only (no CI boundary lines)
                ax.fill_between(x, y_low, y_high, color="C0", alpha=0.15)

            ax.grid(alpha=0.5)
            ax.set_title(", ".join(metrics_in_cell))

    # Axis labelling and tick formatting:
    # - only left column has y-label text; all subplots keep y tick marks/values
    # - disable scientific notation/offset on the y-axis
    # Formatter that shows scientific notation as Python-style 'X.YeK' directly on the ticks
    def _y_formatter(value: float, _pos: int) -> str:
        s = f"{value:.3g}"
        return s
        # if "e" not in s:
        #     return s
        # coeff, exp = s.split("e")
        # try:
        #     exp_int = int(exp)
        # except ValueError:
        #     return s
        # # Use standard Python scientific notation, e.g. '1.5e5'
        # return f"{coeff}e{exp_int}"

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if not ax.get_visible():
                continue

            if col_idx == 0:
                ax.set_ylabel("Measure Value")
            else:
                # Keep y ticks/values visible, but avoid repeating the axis label text.
                ax.set_ylabel("")

            # Apply custom formatter that writes scientific notation as 'XX*10^k'
            ax.yaxis.set_major_formatter(FuncFormatter(_y_formatter))

            ax.tick_params(axis="both")

    # Add group labels above the first row of each group.
    row_has_data = [any(cell for cell in row) for row in plot_grid]
    for row_idx, group_label in enumerate(plot_row_groups):
        if not row_has_data[row_idx]:
            continue
        if row_idx > 0 and group_label == plot_row_groups[row_idx - 1]:
            continue

        ax_first = axes[row_idx][0]
        ax_first.text(
            0.0,
            1.25,  # moved further up so labels sit clearly above subplot titles
            group_label,
            transform=ax_first.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Figure-level legend under the grid.
    # Legend: mean line + shaded confidence interval
    mean_handle = Line2D([0], [0], color="C0", linestyle="-", linewidth=2, label="Mean")
    ci_patch = plt.Rectangle((0, 0), 1, 1, facecolor="C0", alpha=0.15, edgecolor="none")
    ci_patch.set_label("95% confidence interval")
    legend_handles = [mean_handle, ci_patch]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),  # raise legend closer to grid
        ncol=len(legend_handles),
        frameon=False,
    )

    # X-axis: only the lowest non-empty subplot in each column has tick labels and an x-label.
    last_row_with_data = [-1] * n_cols
    for c in range(n_cols):
        for r in range(n_rows):
            if plot_grid[r][c]:
                last_row_with_data[c] = r

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if not ax.get_visible():
                continue
            if r != last_row_with_data[c]:
                # ax.set_xticklabels([])
                # ax.set_xlabel("")
                ax.set_xlabel("Window Size")
            else:
                ax.set_xlabel("Window Size")

    # First pass layout and margin adjustments (slightly more horizontal padding,
    # reduced vertical spacing between subplots).
    fig.tight_layout(pad=0.5, h_pad=0.15, w_pad=0.8)
    # More bottom margin so the legend sits fully below the grid; top margin for group labels.
    fig.subplots_adjust(left=0.08, bottom=0.12, top=0.96)
    # fig.subplots_adjust(bottom=0.09)   # reduce bottom whitespace

    # Align y-labels in the first column.
    fig.align_ylabels(axes[:, 0])

    # Extra vertical space between groups.
    group_labels_per_row = list(plot_row_groups)
    breaks: list[int] = []
    for r in range(len(group_labels_per_row) - 1):
        if group_labels_per_row[r] != group_labels_per_row[r + 1]:
            breaks.append(r)

    # Vertical gap between groups (in figure coordinates) to add whitespace mainly below labels.
    # Keep this small so overall vertical spacing stays tight.
    gap = 0.008
    for br in breaks:
        # start shifting from the *next* row after the break so the label row itself stays put
        for r in range(br + 1, n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                pos = ax.get_position()
                ax.set_position([pos.x0, pos.y0 - gap, pos.width, pos.height])

    fig.canvas.draw()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "sample_cis.png"
    pdf_path = out_dir / "sample_cis.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
