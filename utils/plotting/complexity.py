from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Set non-interactive backend BEFORE importing pyplot (required!)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless operation

import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode to prevent hangs

import pandas as pd

# Project constants (adjust import if your path differs)
from utils.constants import COMPLEXITY_RESULTS_DIR

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def plot_complexity_via_change_point_split(
    dataset_key: str,
    configuration_name: str,
    flat_data: Any,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    y_log: bool = False,
    fig_format: str = "png",
    headroom: float = 0.20,
    title: Optional[str] = None,
) -> None:
    """Plot complexity for *change-point windows* as HORIZONTAL segments.

    By default, generates TWO plots per measure:
    - "_over_time": Plot over time (using start_moment/end_moment)
    - "_traces": Plot over trace index (using first_index/last_index)

    - Draw one horizontal blue segment per window.
    - Write N (from 'size') above each segment.
    - Start/End vertical lines are grey; change-points are red dashed.
    - Add extra headroom at the top; optional log scale (if values > 0).
    - Saves two figures per `measure_*` column (over_time and traces).
    """
    df = _prepare_df(flat_data)
    _require_columns(df, ["size"])  # N labels

    measure_columns = _get_measure_columns(df)
    for idx, mcol in enumerate(measure_columns):
        # Plot over time - always generate this plot
        try:
            _plot_single_cp_segments(
                dataset_key=dataset_key,
                configuration_name=configuration_name,
                df=df,
                measure_column=mcol,
                drift_info_by_id=drift_info_by_id,
                y_log=y_log,
                fig_format=fig_format,
                headroom=headroom,
                title=title,  # No default title
            )
        except Exception as e:
            print(f"    [WARNING] Failed to generate _over_time plot for {mcol}: {e}")

        # Plot over traces (trace index) - always generate this plot
        try:
            _plot_single_cp_segments_traces(
                dataset_key=dataset_key,
                configuration_name=configuration_name,
                df=df,
                measure_column=mcol,
                drift_info_by_id=drift_info_by_id,
                y_log=y_log,
                fig_format=fig_format,
                headroom=headroom,
                title=title,  # No default title
            )
        except Exception as e:
            print(f"    [WARNING] Failed to generate _traces plot for {mcol}: {e}")

    print(f"    Completed plotting all measures")


def plot_complexity_via_fixed_sized_windows(
    dataset_key: str,
    configuration_name: str,
    flat_data: Any,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    window_size: Optional[int] = None,
    offset: Optional[int] = None,
    y_log: bool = False,
    fig_format: str = "png",
    headroom: float = 0.12,
    title: Optional[str] = None,
) -> None:
    """Plot complexity for *fixed-size windows* as a LINE chart.

    - Use the END of each window as the x-position (always draw a point there).
    - Connect those points with a blue line.
    - Do NOT write N labels to avoid clutter.
    - Start/End vertical lines are grey; change-points are red dashed.
    - Header defaults to: "Fixed-size windows (size=…, offset=…)" if provided.
    """
    df = _prepare_df(flat_data)

    default_title = title or (
        f"Fixed-size windows (size={window_size}, offset={offset})"
        if window_size is not None and offset is not None
        else "Fixed-size windows"
    )

    measure_columns = _get_measure_columns(df)
    for mcol in measure_columns:
        _plot_single_fixed_line(
            dataset_key=dataset_key,
            configuration_name=configuration_name,
            df=df,
            measure_column=mcol,
            drift_info_by_id=drift_info_by_id,
            y_log=y_log,
            fig_format=fig_format,
            headroom=headroom,
            title=default_title,
        )


def plot_delta_measures(
    dataset_key: str,
    configuration_name: str,
    paired_df: Any,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    point_position: str = "end_w2",  # or "mid_union"
    y_log: bool = False,
    fig_format: str = "png",
    headroom: float = 0.12,
    title: Optional[str] = None,
) -> None:
    """Plot *window comparison* results as LINE charts over time for delta_* columns.

    - Compute a single timestamp per pair for plotting (default end of window 2).
    - Connect those delta values across time with a blue line (no N labels).
    - Start/End vertical lines are grey; change-points are red dashed.
    - If data include non-positive values and y_log=True, we fall back to linear.
    """
    df = pd.DataFrame(paired_df).copy()
    for c in ["w1_start_moment", "w1_end_moment", "w2_start_moment", "w2_end_moment"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)

    # derive plotting time per pair
    if point_position == "end_w2":
        df["plot_time"] = df["w2_end_moment"]
    elif point_position == "mid_union":
        start_union = df[["w1_start_moment", "w2_start_moment"]].min(axis=1)
        end_union = df[["w1_end_moment", "w2_end_moment"]].max(axis=1)
        df["plot_time"] = start_union + (end_union - start_union) / 2
    else:
        raise ValueError("point_position must be 'end_w2' or 'mid_union'")

    # also useful for vertical start/end lines
    df["start_moment"] = df[["w1_start_moment", "w2_start_moment"]].min(axis=1)
    df["end_moment"] = df[["w1_end_moment", "w2_end_moment"]].max(axis=1)

    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    for dcol in delta_cols:
        _plot_single_delta_line(
            dataset_key=dataset_key,
            configuration_name=configuration_name,
            df=df,
            delta_column=dcol,
            drift_info_by_id=drift_info_by_id,
            y_log=y_log,
            fig_format=fig_format,
            headroom=headroom,
            title=title or "Window comparison (Δ)",
        )


# -----------------------------------------------------------------------------
# Internals – single-figure renderers & helpers
# -----------------------------------------------------------------------------


def _plot_single_cp_segments(
    dataset_key: str,
    configuration_name: str,
    df: pd.DataFrame,
    measure_column: str,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
    *,
    y_log: bool,
    fig_format: str,
    headroom: float,
    title: Optional[str],
) -> None:
    """Render one figure of horizontal segments for a change-point windowing result."""
    assert fig_format in {"png", "pdf"}
    mname = measure_column.removeprefix("measure_")

    fig, ax = plt.subplots(figsize=(12, 5))
    if title:
        fig.suptitle(title, y=0.99)
    # Pretty y-label (replace underscores with spaces)
    ax.set_ylabel(_format_measure_label(mname))

    # Format x-axis as dates to help matplotlib on Windows
    # Don't set formatter here - do it after all plotting is done

    y_series = pd.to_numeric(df[measure_column], errors="coerce").dropna()
    if y_series.empty:
        plt.close(fig)
        return

    y_min, y_max = float(y_series.min()), float(y_series.max())
    y_span = y_max - y_min

    # draw segments + N labels (from traces_in_window)
    for _, row in df.iterrows():
        val = row.get(measure_column)
        if pd.isna(val):
            continue
        ax.plot(
            [row["start_moment"], row["end_moment"]],
            [val, val],
            color="blue",
            linewidth=1.5,
        )
        # N label
        n = int(row["size"])  # guaranteed by caller
        mid = row["start_moment"] + (row["end_moment"] - row["start_moment"]) / 2
        if y_log and val and val > 0:
            y_pos = float(val) * 1.005
        else:
            y_pos = float(val) + 0.002 * (y_span if y_span != 0 else 1.0)
        ax.text(mid, y_pos, f"N={n}", fontsize=7, ha="center", va="bottom")

    _apply_y_scale_and_headroom(ax, y_series, y_log=y_log, headroom=headroom)
    _draw_start_end_and_cps(
        ax, df["start_moment"].min(), df["end_moment"].max(), drift_info_by_id
    )

    _finalize_and_save(
        ax, dataset_key, configuration_name, f"{mname}_over_time.{fig_format}"
    )


def _plot_single_cp_segments_traces(
    dataset_key: str,
    configuration_name: str,
    df: pd.DataFrame,
    measure_column: str,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
    *,
    y_log: bool,
    fig_format: str,
    headroom: float,
    title: Optional[str],
) -> None:
    """Render one figure of horizontal segments vs trace index (not time) for change-point windowing."""
    assert fig_format in {"png", "pdf"}
    mname = measure_column.removeprefix("measure_")

    # Require first_index and last_index columns for trace-based plotting
    if "first_index" not in df.columns or "last_index" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    if title:
        fig.suptitle(title, y=0.99)
    # Pretty y-label (replace underscores with spaces)
    ax.set_ylabel(_format_measure_label(mname))
    ax.set_xlabel("Trace Index")

    y_series = pd.to_numeric(df[measure_column], errors="coerce").dropna()
    if y_series.empty:
        plt.close(fig)
        return

    y_min, y_max = float(y_series.min()), float(y_series.max())
    y_span = y_max - y_min

    # draw segments + N labels using trace indices
    for _, row in df.iterrows():
        val = row.get(measure_column)
        if pd.isna(val):
            continue

        first_idx = int(row["first_index"]) if pd.notna(row.get("first_index")) else 0
        last_idx = (
            int(row["last_index"]) if pd.notna(row.get("last_index")) else first_idx
        )

        ax.plot(
            [first_idx, last_idx],
            [val, val],
            color="blue",
            linewidth=1.5,
        )
        # N label
        n = int(row["size"])  # guaranteed by caller
        mid_idx = first_idx + (last_idx - first_idx) / 2
        if y_log and val and val > 0:
            y_pos = float(val) * 1.005
        else:
            y_pos = float(val) + 0.002 * (y_span if y_span != 0 else 1.0)
        ax.text(mid_idx, y_pos, f"N={n}", fontsize=7, ha="center", va="bottom")

    _apply_y_scale_and_headroom(ax, y_series, y_log=y_log, headroom=headroom)

    # Draw change points using trace indices instead of time
    x_start = int(df["first_index"].min()) if "first_index" in df.columns else 0
    x_end = int(df["last_index"].max()) if "last_index" in df.columns else x_start

    # Convert change point moments to trace indices
    _draw_start_end_and_cps_traces(ax, x_start, x_end, drift_info_by_id, df)

    _finalize_and_save(
        ax, dataset_key, configuration_name, f"{mname}_traces.{fig_format}"
    )


def _plot_single_fixed_line(
    dataset_key: str,
    configuration_name: str,
    df: pd.DataFrame,
    measure_column: str,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
    *,
    y_log: bool,
    fig_format: str,
    headroom: float,
    title: Optional[str],
) -> None:
    """Render one figure of a LINE chart for fixed-size windows.

    Uses the window END time as the x-position and connects the values with a
    blue line; draws a point at every end.
    """
    assert fig_format in {"png", "pdf"}
    mname = measure_column.removeprefix("measure_")

    # Sort by end_moment to ensure monotonic x for line plot
    d = (
        df[["end_moment", "start_moment", measure_column]]
        .sort_values("end_moment")
        .copy()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    if title:
        fig.suptitle(title, y=0.99)

    y_series = pd.to_numeric(d[measure_column], errors="coerce").dropna()
    if y_series.empty:
        plt.close(fig)
        return

    # line + points
    ax.plot(
        d["end_moment"],
        d[measure_column],
        color="blue",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
    # Pretty y-label for fixed-size charts
    ax.set_ylabel(_format_measure_label(mname))

    _apply_y_scale_and_headroom(ax, y_series, y_log=y_log, headroom=headroom)
    _draw_start_end_and_cps(
        ax, d["start_moment"].min(), d["end_moment"].max(), drift_info_by_id
    )

    _finalize_and_save(
        ax, dataset_key, configuration_name, f"{mname}_over_time.{fig_format}"
    )


def _plot_single_delta_line(
    dataset_key: str,
    configuration_name: str,
    df: pd.DataFrame,
    delta_column: str,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
    *,
    y_log: bool,
    fig_format: str,
    headroom: float,
    title: Optional[str],
) -> None:
    """Render one LINE chart for a delta_* column from window comparison.

    If y_log=True but values include <= 0, fall back to linear automatically.
    """
    assert fig_format in {"png", "pdf"}
    dname = delta_column.removeprefix("delta_measure_")

    # Sort by plot_time to get a chronological line
    d = (
        df[["plot_time", "start_moment", "end_moment", delta_column]]
        .dropna(subset=["plot_time"])
        .sort_values("plot_time")
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    if title:
        fig.suptitle(title, y=0.99)

    # Pretty y-label for delta charts
    ax.set_ylabel(f"Δ {_format_measure_label(dname)}")

    y_series = pd.to_numeric(d[delta_column], errors="coerce").dropna()
    if y_series.empty:
        plt.close(fig)
        return

    # If non-positive values exist, disable log (not meaningful for <= 0)
    use_log = y_log and (y_series > 0).all()

    # line + points at plot_time
    ax.plot(
        d["plot_time"],
        d[delta_column],
        color="blue",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )

    _apply_y_scale_and_headroom(ax, y_series, y_log=use_log, headroom=headroom)
    _draw_start_end_and_cps(
        ax, d["start_moment"].min(), d["end_moment"].max(), drift_info_by_id
    )

    _finalize_and_save(
        ax, dataset_key, configuration_name, f"delta_{dname}_over_time.{fig_format}"
    )


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _format_measure_label(name: str) -> str:
    """Replace underscores with spaces for nicer axis labels/titles."""
    if name is None:
        return ""
    return str(name).replace("_", " ").strip()


def _prepare_df(flat_data: Any) -> pd.DataFrame:
    df = pd.DataFrame(flat_data).copy()
    # Convert to datetime, then convert to naive (timezone-unaware)
    # Matplotlib on Windows can hang with timezone-aware timestamps
    df["start_moment"] = pd.to_datetime(df["start_moment"], utc=True)  # required
    df["end_moment"] = pd.to_datetime(df["end_moment"], utc=True)  # required

    # Convert timezone-aware to naive datetime for matplotlib compatibility
    if df["start_moment"].dtype.name.startswith("datetime64"):
        if (
            hasattr(df["start_moment"].iloc[0], "tz")
            and df["start_moment"].iloc[0].tz is not None
        ):
            df["start_moment"] = df["start_moment"].dt.tz_localize(None)
    if df["end_moment"].dtype.name.startswith("datetime64"):
        if (
            hasattr(df["end_moment"].iloc[0], "tz")
            and df["end_moment"].iloc[0].tz is not None
        ):
            df["end_moment"] = df["end_moment"].dt.tz_localize(None)

    return df


def _require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _get_measure_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("measure_")]


def _apply_y_scale_and_headroom(
    ax: plt.Axes, y_series: pd.Series, *, y_log: bool, headroom: float
) -> None:
    """Apply linear/log scaling and add top headroom. Linear bottom is clamped at 0.

    For log, choose a bottom slightly below the smallest positive value.
    If no positive values exist, revert to linear.
    """
    y_series = pd.to_numeric(y_series, errors="coerce").dropna()
    if y_series.empty:
        return

    y_min, y_max = float(y_series.min()), float(y_series.max())
    y_span = y_max - y_min

    if not y_log:
        top_extra = headroom * (y_span if y_span > 0 else max(abs(y_max), 1.0))
        ax.set_ylim(bottom=0, top=y_max + top_extra)
        return

    # log scaling
    positives = y_series[y_series > 0]
    if positives.empty:
        # cannot use log – fall back to linear
        top_extra = headroom * (y_span if y_span > 0 else max(abs(y_max), 1.0))
        ax.set_ylim(bottom=0, top=y_max + top_extra)
        return

    min_pos = float(positives.min())
    bottom = min_pos / 1.5
    top = y_max * (1.0 + headroom if y_max > 0 else 1.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=bottom, top=top)


def _draw_start_end_and_cps(
    ax: plt.Axes,
    x_start: pd.Timestamp,
    x_end: pd.Timestamp,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    """Draw start/end grey dashed lines and optional CPs as red dashed lines."""
    label_map = {
        "sudden": "Sudden",
        "gradual_start": "Gradual start",
        "gradual_end": "Gradual end",
        "start": "Start",
        "end": "End",
    }

    # Convert to naive timestamps if timezone-aware (matplotlib can have issues with tz-aware on Windows)
    if hasattr(x_start, "tz") and x_start.tz is not None:
        x_start = x_start.tz_localize(None)
    if hasattr(x_end, "tz") and x_end.tz is not None:
        x_end = x_end.tz_localize(None)

    # Set x-axis limits first to help matplotlib with date formatting
    try:
        ax.set_xlim(left=x_start, right=x_end)
    except Exception as e:
        print(f"        [WARNING] Error setting xlim: {e}")

    # Use plot() instead of axvline() as workaround for potential matplotlib date axis issue
    ylim = ax.get_ylim()

    try:
        ax.plot([x_start, x_start], ylim, color="grey", linestyle="--", linewidth=1)
    except Exception as e:
        print(f"        [ERROR] Error drawing start line: {e}")
        import traceback

        traceback.print_exc()
        raise
    try:
        ax.plot([x_end, x_end], ylim, color="grey", linestyle="--", linewidth=1)
    except Exception as e:
        print(f"        [ERROR] Error drawing end line: {e}")
        import traceback

        traceback.print_exc()
        raise

    ylim_top = ax.get_ylim()[1]
    ylim_bottom = ax.get_ylim()[0]
    # Add extra space above ylim_top for rotated labels (rotation=45 needs more vertical space)
    label_space = (ylim_top - ylim_bottom) * 0.15  # 15% extra space for labels
    ax.text(
        x_start,
        ylim_top,
        label_map["start"],
        fontsize=8,
        ha="left",
        va="bottom",
        rotation=45,
    )
    ax.text(
        x_end,
        ylim_top,
        label_map["end"],
        fontsize=8,
        ha="left",
        va="bottom",
        rotation=45,
    )

    # change-points (red)
    if drift_info_by_id:
        for cid, info in drift_info_by_id.items():
            if cid == "na":
                continue
            cp_x = pd.to_datetime(info["calc_change_moment"])
            cp_lab = label_map.get(
                info.get("calc_change_type"), info.get("calc_change_type", "cp")
            )
            ax.axvline(x=cp_x, color="red", linestyle="--", alpha=0.5)
            # Place change point label higher to account for rotation
            label_y = ylim_top + label_space
            ax.text(
                cp_x, label_y, cp_lab, fontsize=8, ha="left", va="bottom", rotation=45
            )

    # common axes decorations
    ax.set_xlabel("Time")

    # Don't set explicit date formatter - let matplotlib auto-format to avoid hangs on Windows
    # Explicit formatters can cause hangs with date axes on Windows
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
    ax.grid(True)


def _draw_start_end_and_cps_traces(
    ax: plt.Axes,
    x_start: int,
    x_end: int,
    drift_info_by_id: Optional[Dict[str, Dict[str, Any]]],
    df: pd.DataFrame,
) -> None:
    """Draw start/end grey dashed lines and optional CPs as red dashed lines using trace indices."""
    label_map = {
        "sudden": "Sudden",
        "gradual_start": "Gradual start",
        "gradual_end": "Gradual end",
        "start": "Start",
        "end": "End",
    }

    # Set x-axis limits
    try:
        ax.set_xlim(left=x_start, right=x_end)
    except Exception as e:
        print(f"        [WARNING] Error setting xlim: {e}")

    # Use plot() for vertical lines
    ylim = ax.get_ylim()

    try:
        ax.plot([x_start, x_start], ylim, color="grey", linestyle="--", linewidth=1)
    except Exception as e:
        print(f"        [ERROR] Error drawing start line: {e}")
        import traceback

        traceback.print_exc()
        raise
    try:
        ax.plot([x_end, x_end], ylim, color="grey", linestyle="--", linewidth=1)
    except Exception as e:
        print(f"        [ERROR] Error drawing end line: {e}")
        import traceback

        traceback.print_exc()
        raise

    ylim_top = ax.get_ylim()[1]
    ylim_bottom = ax.get_ylim()[0]
    # Add extra space above ylim_top for rotated labels
    label_space = (ylim_top - ylim_bottom) * 0.15  # 15% extra space for labels
    ax.text(
        x_start,
        ylim_top,
        label_map["start"],
        fontsize=8,
        ha="left",
        va="bottom",
        rotation=45,
    )
    ax.text(
        x_end,
        ylim_top,
        label_map["end"],
        fontsize=8,
        ha="left",
        va="bottom",
        rotation=45,
    )

    # change-points (red) - convert change point moments to trace indices
    if drift_info_by_id:
        for cid, info in drift_info_by_id.items():
            if cid == "na":
                continue
            cp_moment = pd.to_datetime(info["calc_change_moment"])
            # Convert to naive datetime if timezone-aware (to match DataFrame timestamps)
            if hasattr(cp_moment, "tz") and cp_moment.tz is not None:
                cp_moment = cp_moment.tz_localize(None)
            cp_lab = label_map.get(
                info.get("calc_change_type"), info.get("calc_change_type", "cp")
            )

            # Convert change point moment to trace index
            # Find the window that contains this change point moment
            cp_trace_idx = None
            if "start_moment" in df.columns and "end_moment" in df.columns:
                # Find window where start_moment <= cp_moment <= end_moment
                for _, row in df.iterrows():
                    start_m = pd.to_datetime(row["start_moment"])
                    end_m = pd.to_datetime(row["end_moment"])
                    # Ensure both are naive for comparison
                    if hasattr(start_m, "tz") and start_m.tz is not None:
                        start_m = start_m.tz_localize(None)
                    if hasattr(end_m, "tz") and end_m.tz is not None:
                        end_m = end_m.tz_localize(None)
                    if start_m <= cp_moment <= end_m:
                        # Use the first_index of this window as the trace index
                        if "first_index" in row and pd.notna(row["first_index"]):
                            cp_trace_idx = int(row["first_index"])
                            break

                # If not found, try to interpolate based on time position
                if cp_trace_idx is None:
                    # Find the closest window boundaries
                    if len(df) > 0:
                        df_sorted = df.sort_values("start_moment")
                        first_start = pd.to_datetime(df_sorted["start_moment"].iloc[0])
                        last_end = pd.to_datetime(df_sorted["end_moment"].iloc[-1])

                        if cp_moment < first_start:
                            cp_trace_idx = (
                                int(df_sorted["first_index"].iloc[0])
                                if "first_index" in df_sorted.columns
                                else x_start
                            )
                        elif cp_moment > last_end:
                            cp_trace_idx = (
                                int(df_sorted["last_index"].iloc[-1])
                                if "last_index" in df_sorted.columns
                                else x_end
                            )
                        else:
                            # Interpolate between windows
                            for i in range(len(df_sorted) - 1):
                                w1_end = pd.to_datetime(df_sorted["end_moment"].iloc[i])
                                w2_start = pd.to_datetime(
                                    df_sorted["start_moment"].iloc[i + 1]
                                )
                                if w1_end <= cp_moment <= w2_start:
                                    # Interpolate between the two windows
                                    w1_last_idx = (
                                        int(df_sorted["last_index"].iloc[i])
                                        if "last_index" in df_sorted.columns
                                        else x_start
                                    )
                                    w2_first_idx = (
                                        int(df_sorted["first_index"].iloc[i + 1])
                                        if "first_index" in df_sorted.columns
                                        else x_end
                                    )
                                    # Use midpoint or weighted average
                                    cp_trace_idx = (w1_last_idx + w2_first_idx) // 2
                                    break

            if cp_trace_idx is not None:
                ax.plot(
                    [cp_trace_idx, cp_trace_idx],
                    ylim,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )
                # Place change point label higher to account for rotation
                label_y = ylim_top + label_space
                ax.text(
                    cp_trace_idx,
                    label_y,
                    cp_lab,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    rotation=45,
                )

    # common axes decorations
    ax.set_xlabel("Trace Index")  # Already set by caller, but ensure it's set

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
    ax.grid(True)


def _finalize_and_save(
    ax: plt.Axes, dataset_key: str, configuration_name: str, filename: str
) -> None:
    mname = ax.get_ylabel()  # may be empty if not set yet

    # If ylabel not set by caller, try to set from filename
    if not mname:
        # very light inference; main callers set ylabel explicitly though
        try:
            stem = Path(filename).stem
            ax.set_ylabel(stem.replace("_over_time", "").replace("delta_", "Δ "))
        except Exception:
            pass

    fig = ax.figure
    # Skip tight_layout on Windows as it can hang with date axes
    # Use adjust_subplots instead which is more reliable
    try:
        import platform

        if platform.system() == "Windows":
            # Increase top margin to accommodate rotated change point labels
            fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.15)
        else:
            # Increase top margin for change point labels
            fig.tight_layout(rect=[0, 0, 1, 0.88])
    except Exception as e:
        # Continue anyway - layout adjustment is not critical
        pass

    out_dir = COMPLEXITY_RESULTS_DIR / dataset_key / configuration_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract base filename without extension to save both PNG and PDF
    filename_path = Path(filename)
    base_filename = filename_path.stem  # filename without extension

    # Save both PNG and PDF formats
    for fmt in ["png", "pdf"]:
        out_path = out_dir / f"{base_filename}.{fmt}"
        try:
            import platform
            import threading

            # On Windows, use threading with timeout to detect hangs
            if platform.system() == "Windows":
                save_dpi = 150  # Lower DPI for faster saving on Windows

                # Use threading to save with timeout detection
                save_success = threading.Event()
                save_error = [None]

                def save_worker():
                    try:
                        # Ensure canvas is ready - don't call draw() as it can hang
                        # Just save directly with minimal options
                        save_kwargs = {
                            "format": fmt,
                            "facecolor": "white",
                            "edgecolor": "none",
                            "metadata": None,
                        }
                        if fmt == "png":
                            save_kwargs["dpi"] = save_dpi
                            save_kwargs["pil_kwargs"] = {
                                "optimize": False
                            }  # Disable PIL optimization for PNG
                        else:  # PDF
                            save_kwargs["bbox_inches"] = "tight"
                        fig.savefig(out_path, **save_kwargs)
                        save_success.set()
                    except Exception as e:
                        save_error[0] = e
                        save_success.set()

                # Start save in separate thread
                save_thread = threading.Thread(target=save_worker, daemon=True)
                save_thread.start()

                # Wait up to 30 seconds for save to complete
                timeout = 30
                if save_success.wait(timeout=timeout):
                    if save_error[0]:
                        raise save_error[0]
                else:
                    raise TimeoutError(
                        f"savefig() timed out after {timeout} seconds on Windows"
                    )
            else:
                save_dpi = 600 if fmt == "png" else None
                fig.savefig(
                    out_path,
                    dpi=save_dpi,
                    format=fmt,
                    facecolor="white",
                    bbox_inches="tight",
                )
        except Exception as e:
            print(f"        [ERROR] Error saving figure as {fmt}: {e}")
            import traceback

            traceback.print_exc()
            # Continue to try saving the other format even if one fails
            continue

    plt.close(fig)
