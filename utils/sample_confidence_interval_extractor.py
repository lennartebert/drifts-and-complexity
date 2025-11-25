from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SampleConfidenceIntervalExtractor:
    """
    Compute across-sample confidence intervals per window size for many metrics.

    Parameters
    ----------
    conf_level : float
        Two-sided confidence level. Accepts 95 or 0.95 for 95% CI, etc.

    Notes
    -----
    - Expects a dataframe with columns:
        - 'sample_size' (window size identifier)
        - 'sample_id'   (replicate index per window size)
        - one or more *numeric* metric columns (e.g., 'Variant Entropy', ...)
    - CI per window size is computed *across samples* (not within-sample bootstrap),
      which avoids downward bias for richness-like metrics in rare-species settings.
    - Relative CI width is (CI_high - CI_low) / mean_at_window_size.
      If the mean is 0, the relative width is set to NaN for that metric/size.
    """

    conf_level: float = 0.95

    def _bounds(self) -> Tuple[float, float]:
        """Return (lower_quantile, upper_quantile) in [0,1]."""
        cl = self.conf_level
        if cl > 1:
            cl = cl / 100.0
        cl = float(np.clip(cl, 0.0, 1.0))
        alpha = 1.0 - cl
        q_low = alpha / 2.0
        q_high = 1.0 - alpha / 2.0
        return q_low, q_high

    def compute_sample_ci_long(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute across-sample CI low/high and relative CI width per window size from long format.

        Parameters
        ----------
        long_df : pd.DataFrame
            Long format DataFrame with columns:
              - 'Sample Size' (int)
              - 'Sample ID'   (int)
              - 'Metric'      (str)
              - 'Value'       (float/int)
            May have an index (will be reset if present).

        Returns
        -------
        sample_ci_df : pd.DataFrame
            Long format DataFrame with columns:
              - 'Sample Size' (int)
              - 'Metric'      (str)
              - 'Sample CI Low' (float)
              - 'Sample CI High' (float)
              - 'Sample CI Rel Width' (float)
        """
        # Reset index if it's a MultiIndex or has named levels to ensure columns are accessible
        if isinstance(long_df.index, pd.MultiIndex) or any(
            name is not None for name in long_df.index.names
        ):
            long_df = long_df.reset_index(drop=True)

        required = {"Sample Size", "Sample ID", "Metric", "Value"}
        missing = required - set(long_df.columns)
        if missing:
            raise ValueError(f"long_df is missing required columns: {missing}")

        q_low, q_high = self._bounds()

        # Group by Sample Size and Metric, then compute quantiles across samples
        g = long_df.groupby(["Sample Size", "Metric"], sort=True, as_index=True)

        # Compute quantiles and means
        ci_low = g["Value"].quantile(q_low).reset_index(name="Sample CI Low")
        ci_high = g["Value"].quantile(q_high).reset_index(name="Sample CI High")
        means = g["Value"].mean().reset_index(name="mean")

        # Merge and compute relative width
        sample_ci_df = ci_low.merge(ci_high, on=["Sample Size", "Metric"])
        sample_ci_df = sample_ci_df.merge(means, on=["Sample Size", "Metric"])

        # Relative width; guard against division by zero
        sample_ci_df["Sample CI Rel Width"] = (
            sample_ci_df["Sample CI High"] - sample_ci_df["Sample CI Low"]
        ) / sample_ci_df["mean"].replace(0.0, np.nan)
        sample_ci_df = sample_ci_df.drop(columns=["mean"])

        return sample_ci_df
