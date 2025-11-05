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

    @staticmethod
    def _metric_columns(df: pd.DataFrame) -> List[str]:
        """Infer metric columns: numeric columns minus known id columns."""
        id_cols = {"sample_size", "sample_id"}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in id_cols]

    def compute_sample_ci(
        self, measures_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute across-sample CI low/high and relative CI width per window size.

        Parameters
        ----------
        measures_df : pd.DataFrame
            Columns:
              - 'sample_size' (int)
              - 'sample_id'   (int)
              - metric columns (float/int), e.g., 'Variant Entropy', ...

        Returns
        -------
        sample_ci_low_df : pd.DataFrame
            One row per sample_size; columns: ['sample_size', <metric cols...>]
            holding the lower CI bound per metric.

        sample_ci_high_df : pd.DataFrame
            Same, for upper CI bound.

        sample_ci_rel_width_df : pd.DataFrame
            Same, for relative CI width: (high - low) / mean_at_size.
        """
        required = {"sample_size", "sample_id"}
        missing = required - set(measures_df.columns)
        if missing:
            raise ValueError(f"measures_df is missing required columns: {missing}")

        metric_cols = self._metric_columns(measures_df)
        if not metric_cols:
            raise ValueError("No numeric metric columns found besides id columns.")

        # Ensure deterministic order
        metric_cols = list(metric_cols)

        q_low, q_high = self._bounds()

        # Group by window size
        g = measures_df.groupby("sample_size", sort=True, as_index=True)

        # Aggregations across samples (one row per sample_size)
        # Quantiles across samples at each size
        ci_low = g[metric_cols].quantile(q_low)
        ci_high = g[metric_cols].quantile(q_high)

        # Mean across samples at each size (for relative width denominator)
        means = g[metric_cols].mean()

        # Relative width; guard against division by zero
        rel_width = (ci_high - ci_low) / means.replace(0.0, np.nan)

        # Reformat to have 'sample_size' as a column
        sample_ci_low_df = ci_low.reset_index()[["sample_size"] + metric_cols]
        sample_ci_high_df = ci_high.reset_index()[["sample_size"] + metric_cols]
        sample_ci_rel_width_df = rel_width.reset_index()[["sample_size"] + metric_cols]

        return sample_ci_low_df, sample_ci_high_df, sample_ci_rel_width_df
