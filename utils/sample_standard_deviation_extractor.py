from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SampleStandardDeviationExtractor:
    """
    Compute across-sample standard deviation per window size for many metrics.

    Parameters
    ----------
    ddof : int, optional
        Delta degrees of freedom. The divisor used in calculations is N - ddof,
        where N is the number of samples. Default is 1 (sample standard deviation).

    Notes
    -----
    - Expects a DataFrame with columns:
        - 'Sample Size' (window size identifier)
        - 'Sample ID'   (replicate index per window size)
        - 'Metric'      (str)
        - 'Value'       (float/int)
    - Standard deviation per window size is computed *across samples*
      (across Sample ID within each Sample Size and Metric).
    """

    ddof: int = 1

    def compute_sample_std_long(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute across-sample standard deviation per window size from long format.

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
        sample_std_df : pd.DataFrame
            Long format DataFrame with columns:
              - 'Sample Size' (int)
              - 'Metric'      (str)
              - 'Sample Std'  (float)
        """
        if isinstance(long_df.index, pd.MultiIndex) or any(
            name is not None for name in long_df.index.names
        ):
            long_df = long_df.reset_index(drop=True)

        required = {"Sample Size", "Sample ID", "Metric", "Value"}
        missing = required - set(long_df.columns)
        if missing:
            raise ValueError(f"long_df is missing required columns: {missing}")

        g = long_df.groupby(["Sample Size", "Metric"], sort=True, as_index=True)
        sample_std_df = g["Value"].std(ddof=self.ddof).reset_index(name="Sample Std")

        return sample_std_df
