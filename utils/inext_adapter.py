from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Sequence, Optional
import pandas as pd

# --- rpy2 / R imports isolated here ------------------------------------------
try:
    from rpy2 import robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import ListVector, IntVector, FloatVector, NULL
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
except Exception as e:
    raise RuntimeError(
        "rpy2 is required but not available. Install it and ensure R is in this environment.\n"
        "conda install -c conda-forge r-base rpy2 r-inext"
    ) from e

# NOTE: DO NOT call pandas2ri.activate() (deprecated). Use localconverter below.

# Try importing iNEXT once
try:
    iNEXT_pkg = importr("iNEXT")
except Exception as e:
    raise RuntimeError(
        "R package 'iNEXT' is not installed in this environment.\n"
        "Install inside the active conda env: conda install -c conda-forge r-inext\n"
        "Or from R: install.packages('iNEXT', repos='https://cloud.r-project.org')"
    ) from e




# --------------------------- Public API --------------------------------------

def to_r_abundance_list(
    abund_list: Sequence[Dict[object, int]],
    assemblage_prefix: str = "W",
) -> ListVector:
    """
    Convert a list of Python Counters/dicts of species->count to an R list of
    integer vectors (one per assemblage). Names become Assemblage labels in iNEXT.
    """
    content = {}
    for idx, cnt in enumerate(abund_list, start=1):
        vec = [int(v) for v in cnt.values() if int(v) > 0]
        # iNEXT expects at least one positive count
        if not vec:
            vec = [0]
        content[f"{assemblage_prefix}{idx}"] = IntVector(vec)
    return ListVector(content)


def estimateD_common_coverage(
    r_abund_list: ListVector,
    q_orders: Sequence[float] = (0, 1, 2),
    nboot: int = 200,
    conf: float = 0.95,
    datatype: str = "abundance",
) -> pd.DataFrame:
    """
    Run iNEXT::estimateD at a *shared coverage* automatically selected by iNEXT
    (minimum coverage each assemblage can reach when extrapolated up to 2x size).
    Returns a tidy pandas DataFrame.
    """
    out = iNEXT_pkg.estimateD(
        x=r_abund_list,
        q=FloatVector(list(map(float, q_orders))),
        datatype=datatype,
        base="coverage",
        level=NULL,         # let iNEXT pick the shared coverage C*
        nboot=int(nboot),
        conf=float(conf),
    )
    return _r_df_to_pandas(out)


def estimateD_at_coverage(
    r_abund_list: ListVector,
    coverage_level: float,
    q_orders: Sequence[float] = (0, 1, 2),
    nboot: int = 200,
    conf: float = 0.95,
    datatype: str = "abundance",
) -> pd.DataFrame:
    """
    Run iNEXT::estimateD at a *fixed* sample coverage (e.g., 0.9 or 0.95).
    iNEXT will interpolate/extrapolate each assemblage to that coverage if possible.
    """
    out = iNEXT_pkg.estimateD(
        x=r_abund_list,
        q=FloatVector(list(map(float, q_orders))),
        datatype=datatype,
        base="coverage",
        level=float(coverage_level),
        nboot=int(nboot),
        conf=float(conf),
    )
    return _r_df_to_pandas(out)


def datainfo(
    r_abund_list: ListVector,
    datatype: str = "abundance",
) -> pd.DataFrame:
    """
    iNEXT::DataInfo - basic stats per assemblage (n_ref, f1, f2, coverage, etc.).
    Useful for diagnostics and transparency.
    """
    out = iNEXT_pkg.DataInfo(x=r_abund_list, datatype=datatype)
    return _r_df_to_pandas(out)


# ------------------------- Internal helpers ----------------------------------

def _r_df_to_pandas(r_obj) -> pd.DataFrame:
    """Convert an R data.frame/tibble to pandas using a local conversion context."""
    with localconverter(default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_obj)
    # Some iNEXT versions label effective size column 'm', others 't' -> normalize
    if "m" not in df.columns and "t" in df.columns:
        df = df.rename(columns={"t": "m"})
    if "Assemblage" in df.columns:
        df["Assemblage"] = df["Assemblage"].astype(str)
    return df