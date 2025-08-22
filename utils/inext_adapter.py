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

# Try importing ggplot and iNEXT once
try:
    ggplot2_pkg = importr("ggplot2")
except Exception as e:
    raise RuntimeError(
        "R package 'ggplot2' is not installed in this environment."
    ) from e
try:
    iNEXT_pkg = importr("iNEXT")
except Exception as e:
    raise RuntimeError(
        "R package 'iNEXT' is not installed in this environment.\n"
        "Install inside the active conda env: conda install -c conda-forge r-inext\n"
        "Or from R: install.packages('iNEXT', repos='https://cloud.r-project.org')"
    ) from e



# --------------------------- Public API --------------------------------------

from typing import Sequence, Mapping, Dict, Any
from rpy2.robjects.vectors import IntVector, StrVector, ListVector

def to_r_abundance_list(
    abund_list: Sequence[Mapping[str, Any]],
    assemblage_prefix: str = "W",
) -> ListVector:
    """
    Convert a list of assemblages to an R list of named integer vectors.

    Accepted shapes:
      1) [{"Assemblage 1": {"A": 10, "B": 5, ...}}, {"Assemblage 2": {...}}, ...]
      2) [{"A": 10, "B": 5, ...}, {"A": 8, "C": 3, ...}, ...]  # auto-named W1, W2, ...

    Returns:
      R ListVector where each element is an IntVector with species names.
    """
    if not isinstance(abund_list, (list, tuple)):
        raise TypeError("abund_list must be a list/tuple of dicts")

    r_entries: Dict[str, IntVector] = {}

    for idx, item in enumerate(abund_list, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Item #{idx} must be a dict; got {type(item).__name__}")

        # Case 1: {"Assemblage": {species: count, ...}}
        if len(item) == 1 and isinstance(next(iter(item.values())), dict):
            (assemblage_name, species_map), = item.items()
            if not isinstance(species_map, dict):
                raise TypeError(f'Assemblage "{assemblage_name}" must map to a dict of species->count')
        else:
            # Case 2: {species: count, ...}  -> auto-name
            assemblage_name = f"{assemblage_prefix}{idx}"
            species_map = item

        species: list[str] = []
        counts: list[int] = []
        for sp, val in species_map.items():
            try:
                n = int(val)  # handles numpy ints too
            except Exception:
                raise TypeError(f'Count for species "{sp}" in "{assemblage_name}" must be int-like; got {val!r}')
            if n > 0:
                species.append(str(sp))
                counts.append(n)

        if not counts:
            raise ValueError(f'Assemblage "{assemblage_name}" has no positive counts.')

        vec = IntVector(counts)
        vec.names = StrVector(species)
        r_entries[str(assemblage_name)] = vec

    return ListVector(r_entries)



def iNEXT_estimateD(
    r_abund_list: ListVector,
    coverage_level: float = None,
    q_orders: Sequence[float] = (0, 1, 2),
    nboot: int = 200,
    conf: float = 0.95,
    datatype: str = "abundance",
) -> pd.DataFrame:
    """
    Run iNEXT::estimateD.
    If: coverage_level == None: *shared coverage* automatically selected by iNEXT
    Else: The supplied coverage_level will be used (e.g., 0.9 or 0.95). iNEXT will interpolate/extrapolate each assemblage to that coverage if possible.
    """
    if coverage_level is None:
        coverage_level = NULL
    else:
        coverage_level = float(coverage_level)
    out = iNEXT_pkg.estimateD(
        x=r_abund_list,
        q=FloatVector(list(map(float, q_orders))),
        datatype=datatype,
        base="coverage",
        level=coverage_level,
        nboot=int(nboot),
        conf=float(conf),
    )
    return _r_df_to_pandas(out)


def iNEXT_plot_coverage_curves(
    r_abund_list: ListVector,
    q_orders: Sequence[float] = (0, 1, 2),
    datatype: str = "abundance",
    nboot: int = 200,
    conf: float = 0.95,
    xlim: Tuple[float, float] = (0.0, 1.0),
    base_size: int = 18,
    facet_var: str = "Assemblage",
    color_var: str = "Assemblage",
    outfile: Optional[str] = "inext_coverage_curves.png",
    width_in: float = 9,
    height_in: float = 6,
    dpi: int = 300,
):
    """
    Make the coverage-based R/E curves like:
      ggiNEXT(out, type = 3, color.var = "Assemblage") +
      xlim(c(0,1)) + theme_bw(base_size=18) + theme(...legend bottom...)
    Returns:
      (pandas_df, r_ggplot_obj, saved_path_or_None)
    """
    # --- run iNEXT (coverage curves need SE for ribbons) ---
    out = iNEXT_pkg.iNEXT(
        r_abund_list,
        q=FloatVector([float(q) for q in q_orders]),
        datatype=datatype,
        se=True,
        nboot=int(nboot),
        conf=float(conf) if conf is not None else NULL,
    )

    # optional: the long table used for plotting (handy for debugging / python-side plotting)
    df_list = _r_list_df_to_list_pandas(out.rx2("iNextEst"))

    # --- build the ggplot (type=3 -> coverage-based curves) ---
    p = iNEXT_pkg.ggiNEXT(out, type=3, facet_var=facet_var, color_var=color_var)

    # --- apply theming and (optionally) save via R ---
    # Define a tiny R helper that styles and optionally saves
    robjects.r(f"""
    save_or_style_inext <- function(p, file, w, h, dpi, x_min, x_max, base_size) {{
      p <- p +
        xlim(c(x_min, x_max)) +
        theme_bw(base_size = base_size) +
        theme(
          legend.position = "bottom",
          legend.title = ggplot2::element_blank(),
          text = ggplot2::element_text(size = base_size),
          legend.box = "vertical"
        )
      if (!is.null(file) && nzchar(file)) {{
        ggplot2::ggsave(filename = file, plot = p, width = w, height = h, dpi = dpi,
                        units = "in", bg = "white")
      }}
      p
    }}
    """)
    save_or_style_inext = robjects.globalenv["save_or_style_inext"]

    styled_p = save_or_style_inext(
        p,
        outfile if outfile else "",
        width_in,
        height_in,
        dpi,
        float(xlim[0]),
        float(xlim[1]),
        int(base_size),
    )

    saved = outfile if outfile else None
    return df_list, styled_p, saved


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

def _r_df_to_pandas(r_df) -> pd.DataFrame:
    """Convert a single R data.frame to pandas.DataFrame."""
    with localconverter(robjects.default_converter + pandas2ri.converter):
        df = robjects.conversion.rpy2py(r_df)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected R data.frame, got {type(df)}")
    return df

def _r_list_df_to_list_pandas(r_list) -> dict[str, pd.DataFrame]:
    """
    Convert an R list of data.frames to a dict of pandas.DataFrames.
    Keeps R list names as keys.
    """
    with localconverter(robjects.default_converter + pandas2ri.converter):
        py_obj = robjects.conversion.rpy2py(r_list)

    if not isinstance(py_obj, dict):
        raise TypeError(f"Expected R list of data.frames, got {type(py_obj)}")

    out = {}
    for name, r_df in py_obj.items():
        if not isinstance(r_df, pd.DataFrame):
            raise TypeError(f"Element '{name}' is not a data.frame, got {type(r_df)}")
        out[name] = r_df
    return out