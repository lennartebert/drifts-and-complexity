from pathlib import Path
from utils.population.estimators import EXTRACTORS
from utils.population.inext_adapter import iNEXT_plot_coverage_curves, to_r_abundance_list

def plot_coverage_curves_for_cp_windows(
    windows, out_dir: Path, q_orders=(0,), xlim=(0, 1.0)
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("activities", "dfg_edges", "trace_variants"):
        ext = EXTRACTORS[sp]
        abund_all = [{w.id: ext(w.traces)} for w in windows]
        r_list = to_r_abundance_list(abund_all)
        outfile = out_dir / f"{sp}.png"
        iNEXT_plot_coverage_curves(
            r_list,
            q_orders=list(q_orders),
            xlim=xlim,
            outfile=str(outfile),
        )
