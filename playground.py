from pathlib import Path
from utils import helpers, constants

from utils.complexity.assessor import assess_complexity_via_fixed_sized_windows
from utils.drift_io import (
    drift_info_to_dict, load_xes_log
)
from utils.population import inext_adapter
from utils.windowing.windowing import split_log_into_fixed_windows
from utils.complexity.assessor import (
    assess_complexity_via_fixed_sized_windows,
)
from utils.windowing.windowing import Window

def main():
    print('iNEXT Adapter test')
    # 1) Build a simple two-assemblage test
    abund_list = [
        {"W1": {"A": 10, "B": 5, "C": 2, "D": 2, "E": 1}},
        {"W2": {"A": 10, "B": 5, "C": 2, "D": 2, "E": 2, "G": 1}},
        {"W3": {"A": 10, "B": 5, "C": 2, "D": 1, "E": 1}}
    ]

    # 2) Convert with your helper (assuming it makes an R list of named integer vectors)
    r_abundance_list = inext_adapter.to_r_abundance_list(abund_list, assemblage_prefix="")

    # 3) Ask for a realistic coverage, e.g., 0.9 (or let iNEXT pick shared coverage by passing None)
    df = inext_adapter.iNEXT_estimateD(r_abundance_list, coverage_level=None)
    print(df)

    print('Trying to plot')
    
    df_list, ggplot_obj, path = inext_adapter.iNEXT_plot_coverage_curves(
        r_abundance_list,
        q_orders=[0],
        xlim=(0, 1.0),
        outfile="inext_coverage_curves.png",
    )
    print("Saved to:", path)

    # Print the output list
    print(df_list)

    # load an event log
    data_dict = helpers.load_data_dictionary(constants.DATA_DICTIONARY_FILE_PATH)
    
    # get BPIC12
    pm4py_log = load_xes_log(data_dict['BPIC12']['path'])
    
    adapter_names = ["vidgof_sample", "population_simple"] # , "population_inext"]

    df = assess_complexity_via_fixed_sized_windows(
        pm4py_log=pm4py_log,
        window_size=2,
        offset=2,
        dataset_key='BPIC12',
        configuration_name="cfg",
        approach_name="quickcheck",
        adapter_names=adapter_names,
        add_prefix=True,
        include_adapter_name=True,
    )

    print("=== Output DataFrame ===")
    print(df.columns)
    print(df.head())


if __name__ == "__main__":
    main()
