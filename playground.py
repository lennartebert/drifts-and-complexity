from pathlib import Path
from utils import helpers, inext_adapter, constants

from utils.drift_io import (
    drift_info_to_dict, load_xes_log
)
from utils.population_estimates import estimate_populations, plot_coverage_curves
from utils.windowing.windowing import split_log_into_fixed_windows

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

    # split log into windows
    windows = split_log_into_fixed_windows(pm4py_log, 1000, 1000)

    # only keep first 5 windows for now
    windows = windows[0:5]

    # get measure:
    get_measures_for_windows


    # calculate the population estimates 
    population_estimates_full_coverage = estimate_populations(windows)
    print(population_estimates_full_coverage) # -> this is a list of dataframes, one per window

    # try to do some plotting
    # get the abundance list for all species
    plot_coverage_curves(windows, out_dir='results/') # creates one plot per species

if __name__ == "__main__":
    main()
