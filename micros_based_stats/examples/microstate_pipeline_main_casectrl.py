from micros_based_stats.aggregate_peak_backfits import process_microstate_results_ttest
from micros_based_stats.plot_stat_per_k import plot_microstate_Tstatistics


if __name__ == "__main__":
    
    # Path to the base folder containing BIDS data and derivatives for microstate analysis
    base_folder = './data/'
    # Path to the participants.tsv file containing participant information
    participant_file = './data/participants.tsv'
    process_microstate_results_ttest(base_folder=base_folder, case_ctrl=participant_file)
    plot_microstate_Tstatistics(base_folder=base_folder, alpha=0.05)
