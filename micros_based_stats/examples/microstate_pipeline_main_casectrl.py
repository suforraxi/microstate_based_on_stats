from micros_based_stats.aggregate_peak_backfits import process_microstate_results_ttest
from micros_based_stats.plot_stat_per_k import plot_microstate_Tstatistics


if __name__ == "__main__":
    base_folder = '/Volumes/CrucialX6/matteo/bids_ms/derivatives/microstate_analysis/3000_peaks'
    participant_file = '/Volumes/CrucialX6/matteo/bids_ms/participants.tsv'
    process_microstate_results_ttest(base_folder=base_folder, case_ctrl=participant_file)
    plot_microstate_Tstatistics(base_folder=base_folder, alpha=0.05)
