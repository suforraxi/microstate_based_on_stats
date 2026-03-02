"""
This module provides a restructured workflow for microstate analysis using the PyCrostates API.
The workflow is organized into modular functions with clear input/output specifications.
"""

import os
import pandas as pd
import numpy as np
from pycrostates.io import ChData, read_cluster
from pycrostates.cluster import ModKMeans
import mne
import glob
from scipy.stats import ttest_ind

from statsmodels.stats.multitest import multipletests  # Import for FDR correction
import matplotlib.pyplot as plt


# Function 1: Load and preprocess data
def load_and_preprocess_data(subject,
                             bids_root,
                             sessions,
                             l_freq=None,
                             h_freq=None,
                             downsample=None,
                             min_run_length=2048,
                             out_dir=None
                             ):
    """
    Load and preprocess MEG/EEG data for a given subject and session.

    Parameters:
    - subject (str): Subject ID.
    - bids_root (str): Path to BIDS root directory.
    - sessions (list of str): List of session IDs.
    - l_freq (float): Low cutoff frequency for bandpass filter.
    - h_freq (float): High cutoff frequency for bandpass filter.
    - downsample (int): Downsampling frequency in Hz.
    - min_run_length (int): Minimum number of samples required for a run to be processed.

    Returns:
    - preprocessed_data (list of np.ndarray): List of preprocessed data arrays for each session.
    """
    for session in sessions:
        ses_dir = os.path.join(bids_root, f'sub-{subject}', f'ses-{session}', 'meg')
        fif_files = sorted(glob.glob(os.path.join(ses_dir, f'sub-{subject}_ses-{session}_task-rest_acq-*_run-*_meg.fif')))

        # Group runs by acquisition
        acquisitions = {}
        for fif_file in fif_files:
            acq = fif_file.split('_acq-')[1].split('_')[0]  # Extract acquisition label
            if acq not in acquisitions:
                acquisitions[acq] = []
            acquisitions[acq].append(fif_file)

        # Process only the first acquisition
        if acquisitions:
            first_acq = list(acquisitions.keys())[0]  # Get the first acquisition
            acq_files = acquisitions[first_acq]  # Get the files for the first acquisition

            raw_list = []

        for fif_file in acq_files:
            raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
            
            # 1. Rename channels
            rename_dict = {ch: f"roi_{i}" for i, ch in enumerate(raw.info["ch_names"])}
            raw.rename_channels(rename_dict)
            
            # 2. Check run length
            if raw.n_times < min_run_length:
                print(f"Skipping run {fif_file} (length: {raw.n_times} < {min_run_length})")
                continue
            # Apply bandpass filter if specified
            if l_freq is not None or h_freq is not None:
                raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
            # 3. Apply downsampling
            if downsample is not None:
                raw.resample(downsample, npad="auto")
            
            # 4. ADD CUSTOM ANNOTATION: Label this segment before merging
            # This ensures you can find 'Run_0', 'Run_1', etc., in the final file
            run_idx = int(raw.filenames[0].name.split('_run-')[1].split('_')[0])  # Extract run index from filename
            annot = mne.Annotations(onset=[0], 
                                    duration=[raw.times[-1]], 
                                    description=[f'Run_{run_idx}'])
            raw.set_annotations(annot)
            
            raw_list.append(raw)

        if raw_list:
            # 5. Concatenate everything into one object
            # MNE will automatically add 'BAD boundary' annotations between them
            combined_raw = mne.concatenate_raws(raw_list)
            
            # 6. Compute Global Z-Score
            # We get the data matrix (n_channels, n_times)
            data = combined_raw.get_data()
            
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
            
            # Apply Z-score (using np.abs as per your original logic)
            data_z = np.abs((data - mean) / std)
            
            # Overwrite the data in the MNE object
            combined_raw._data = data_z
            
            # 7. Save as a standard MNE .fif file
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)
                save_fname = f"sub-{subject}_ses-{session}_task-combined_raw.fif"
                save_path = os.path.join(out_dir, save_fname)
                
                # We use .save() because it preserves all MNE metadata/annotations
                combined_raw.save(save_path, overwrite=True)
                print(f"Concatenated Z-scored data saved to {save_path}")

            
# Function 2: Extract GFP peaks
def extract_gfp_peaks(preprocessed_data_in_dir=None,
                      base_out= None,
                      n_peaks=None,
                      peak_distance=None,
                      smoothing_window=None):
    """
    Extract GFP peaks from preprocessed data.

    Parameters:
    - preprocessed_data_in_dir (str): Directory containing preprocessed data files.
    - peak_distance (int): Minimum distance between peaks in samples.
    - smoothing_window (int): Size of the moving average window for smoothing GFP.

    Returns:
    - peaks (list of np.ndarray): List of GFP peak indices for each session.
    """
    fif_files = sorted(glob.glob(os.path.join(preprocessed_data_in_dir, '*.fif')))

    for f in fif_files:
        raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
        data = raw.get_data()
        
        # Compute GFP
        gfp = np.std(data, axis=0)

        # Optional: Smooth GFP
        if smoothing_window is not None:
            gfp = np.convolve(gfp, np.ones(smoothing_window)/smoothing_window, mode='same')
        
        # Find peaks in GFP
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(gfp, distance=peak_distance)
        
        if n_peaks is not None and len(peaks) > n_peaks:
                gfp_amplitudes = gfp[peaks]
                top_peaks_indices = np.argsort(gfp_amplitudes)[-n_peaks:]
                peaks = peaks[top_peaks_indices]

        peak_data = data[:, peaks]  # Extract data at peak indices
        # Store peaks for this session
        print(f"Found {len(peaks)} GFP peaks in {f}")
        file_out = os.path.basename(f).replace('-combined_raw.fif', '_gfp_peaks.npz')
        save_path = os.path.join(base_out, file_out)
        np.savez(save_path, peaks=peaks, peak_data=peak_data)
        print(f"GFP peak data saved to {save_path}")
        
def combine_peaks(peaks_dir, base_out):
    """
    Combine GFP peaks from all sessions into a single array.

    Parameters:
    - peaks_dir (str): Directory containing GFP peak files.
    - base_out (str): Directory where combined peaks will be saved.
    Returns:
    - None
    """
    peak_files = sorted(glob.glob(os.path.join(peaks_dir, '*_gfp_peaks.npz')))
    all_peaks = []
    
    for pf in peak_files:
        data = np.load(pf)
        peaks = data['peak_data']  # Assuming you saved the peak data as 'peak_data'
        all_peaks.append(peaks)
    
    combined_peaks = np.concatenate(all_peaks, axis=1)  # Concatenate along the time axis
    print(f"Combined total of {len(combined_peaks)} GFP peaks from all sessions.")
    save_path = os.path.join(base_out, 'combined_peaks','combined_gfp_peaks.npz')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, peaks=combined_peaks)
    print(f"Combined GFP peaks saved to {save_path}")


# Function 3: Perform clustering
def perform_clustering(peaks_in_dir,
                       base_out_dir=None, 
                       n_microstates=None,
                       infoStub=None
                       ):
    """
    Perform clustering on GFP peaks to identify microstates.

    Parameters:
    - peaks_in_dir (str): Directory containing GFP peak files.
    - base_out_dir (str): Directory where clustering results will be saved.
    - n_microstates (int): Number of microstates to compute.
    - infoStub (str): Additional information or identifier for the clustering run.
    Returns:
    - clustering_results (dict): Dictionary containing clustering results.
    """
    # Load combined peaks
    combined_peaks_path = os.path.join(peaks_in_dir, 'combined_gfp_peaks.npz')
    data = np.load(combined_peaks_path)
    peaks = data['peaks']


    
    peaks = ChData(data=peaks, info=infoStub)  # Create ChData object with the combined peaks and info
    clustering = ModKMeans(n_clusters=n_microstates, random_state=42)
    clustering.fit(peaks, picks='all')

    save_path = os.path.join(base_out_dir,
                                     "clustering")
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{n_microstates}_clustering.fif")
    clustering.save(save_path)
    print(f"Clustering results saved to {save_path}")

def backfit_peaks(clustering_in_dir=None,
                  n_microstates=None, 
                  gfp_peaks_in_dir=None, 
                  base_out_dir=None,
                  infoStub=None):
    
    clustering = read_cluster(os.path.join(clustering_in_dir, 
                                           f"{n_microstates}_clustering.fif")
                                           )
    save_path = os.path.join(base_out_dir,
                             "backfitted_peaks"
                             )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    df = pd.DataFrame(columns=['sub', 'ses', 'microstate', 'visits'])

    
    for peak_f in glob.glob(os.path.join(gfp_peaks_in_dir, '*_gfp_peaks.npz')): 
        data = np.load(peak_f)
        peaks = data['peak_data']  # Assuming you saved the peak data as 'peak_data'
        raw = mne.io.RawArray(peaks, infoStub)
        segmentation = clustering.predict(raw, picks='all')

        unique, counts = np.unique(segmentation.labels, return_counts=True)
        for u, c in zip(unique, counts):
            row = {
                'sub': os.path.basename(peak_f).split('_')[0].replace('sub-', ''),
                'ses': os.path.basename(peak_f).split('_')[1].replace('ses-', ''),
                'microstate': u,
                'visits': c}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) 
       
     
                 
    df.to_csv(os.path.join(save_path,
                           f"{n_microstates}_backfitted_peaks.csv"), 
                           index=False
                           )
    print(f"Backfitted peaks saved to {save_path}")


def ttest_microstate_visits(backfitted_peaks_in_dir=None,
                            base_out_dir=None,
                            bids_root=None):
    
    part_df =pd.read_csv(bids_root + '/participants.tsv', sep='\t')
    rename_dict = {'participant_id': 'sub'}
    part_df = part_df.rename(columns=rename_dict)
    
    # Load all backfitted peaks CSV files
    backfit_f = glob.glob(os.path.join(backfitted_peaks_in_dir, 
                                       '*_backfitted_peaks.csv'))
    all_test_res = []
    for f in backfit_f:
        c_df = pd.read_csv(f)
        c_df['sub'] = c_df['sub'].apply(lambda x: f"sub-{int(x):02d}")  # Ensure 'sub' is a zero-padded string
        c_df['visits_z'] = c_df.groupby(['sub', 'ses'])['visits'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        # Remove microstate '-1' from all subjects
        c_df = c_df[c_df['microstate'] != -1]
        u_micro = c_df['microstate'].unique()
        for u_m in u_micro:
            micro_df = c_df[c_df['microstate'] == u_m]
            merged_df = pd.merge(micro_df, part_df, on='sub', how='left')
            # Perform t-test or any other statistical test here using merged_df
            group1 = merged_df[merged_df['case_ctrl'] == 1]['visits_z']
            group2 = merged_df[merged_df['case_ctrl'] == 0]['visits_z']
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            print(f"Microstate {u_m} - t-statistic: {t_stat}, p-value: {p_value}")  
            all_test_res.append({
                'Microstate': u_m,
                'T-Statistic': t_stat,
                'p-value': p_value,
                'N_Microstate': len(u_micro)
            })
    test_res_df = pd.DataFrame(all_test_res)
    save_path = os.path.join(base_out_dir, 'ttest')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    test_res_df.to_csv(os.path.join(save_path, 'final_t_results.csv'), index=False)
    print(f"T-test results saved to {save_path}")    

# Function 4: Backfit microstates
def backfit_microstates(clustering_results, preprocessed_data):
    """
    Backfit microstates to the preprocessed data.

    Parameters:
    - clustering_results (dict): Dictionary containing clustering results.
    - preprocessed_data (list of np.ndarray): List of preprocessed data arrays.

    Returns:
    - backfitted_labels (list of np.ndarray): List of backfitted microstate labels for each session.
    """
    pass

# Function 5: Compute metrics
def compute_microstate_metrics(backfitted_labels, preprocessed_data):
    """
    Compute metrics for the identified microstates.

    Parameters:
    - backfitted_labels (list of np.ndarray): List of backfitted microstate labels.
    - preprocessed_data (list of np.ndarray): List of preprocessed data arrays.

    Returns:
    - metrics (dict): Dictionary containing computed metrics.
    """
    pass

# Function 6: Save results
def save_results(output_dir, clustering_results, backfitted_labels, metrics):
    """
    Save the results of the microstate analysis to disk.

    Parameters:
    - output_dir (str): Directory where results will be saved.
    - clustering_results (dict): Dictionary containing clustering results.
    - backfitted_labels (list of np.ndarray): List of backfitted microstate labels.
    - metrics (dict): Dictionary containing computed metrics.

    Returns:
    - None
    """
    pass


def apply_fdr_correction(group,  alpha = 0.05):
    """
    Apply False Discovery Rate (FDR) correction to a group of p-values.

    This function adjusts the p-values in the input group using the Benjamini-Hochberg 
    procedure to control the false discovery rate. It adds two new columns to the input 
    DataFrame: 'FDR_p', which contains the FDR-corrected p-values, and 'Significant_FDR', 
    which indicates whether each p-value is significant after the correction.

    Parameters:
    -----------
    group : pandas.DataFrame
        A DataFrame containing a column named "p-value" with the p-values to be corrected.
    alpha : float, optional
        The significance level to control the false discovery rate (default is 0.05).

    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with two additional columns:
        - 'FDR_p': The FDR-corrected p-values.
        - 'Significant_FDR': A boolean column indicating whether each p-value is 
            significant after FDR correction.

    Notes:
    ------
    This function uses the `multipletests` function from the `statsmodels.stats.multitest` 
    module with the 'fdr_bh' method for FDR correction.

    Example:
    --------
    >>> import pandas as pd
    >>> from statsmodels.stats.multitest import multipletests
    >>> data = pd.DataFrame({"p-value": [0.01, 0.04, 0.03, 0.2, 0.5]})
    >>> corrected_data = apply_fdr_correction(data)
    >>> print(corrected_data)
    """
    _, fdr_corrected_pvals, _, _ = multipletests(group["p-value"], alpha=alpha, method="fdr_bh")
    group["FDR_p"] = fdr_corrected_pvals
    group["Significant_FDR"] = group["FDR_p"] < alpha
    return group

def plot_t_statistics(base_folder, alpha=0.05):
       # Load the final_t_results.csv file
    file_path = os.path.join(base_folder, "final_t_results.csv")
    df = pd.read_csv(file_path)

    # Compute the maximum absolute T value for each N_Microstate, but save the original value
    summary_df = df.loc[df.groupby("N_Microstate")['T-Statistic'].apply(lambda x: abs(x).idxmax())]
    summary_df = summary_df[['N_Microstate', 'T-Statistic']].rename(columns={'T-Statistic': 'max_T'})

    # Apply Bonferroni correction and count significant states
    df["Bonferroni_p"] = alpha / df["N_Microstate"]  # Adjust p-values using Bonferroni correction
    df["Significant_Bonferroni"] = df['p-value'] < df["Bonferroni_p"]  # Determine significance after correction

    #df = df.groupby("N_Microstate").apply(apply_fdr_correction).reset_index(drop=True)  # Reset index to avoid ambiguity
    # Exclude grouping columns explicitly during the groupby operation
    df = df.groupby("N_Microstate", group_keys=False).apply(
            lambda group: apply_fdr_correction(group).assign(N_Microstate=group.name)
    ).reset_index(drop=True)
    # Count the number of significant states for each N_Microstate
    significance_summary = df.groupby("N_Microstate")["Significant_FDR"].sum().reset_index()
    significance_summary.rename(columns={"Significant_FDR": "Num_Significant_States_FDR"}, inplace=True)

    # Subplot 1: Maximum T values
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["N_Microstate"], summary_df["max_T"], marker="o", label="Max T", color="blue")
    plt.ylabel("T Value", fontsize=12)
    plt.xlabel("Number of k Microstates", fontsize=12)
    plt.title("Maximum T-Values across different number of microstates", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # Highlight the maximum T value(s) with a red point
    max_t_abs = summary_df['max_T'].abs().max()
    max_t_points = summary_df[summary_df['max_T'].abs() == max_t_abs]
    plt.scatter(max_t_points['N_Microstate'], max_t_points['max_T'], color='red', label='Max T (highlighted)', zorder=5)

    # Set x-ticks to integer values from 0 to max N_Microstate + 2 and rotate them by 45 degrees
    max_n_microstate = summary_df["N_Microstate"].max()
    plt.xticks(np.arange(0, max_n_microstate + 3, step=1), rotation=45)

    output_path_max_t = os.path.join(base_folder, "Max_T_Values_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_max_t, dpi=300)
    print(f"Max T values plot saved to {output_path_max_t}")

    # Subplot 2: Number of significant states (FDR correction)
    plt.figure(figsize=(10, 6))
    plt.bar(significance_summary["N_Microstate"], significance_summary["Num_Significant_States_FDR"], color="purple", alpha=0.7)
    plt.ylabel("Number of Significant States (FDR)", fontsize=12)
    plt.title("Number of Significant States After FDR Correction", fontsize=14)
    plt.grid(alpha=0.3)
    output_path_significant_states = os.path.join(base_folder, "Significant_States_FDR_Plot_T.png")
    plt.tight_layout()
    plt.savefig(output_path_significant_states, dpi=300)
    print(f"Significant states plot saved to {output_path_significant_states}")

    # Subplot 4: Significant states for each N_Microstate (FDR correction)
    plt.figure(figsize=(10, 6))
    for n_microstate in df["N_Microstate"].unique():
        significant_states = df[(df["N_Microstate"] == n_microstate) & (df["Significant_FDR"])]
        plt.scatter(
            [n_microstate] * len(significant_states),
            significant_states["Microstate"],
            label=f"N_Microstate={n_microstate}" if len(significant_states) > 0 else "",
            alpha=0.7
        )
    plt.xlabel("Number of Microstates (N_Microstate)", fontsize=12)
    plt.ylabel("Microstate", fontsize=12)
    plt.title("Significant States After FDR Correction", fontsize=14)
    plt.grid(alpha=0.3)
    output_path_significant_states_scatter = os.path.join(base_folder, "Significant_States_Scatter_Plot_T.png")
    plt.tight_layout()
    plt.savefig(output_path_significant_states_scatter, dpi=300)
    print(f"Significant states scatter plot saved to {output_path_significant_states_scatter}")
    #plt.show()

# Main workflow
def main_workflow():
    """
    Main workflow for microstate analysis using PyCrostates.

    Parameters:
    - subject (str): Subject ID.
    - bids_root (str): Path to BIDS root directory.
    - sessions (list of str): List of session IDs.
    - output_dir (str): Directory where results will be saved.
    - l_freq (float): Low cutoff frequency for bandpass filter.
    - h_freq (float): High cutoff frequency for bandpass filter.
    - downsample (int): Downsampling frequency in Hz.
    - n_microstates (int): Number of microstates to compute.

    Returns:
    - None
    """

    # bids root directory
    bids_root = '/Volumes/CrucialX6/matteo/bids_ms'  # Change this to your BIDS root directory
    # Define sessions
    sessions = ['01']
    n_subjects = 50 # change according to your dataset
    subject_list = [f"{i:02d}" for i in range(1, n_subjects+1)]  # Subject IDs from '01'
    #subject_list = ['01', '02', '26', '27']  # Add 'sub-' prefix to each subject ID
    min_run_length = 2048  # Minimum number of time points required for a run to be included
    l_freq = 2.0
    h_freq = 30.0
    downsample = 256
    # output directory
    base_output_dir = '/Volumes/CrucialX6/matteo/bids_ms/derivatives'  # Change this to your desired output directory'
    out_dir_preprocessed = os.path.join(base_output_dir, 'preprocessed_data/')
    out_dir_peaks = os.path.join(base_output_dir, 'gfp_peaks/')
    out_dir_combined_peaks = os.path.join(base_output_dir, 'combined_peaks/')

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
   
    # Directory to save individual subject peaks
    if not os.path.exists(out_dir_preprocessed):
        os.makedirs(out_dir_preprocessed)
    if not os.path.exists(out_dir_peaks):
        os.makedirs(out_dir_peaks)
    if not os.path.exists(out_dir_combined_peaks):
        os.makedirs(out_dir_combined_peaks)

    """
    for subj in subject_list:
        # Step 1: Load and preprocess data
        load_and_preprocess_data(subject=subj,
                             bids_root=bids_root,
                             sessions=sessions,
                             l_freq=l_freq,
                             h_freq=h_freq,
                             downsample=downsample,
                             min_run_length=min_run_length,
                             out_dir=out_dir_preprocessed
                             )
    
    #Step 2: Extract GFP peaks
    extract_gfp_peaks(preprocessed_data_in_dir=out_dir_preprocessed,
                      base_out= out_dir_peaks,
                      n_peaks=1000,
                      peak_distance=10,
                      smoothing_window=5)

    
    combine_peaks(peaks_dir=out_dir_peaks, 
                  base_out=base_output_dir)
    
    # Step 3: Perform clustering
    file_stub='/Volumes/CrucialX6/matteo/bids_ms/derivatives/preprocessed_data/sub-01_ses-01_task-combined_raw.fif'
    raw = mne.io.read_raw_fif(file_stub, preload=True, verbose=False)
    infoStub = raw.info
    n_microstates = list(range(2, 41))  # Change this to the desired number of microstates
    for n in n_microstates:
        perform_clustering(peaks_in_dir=out_dir_combined_peaks,
                        base_out_dir=base_output_dir, 
                        n_microstates=n,
                        infoStub=infoStub
                        )

        backfit_peaks(
            clustering_in_dir=os.path.join(base_output_dir, "clustering"),
            n_microstates=n,
            gfp_peaks_in_dir=out_dir_peaks,
            base_out_dir=base_output_dir,
            infoStub=infoStub
        )
    """
    backfitted_peaks_dir = os.path.join(base_output_dir, "backfitted_peaks")
    ttest_microstate_visits(backfitted_peaks_in_dir=backfitted_peaks_dir,
                            base_out_dir=base_output_dir,
                            bids_root=bids_root )
    
    plot_t_statistics(base_folder=os.path.join(base_output_dir, 
                                               "ttest"),
                                                 alpha= 0.05
                                                 )
    # Step 4: Backfit microstates
    #backfitted_labels = backfit_microstates(clustering_results, preprocessed_data)

    # Step 5: Compute metrics
    #metrics = compute_microstate_metrics(backfitted_labels, preprocessed_data)

    # Step 6: Save results
    #save_results(output_dir, clustering_results, backfitted_labels, metrics)

    print("Microstate analysis completed successfully.")

if __name__ == "__main__":
    main_workflow()