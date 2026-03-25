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

from statsmodels.stats.multitest import multipletests # Import for FDR correction
from statsmodels.stats.anova import AnovaRM 

import matplotlib.pyplot as plt

from scipy.stats import permutation_test



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
                  out_dir=None,
                  infoStub=None):
    
    clustering = read_cluster(os.path.join(clustering_in_dir, 
                                           f"{n_microstates}_clustering.fif")
                                           )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(columns=['sub', 'ses', 'microstate', 'visits'])

    
    for peak_f in glob.glob(os.path.join(gfp_peaks_in_dir, '*_gfp_peaks.npz')): 
        data = np.load(peak_f)
        peaks = data['peak_data']  # Assuming you saved the peak data as 'peak_data'
        raw = mne.io.RawArray(peaks, infoStub)
        segmentation = clustering.predict(raw,
                                          picks='all')

        unique, counts = np.unique(segmentation.labels, return_counts=True)
        label_range = range(-1, n_microstates)
        label_counts = {label: 0 for label in label_range}  # Initialize all labels with 0 visits
        label_counts.update(dict(zip(unique, counts)))  # Update with actual counts

        for u in label_range:
            row = {
                'sub': os.path.basename(peak_f).split('_')[0],
                'ses': os.path.basename(peak_f).split('_')[1],
                'microstate': u,
                'visits': label_counts[u]}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True) 
       
     
                 
    df.to_csv(os.path.join(out_dir,
                           f"{n_microstates}_backfitted_peaks.csv"), 
                           index=False
                           )
    print(f"Backfitted peaks saved to {out_dir}")

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
        #c_df['sub'] = c_df['sub'].apply(lambda x: f"sub-{int(x):02d}")  # Ensure 'sub' is a zero-padded string
        # Remove microstate '-1' from all subjects
        c_df = c_df[c_df['microstate'] != -1]
        c_df['visits_z'] = c_df.groupby(['sub', 'ses'])['visits'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean()
        )
        
        u_micro = c_df['microstate'].unique()
        for u_m in u_micro:
            micro_df = c_df[c_df['microstate'] == u_m]
            merged_df = pd.merge(micro_df, part_df, on='sub', how='left')
            # Perform t-test or any other statistical test here using merged_df
            group1 = merged_df[merged_df['case_ctrl'] == 0]['visits_z']
            group2 = merged_df[merged_df['case_ctrl'] == 1]['visits_z']
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            print(f"Microstate {u_m} - t-statistic: {t_stat}, p-value: {p_value}")  
            all_test_res.append({
                'Microstate': u_m,
                'T-Statistic': np.abs(t_stat),
                'p-value': p_value,
                'N_Microstate': len(u_micro)
            })
        
    test_res_df = pd.DataFrame(all_test_res)
    
    # Group by 'N_Microstate', sort by 'T-Statistic', and relabel 'Microstate'
    test_res_df = test_res_df.sort_values(['N_Microstate', 'T-Statistic'], ascending=[True, False])
    test_res_df['unordered_Microstate'] = test_res_df['Microstate']
    test_res_df['Microstate'] = test_res_df.groupby('N_Microstate').cumcount()

    # add FDR correction for each N_Microstate group
    

    
    save_path = os.path.join(base_out_dir, 'ttest')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    test_res_df.to_csv(os.path.join(save_path, 'final_t_results.csv'), index=False)
    print(f"T-test results saved to {save_path}")    

def perm_test_microstate_visits(backfitted_peaks_in_dir=None,
                            base_out_dir=None,
                            bids_root=None,
                            n_perms=1000):
    
    part_df =pd.read_csv(bids_root + '/participants.tsv', sep='\t')
    rename_dict = {'participant_id': 'sub'}
    part_df = part_df.rename(columns=rename_dict)
    
    # Load all backfitted peaks CSV files
    backfit_f = glob.glob(os.path.join(backfitted_peaks_in_dir, 
                                       '*_backfitted_peaks.csv'))
    all_test_res = []
    for f in backfit_f:
        c_df = pd.read_csv(f)
        #c_df['sub'] = c_df['sub'].apply(lambda x: f"sub-{int(x):02d}")  # Ensure 'sub' is a zero-padded string
        # Remove microstate '-1' from all subjects
        c_df = c_df[c_df['microstate'] != -1]
        c_df['visits_z'] = c_df.groupby(['sub', 'ses'])['visits'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean()
        )
        
        u_micro = c_df['microstate'].unique()
        for u_m in u_micro:
            micro_df = c_df[c_df['microstate'] == u_m]
            merged_df = pd.merge(micro_df, part_df, on='sub', how='left')
            # Perform t-test or any other statistical test here using merged_df
            group1 = merged_df[merged_df['case_ctrl'] == 0]['visits_z']
            group2 = merged_df[merged_df['case_ctrl'] == 1]['visits_z']
            
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            
            res = permutation_test(
                (group1, group2), 
                cohens_d_statistic, 
                permutation_type='independent', 
                n_resamples=n_perms,
                alternative='two-sided'
            )
            
            
            print(f"Microstate {u_m} - d-statistic: {t_stat}, p-value: {p_value}")  
            all_test_res.append({
                'Microstate': u_m,
                'D-Statistic': np.abs(res.statistic),
                'p-value': res.pvalue,
                'N_Microstate': len(u_micro)
            })
        
    test_res_df = pd.DataFrame(all_test_res)
    
    # Group by 'N_Microstate', sort by 'D-Statistic', and relabel 'Microstate'
    test_res_df = test_res_df.sort_values(['N_Microstate', 'D-Statistic'], ascending=[True, False])
    test_res_df['unordered_Microstate'] = test_res_df['Microstate']
    test_res_df['Microstate'] = test_res_df.groupby('N_Microstate').cumcount()

    # add FDR correction for each N_Microstate group
    

    
    save_path = os.path.join(base_out_dir, 'permutation_test')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    test_res_df.to_csv(os.path.join(save_path, 'final_d_results.csv'), index=False)
    print(f"Permutation test results saved to {save_path}")    

def cohens_d_statistic(group1, group2):
    """
    Calculates Cohen's d for two independent groups.
    This serves as the 'statistic' for our permutation test.
    """
    n1, n2 = len(group1), len(group2)
    v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    
    # Standardize the difference
    # so positive values mean group2 has more visits
    return (np.mean(group2) - np.mean(group1)) / s_pooled

def rm_anova_microstate_visits(backfitted_peaks_in_dir=None,
                               base_out_dir=None,
                               bids_root=None):
    
    # Load all backfitted peaks CSV files
    backfit_f = glob.glob(os.path.join(backfitted_peaks_in_dir, 
                                       '*_backfitted_peaks.csv'))
    all_test_res = []
    for f in backfit_f:
        c_df = pd.read_csv(f)
     
        # Remove microstate '-1' from all subjects
        c_df = c_df[c_df['microstate'] != -1]
        c_df['visits_z'] = c_df.groupby(['sub', 'ses'])['visits'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x - x.mean()
        )
        
        u_micro = c_df['microstate'].unique()
        for u_m in u_micro:
            micro_df = c_df[c_df['microstate'] == u_m]
            # Perform t-test or any other statistical test here using merged_df
            micro_df['ses'] = micro_df['ses'].astype('category')
            micro_df['microstate'] = micro_df['microstate'].astype('category')
            micro_df['sub'] = micro_df['sub'].astype('category')
                # Perform repeated measures ANOVA
            aov = AnovaRM(data=micro_df,
                          depvar='visits_z',
                          subject='sub',
                          within=['ses'])
            res = aov.fit()

            # Extract F-value and p-value
            f_value = res.anova_table.iloc[0, 0] # F value for 'Session'
            p_value = res.anova_table.iloc[0, 3]
            
            print(f"Microstate {u_m} - F-statistic: {f_value}, p-value: {p_value}")  
            all_test_res.append({
                'Microstate': u_m,
                'F-Statistic': np.abs(f_value),
                'p-value': p_value,
                'N_Microstate': len(u_micro)
            })
        
    test_res_df = pd.DataFrame(all_test_res)
    
    # Group by 'N_Microstate', sort by 'F-Statistic', and relabel 'Microstate'
    test_res_df = test_res_df.sort_values(['N_Microstate', 'F-Statistic'], ascending=[True, False])
    test_res_df['unordered_Microstate'] = test_res_df['Microstate']
    test_res_df['Microstate'] = test_res_df.groupby('N_Microstate').cumcount()
     
    save_path = os.path.join(base_out_dir, 'anova')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    test_res_df.to_csv(os.path.join(save_path, 'final_anova_results.csv'), index=False)
    print(f"ANOVA results saved to {save_path}")    

 
def reorder_microstates(clustering_in_dir,
                        test_result_df,
                        base_out_dir=None
                        ):
    """
    Reorder microstates based on a specific criterion (e.g., explained variance).

    Parameters:
    - clustering_results (dict): Dictionary containing clustering results.

    Returns:
    - reordered_results (dict): Dictionary containing reordered clustering results.
    """
    n_microstates = None
    for f in glob.glob(os.path.join(clustering_in_dir, '*_clustering.fif')):
        n_microstates = os.path.basename(f).split('_')[0]  # Extract N_Microstate from filename
    
        df = test_result_df[test_result_df['N_Microstate'] == int(n_microstates)].copy()
       
        clustering = read_cluster(os.path.join(clustering_in_dir, 
                                           f"{n_microstates}_clustering.fif")
                                           )
        clustering.reorder_clusters(order=df['unordered_Microstate'].tolist())  
        
        save_path = os.path.join(base_out_dir,
                                 f"{n_microstates}_clustering.fif")
        clustering.save(save_path)
# Function 4: Backfit microstates
def backfit_data(clustering_f,
                preprocessed_data,
                factor=0,
                half_window_size=10,    
                min_segment_length=5,
                out_dir=None
                ):
    """
    Backfit microstates to the preprocessed data.

    Parameters:
    - clustering_f (str): Path to the clustering file.
    - preprocessed_data (list of np.ndarray): List of preprocessed data arrays.

    Returns:
    - backfitted_labels (list of np.ndarray): List of backfitted microstate labels for each session.
    """
    clustering = read_cluster(clustering_f)
    n_clusters = clustering.n_clusters
    measures_df = pd.DataFrame(columns=['sub',
                                        'ses',
                                        'entropy',
                                        'unlabeled'] + 
                               [f'{metric}_{i}' for i in range(n_clusters) for metric in ['mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs']])
    file_list = glob.glob(os.path.join(preprocessed_data, '*.fif')) 
    tra_m = np.zeros((len(file_list), 
                      n_clusters, 
                      n_clusters)
                      )  # Initialize transition matrix
    tra_e_m = np.zeros((len(file_list),
                        n_clusters, 
                        n_clusters)
                        )  # Initialize transition expected matrix
    for fidx, f in enumerate(file_list):
            # Backfit the group-level clustering
        print(f"Backfitting microstates for {f} using clustering from {clustering_f}")
        raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
        segmentation = clustering.predict(raw,
                                    factor=factor,
                                    half_window_size=half_window_size,
                                    min_segment_length=min_segment_length,
                                    reject_edges=True,
                                    reject_by_annotation=True,
                                    )
        subject, session , _ = raw.filenames[0].name.split('_')[:-1]
        # Compute measures
        measures = segmentation.compute_parameters()
        e_h = segmentation.entropy(ignore_repetitions=False)
        row = {
            'sub': subject,
            'ses': session,
            'entropy': e_h,
            'unlabeled': measures.get('unlabeled', 0),
        }
        for i in range(n_clusters):
            for metric in ['mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs']:
                row[f'{metric}_{i}'] = measures.get(f'{i}_{metric}', np.nan)

        measures_df = pd.concat([measures_df, pd.DataFrame([row])], ignore_index=True)

        tra_m[fidx,:,:] = segmentation.compute_transition_matrix(
            ignore_repetitions=False
            )
        tra_e_m[fidx,:,:] = segmentation.compute_expected_transition_matrix(
            ignore_repetitions=False
            )

        
        #dist = segmentation.compute_parameters(return_dist=True)
        # Save the distribution dictionary as a .npy file
        #np.save(distribution_path, dist)
    file_out = os.path.join(out_dir, 
                            'backfitted'+ f'_factor_{factor}_ws_{half_window_size}_{n_clusters}_microstates_measures.csv')
    measures_df.to_csv(file_out, index=False)
    print(f"Backfitted measures saved to {file_out}")

    tra_m_out = os.path.join(out_dir,
                             f'backfitted_factor_{factor}_ws_{half_window_size}_{n_clusters}_transition_matrices.npy')
    np.save(tra_m_out, tra_m)
    print(f"Transition matrices saved to {tra_m_out}")  

    tra_e_m_out = os.path.join(out_dir,
                               f'backfitted_factor_{factor}_ws_{half_window_size}_{n_clusters}_expected_transition_matrices.npy')
    np.save(tra_e_m_out, tra_e_m)
    print(f"Expected transition matrices saved to {tra_e_m_out}")


def corr_inv_dot(source_vector, target_map):
    """
    Compute polarity-invariant correlation using dot product between a source vector and rows of target_map.

    Parameters:
    -----------
    source_vector : np.ndarray
        A 1D array representing the source vector (shape: (x,)).
    target_map : np.ndarray
        A 2D array where each row is a target vector (shape: (y, x)).

    Returns:
    --------
    np.ndarray
        A 1D array of shape (y,) containing the maximum correlation values for the source vector
        with each target row, considering polarity invariance.
    """
    # Normalize source_vector and rows of target_map
    source_norm = source_vector / np.linalg.norm(source_vector)
    target_norm = target_map / np.linalg.norm(target_map, axis=1, keepdims=True)

    # Compute dot product for original and inverted polarity
    corr = np.dot(target_norm, source_norm)
    corr_inv = np.dot(target_norm, -source_norm)

    # Take the maximum correlation for polarity invariance
    return np.maximum(corr, corr_inv)

"""
def correlate_maps(maps_file_source,
                   source_rois,
                   in_folder,
                   base_output_dir):
    
    # read source maps and select ROIs
    source_maps = read_cluster(maps_file_source)._cluster_centers
    source_matrix = source_maps[source_rois, :]
    # read all the clustering solutions
    all_files = sorted(
            glob.glob(
                os.path.join(
                    in_folder,
                    "**_clustering.fif",
                ),
                recursive=True
            )
        )
    
    df_test = pd.read_csv(os.path.join(base_output_dir, 'ttest', 'significance_summary.csv'))
    all_corr_m = []

    for f in all_files:
        n_microstates = os.path.basename(os.path.dirname(f)).split("_")[0]
        target_map = read_cluster(f)._cluster_centers
        print(f"Loaded microstate_maps with shape: {target_map.shape}")
    
        # Perform t-test for each microstate map against the source maps

        if n_sig_states > 0:
            n_sig_x_k[int(n_microstates)] = n_sig_states
            sig_k.append(int(n_microstates))
            print(f"Number of significant states after FDR correction: {n_sig_states}")
            #Compute the correlation between source_maps and target_map
            corr_m = np.zeros((source_matrix.shape[0], target_map.shape[0]))
            for i in range(source_matrix.shape[0]):
                source_vector = source_matrix[i, :]
                corr_m[i, :] = corr_inv_dot(source_vector, target_map)
            
            all_corr_m.append(corr_m)

            
    max_n_hdm = max([m.shape[1] for m in all_corr_m])
    m = np.zeros((max_n_hdm, len(all_corr_m)))
    
    source_map = [0, 1] # Assuming you want to use the first two source maps for correlation
    for source_idx in source_map:
        for i in range(len(all_corr_m)):
                n_hdm = all_corr_m[i].shape[1]
                m[:n_hdm, i] = all_corr_m[i][source_idx, :n_hdm]
        
        heat_m_df = pd.DataFrame(m, columns=[f'K={k}' for k in sig_k])
        # Reorder the columns in ascending order according to k
        heat_m_df = heat_m_df[sorted(heat_m_df.columns, key=lambda col: int(col.split('=')[1]))]

        # Set the index names as 'HDM 0', 'HDM 1', etc.
        heat_m_df.index = [f'HDM {i}' for i in range(heat_m_df.shape[0])]

        # Create a mask for zeros to make them black and hide the numbers
        mask = (heat_m_df == 0)

        plt.figure(figsize=(12, 10))  # Adjust the figure size as needed
        hm = sns.heatmap(heat_m_df,
                    cmap='viridis',
                    annot=True,
                    annot_kws={"size": 12,
                               "weight": 'bold'
                        },
                    fmt=".2f",
                    mask=mask,
                    cbar_kws={'label': 'Spatial Correlation',
                              'ticks': []
                              }
                )

        # Overlay black boxes for zeros
        for (i, j), val in np.ndenumerate(heat_m_df.values):
            if val == 0:
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color='white'))  # Add white box to hide the number
                plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=0.5))  # Add visible borders

        # Add borders to the bottom of the heatmap
        plt.gca().add_patch(plt.Rectangle((-0.5, heat_m_df.shape[0] - 0.1), heat_m_df.shape[1], 1, 
                                          fill=False, edgecolor='black', lw=1))

        plt.title("")
        plt.xlabel("K solutions", fontsize=20)
        plt.ylabel("HDM maps", fontsize=20)
        plt.xticks(rotation=45, fontsize=18)
        plt.yticks(rotation=0, fontsize=18)
        hm.figure.axes[-1].yaxis.label.set_size(18) 
        #plt.show()
        save_path = os.path.join(base_output_dir, f"{source_idx}_heat_map_matrices")
        plt.savefig(save_path)
        plt.close()
        
        # Create a new DataFrame for n_sig_x_k values
        n_sig_x_k_df = pd.DataFrame.from_dict(n_sig_x_k, orient='index',
                                              columns=['n_sig_states'])
        n_sig_x_k_df.index.name = 'K'

        # Order the rows based on K
        n_sig_x_k_df = n_sig_x_k_df.sort_index()
        #n_sig_x_k_df['K-clusters'] = n_sig_x_k_df.index  # Convert K to string for better display
        #col_order = ['K-clusters','n_sig_states']
        #n_sig_x_k_df = n_sig_x_k_df[col_order]
        
        # Update the heatmap for n_sig_x_k values
        plt.figure(figsize=(6, 16))  # Keep the compact width
        sns.heatmap(n_sig_x_k_df,
                    cmap=sns.color_palette("coolwarm", as_cmap=True),
                    annot=True,
                    annot_kws={"size": 25, "weight": 'bold'},  # Adjust annotation font size and weight
                    fmt="d",
                    linewidths=0.5,
                    linecolor='black',
                    cbar=False,
                    cbar_kws={'label': 'Number of Significant HDMs\nafter FDR Correction', 
                              'ticks': []}
                              )  # Remove ticks
        plt.title("",
                  fontsize=18,
                  fontweight='bold',
                  pad=10
                  )
        plt.xlabel("")  # Remove column labels
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks(rotation=0, 
                   fontsize=20,
                   fontweight='bold')  # Rotate y-ticks to horizontal and adjust font size
        plt.ylabel("",
                   rotation=0,
                   fontsize=22,
                   fontweight='bold',
                   loc='top'
                   )

        # Save the heatmap
        save_path = os.path.join(base_output_dir, "n_sig_x_k_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        n_sig_x_k_df.to_csv(os.path.join(base_output_dir,
                                         "n_sig_x_k.csv"),
                            index=True
                            )
"""


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
    df.to_csv(os.path.join(base_folder, "significance_summary.csv"), index=False)
    # Subplot 1: Maximum T values and Number of Significant States (FDR correction)
    plt.figure(figsize=(10, 6))

    # Plot Maximum T values on the left y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(summary_df["N_Microstate"],
             summary_df["max_T"],
             marker="o",
             label="Max T",
             color="black"
             )
    ax1.set_ylabel("Maximum T Value",
                   fontsize=12,
                   color="black")
    ax1.set_xlabel("Number of k Microstates", fontsize=12)
    ax1.set_title("Maximum T-Values and significant states across different number of microstates", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(alpha=0.3)

    # Highlight the maximum T value(s) with a red point
    max_t_abs = summary_df['max_T'].abs().max()
    max_t_points = summary_df[summary_df['max_T'].abs() == max_t_abs]
    ax1.scatter(max_t_points['N_Microstate'],
                max_t_points['max_T'],
                color='red',
                label='Global Max T (highlighted)',
                zorder=5
                )

    # Set x-ticks to integer values from 0 to max N_Microstate + 2 and rotate them by 45 degrees
    max_n_microstate = summary_df["N_Microstate"].max()
    ax1.set_xticks(np.arange(0, max_n_microstate + 3, step=1))
    ax1.tick_params(axis='x', rotation=45)

    # Create a second y-axis for the Number of Significant States (FDR correction)
    ax2 = ax1.twinx()
    ax2.plot(significance_summary["N_Microstate"],
             significance_summary["Num_Significant_States_FDR"],
             linestyle="--",
             color="green",
             label="Num Significant States (FDR)"
             )
    ax2.set_ylabel("Number of Significant States (FDR)",
                   fontsize=12,
                   color="green")
    ax2.set_yticks(np.arange(0, 
                             significance_summary['Num_Significant_States_FDR'].max()+1,
                             step=1)
                             )
    ax2.tick_params(axis='y',
                    labelcolor="green")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

    # Save the combined plot
    output_path_combined = os.path.join(base_folder, "Max_T_and_Significant_States_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_combined, dpi=300)
    print(f"Combined plot saved to {output_path_combined}")

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


def plot_f_statistics(base_folder, alpha=0.05):
        # Load the final_anova_results.csv file
    file_path = os.path.join(base_folder, "final_anova_results.csv")
    df = pd.read_csv(file_path)

    # Compute the maximum a F value for each N_Microstate, but save the original value
    summary_df = df.loc[df.groupby("N_Microstate")['F-Statistic'].apply(lambda x: abs(x).idxmax())]
    summary_df = summary_df[['N_Microstate', 'F-Statistic']].rename(columns={'F-Statistic': 'max_F'})

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
    df.to_csv(os.path.join(base_folder, "significance_summary.csv"), index=False)
    # Subplot 1: Maximum F values and Number of Significant States (FDR correction)
    plt.figure(figsize=(10, 6))

    # Plot Maximum F values on the left y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(summary_df["N_Microstate"],
             summary_df["max_F"],
             marker="o",
             label="Max F",
             color="black"
             )
    ax1.set_ylabel("Maximum F Value",
                   fontsize=12,
                   color="black")
    ax1.set_xlabel("Number of k Microstates", fontsize=12)
    ax1.set_title("Maximum F-Values and significant states across different number of microstates", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(alpha=0.3)

    # Highlight the maximum F value(s) with a red point
    max_f_abs = summary_df['max_F'].max()
    max_f_points = summary_df[summary_df['max_F'] == max_f_abs]
    ax1.scatter(max_f_points['N_Microstate'],
                max_f_points['max_F'],
                color='red',
                label='Global Max F (highlighted)',
                zorder=5
                )

    # Set x-ticks to integer values from 0 to max N_Microstate + 2 and rotate them by 45 degrees
    max_n_microstate = summary_df["N_Microstate"].max()
    ax1.set_xticks(np.arange(0, max_n_microstate + 3, step=1))
    ax1.tick_params(axis='x', rotation=45)

    # Create a second y-axis for the Number of Significant States (FDR correction)
    ax2 = ax1.twinx()
    ax2.plot(significance_summary["N_Microstate"],
             significance_summary["Num_Significant_States_FDR"],
             linestyle="--",
             color="green",
             label="Num Significant States (FDR)"
             )
    ax2.set_ylabel("Number of Significant States (FDR)",
                   fontsize=12,
                   color="green")
    ax2.set_yticks(np.arange(0, 
                             significance_summary['Num_Significant_States_FDR'].max()+1,
                             step=1)
                             )
    ax2.tick_params(axis='y',
                    labelcolor="green")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

    # Save the combined plot
    output_path_combined = os.path.join(base_folder, "Max_F_and_Significant_States_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_combined, dpi=300)
    print(f"Combined plot saved to {output_path_combined}")

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
    output_path_significant_states_scatter = os.path.join(base_folder, "Significant_States_Scatter_Plot_FDR.png")
    plt.tight_layout()
    plt.savefig(output_path_significant_states_scatter, dpi=300)
    print(f"Significant states scatter plot saved to {output_path_significant_states_scatter}")
    #plt.show()



def plot_d_statistics(base_folder, alpha=0.05):
       # Load the final_d_results.csv file
    file_path = os.path.join(base_folder, "final_d_results.csv")
    df = pd.read_csv(file_path)

    # Compute the maximum absolute T value for each N_Microstate, but save the original value
    summary_df = df.loc[df.groupby("N_Microstate")['D-Statistic'].apply(lambda x: abs(x).idxmax())]
    summary_df = summary_df[['N_Microstate', 'D-Statistic']].rename(columns={'D-Statistic': 'max_D'})

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
    df.to_csv(os.path.join(base_folder, "significance_summary.csv"), index=False)
    # Subplot 1: Maximum T values and Number of Significant States (FDR correction)
    plt.figure(figsize=(10, 6))

    # Plot Maximum T values on the left y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(summary_df["N_Microstate"],
             summary_df["max_D"],
             marker="o",
             label="Max D",
             color="black"
             )
    ax1.set_ylabel("Maximum D Value",
                   fontsize=12,
                   color="black")
    ax1.set_xlabel("Number of k Microstates", fontsize=12)
    ax1.set_title("Maximum D-Values and significant states across different number of microstates", fontsize=14)
    ax1.tick_params(axis='y', labelcolor="black")
    ax1.grid(alpha=0.3)

    # Highlight the maximum D value(s) with a red point
    max_d_abs = summary_df['max_D'].abs().max()
    max_d_points = summary_df[summary_df['max_D'].abs() == max_d_abs]
    ax1.scatter(max_d_points['N_Microstate'],
                max_d_points['max_D'],
                color='red',
                label='Global Max D (highlighted)',
                zorder=5
                )

    # Set x-ticks to integer values from 0 to max N_Microstate + 2 and rotate them by 45 degrees
    max_n_microstate = summary_df["N_Microstate"].max()
    ax1.set_xticks(np.arange(0, max_n_microstate + 3, step=1))
    ax1.tick_params(axis='x', rotation=45)

    # Create a second y-axis for the Number of Significant States (FDR correction)
    ax2 = ax1.twinx()
    ax2.plot(significance_summary["N_Microstate"],
             significance_summary["Num_Significant_States_FDR"],
             linestyle="--",
             color="green",
             label="Num Significant States (FDR)"
             )
    ax2.set_ylabel("Number of Significant States (FDR)",
                   fontsize=12,
                   color="green")
    ax2.set_yticks(np.arange(0, 
                             significance_summary['Num_Significant_States_FDR'].max()+1,
                             step=1)
                             )
    ax2.tick_params(axis='y',
                    labelcolor="green")

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=10)

    # Save the combined plot
    output_path_combined = os.path.join(base_folder, "Max_D_and_Significant_States_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_combined, dpi=300)
    print(f"Combined plot saved to {output_path_combined}")

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
    output_path_significant_states_scatter = os.path.join(base_folder, "Significant_States_Scatter_Plot_D.png")
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
    bids_root = '/Volumes/CrucialX6/matteo/bids_als'  # Change this to your BIDS root directory
    # Define sessions
    sessions = ['01']
    n_subjects = 78# change according to your dataset
    subject_list = [f"{i:02d}" for i in range(1, n_subjects+1)]  # Subject IDs from '01'
    #subject_list = ['01', '02', '26', '27']  # Add 'sub-' prefix to each subject ID
    min_run_length = 2048  # Minimum number of time points required for a run to be included
    l_freq = 2.0
    h_freq = 30.0
    downsample = 256
    # output directory
    base_output_dir = '/Volumes/CrucialX6/matteo/bids_als/derivatives'  # Change this to your desired output directory'
    out_dir_preprocessed = os.path.join(base_output_dir, 'preprocessed_data/')
    out_dir_peaks = os.path.join(base_output_dir, 'gfp_peaks/')
    out_dir_combined_peaks = os.path.join(base_output_dir, 'combined_peaks/')
    out_dir_clustering_ordered = os.path.join(base_output_dir, 'clustering_ordered/')

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
   
    # Directory to save individual subject peaks
    if not os.path.exists(out_dir_preprocessed):
        os.makedirs(out_dir_preprocessed)
    if not os.path.exists(out_dir_peaks):
        os.makedirs(out_dir_peaks)
    if not os.path.exists(out_dir_combined_peaks):
        os.makedirs(out_dir_combined_peaks)
    if not os.path.exists(out_dir_clustering_ordered):
        os.makedirs(out_dir_clustering_ordered)

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
    
    """
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
            out_dir=os.path.join(base_output_dir, "backfitted_peaks"),
            infoStub=infoStub
        )
    
    backfitted_peaks_dir = os.path.join(base_output_dir, "backfitted_peaks")
    ttest_microstate_visits(backfitted_peaks_in_dir=backfitted_peaks_dir,
                            base_out_dir=base_output_dir,
                            bids_root=bids_root )
    
    test_file = os.path.join(base_output_dir, 'ttest', 'final_t_results.csv')
    test_result_df = pd.read_csv(test_file)
    # reorder microstates based on t-test results
    reorder_microstates(clustering_in_dir=os.path.join(base_output_dir, "clustering"),
                        test_result_df=test_result_df,
                        base_out_dir=os.path.join(base_output_dir, "clustering_ordered")
                        )
    for n in n_microstates:
         backfit_peaks(
            clustering_in_dir=os.path.join(base_output_dir, "clustering_ordered"),
            n_microstates=n,
            gfp_peaks_in_dir=out_dir_peaks,
            out_dir=os.path.join(base_output_dir, "backfitted_peaks_ordered"),
            infoStub=infoStub
        )
    

    #reorder peak backfitting based on t-test results

    plot_t_statistics(base_folder=os.path.join(base_output_dir, 
                                              "ttest"),
                                                alpha= 0.05
                                                )
    print("Microstate analysis completed successfully.")

if __name__ == "__main__":
    main_workflow()