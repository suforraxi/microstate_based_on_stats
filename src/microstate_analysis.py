import os
import glob
import mne
import numpy as np
import neurokit2 as nk
from scipy.signal import find_peaks
from neurokit2.stats.cluster_quality import _cluster_quality_gev
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from plot_utils import plot_gfp

def extract_peaks(
    subject,
    bids_root,
    sessions,
    save_dir,
    l_freq=2,
    h_freq=30,
    n_peaks=None,
    min_run_length=2048,
    peak_distance=10,
    smoothing_window=5
):
    """
    Extract GFP peaks for a subject across sessions and save the peaks matrix and session limits.

    Parameters
    ----------
    subject : str
        Subject ID.
    bids_root : str
        Path to BIDS root.
    sessions : list of str
        List of session IDs.
    save_dir : str
        Directory to save results.
    l_freq : float or None
        Low cutoff frequency for bandpass filter.
    h_freq : float or None
        High cutoff frequency for bandpass filter.
    n_peaks : int or None
        Total number of GFP peaks to select for each session of each subject. If None, consider all peaks.
    min_run_length : int
        Minimum number of time points required for a run to be included.
    peak_distance : int
        Minimum distance between peaks in samples.
    smoothing_window : int or None
        Size of the moving average window for smoothing GFP. If None, no smoothing is applied.

    Returns
    -------
    None
    """
    print(f"Extracting peaks for subject {subject}...")

    # Create a directory for the subject's results
    subject_save_dir = os.path.join(save_dir, f"sub-{subject}")
    os.makedirs(subject_save_dir, exist_ok=True)
    gfp_dir = os.path.join(save_dir, "gfp_plots")
    os.makedirs(gfp_dir, exist_ok=True)

    all_selected_data = []
    session_limits = {}  # To store session start and end indices
    channel_names = None
    sampling_rate = None
    current_index = 0

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

        session_peak_data = []
        session_peak_count = 0  # Initialize peak count for the session

        # Process only the first acquisition
        if acquisitions:
            first_acq = list(acquisitions.keys())[0]  # Get the first acquisition
            acq_files = acquisitions[first_acq]  # Get the files for the first acquisition

            acq_data = []
        #for acq, acq_files in acquisitions.items():
        #    acq_data = []

            for fif_file in acq_files:
                raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)

                # Check if the run length meets the minimum requirement
                if raw.n_times < min_run_length:
                    print(f"Skipping run {fif_file} (length: {raw.n_times} < {min_run_length})")
                    continue

                # Apply bandpass filter if specified
                if l_freq is not None or h_freq is not None:
                    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
                data = raw.get_data()  # shape: (n_channels, n_times)
                if channel_names is None:
                    channel_names = raw.ch_names
                if sampling_rate is None:
                    sampling_rate = raw.info['sfreq']

                # Append data for the current run
                acq_data.append(data)

            # Concatenate data across all runs in the same acquisition
            if len(acq_data) == 0:
                continue  # Skip if no valid runs in this acquisition
            acq_data_concat = np.concatenate(acq_data, axis=1)  # shape: (n_channels, total_n_times_acq)

            # Compute z-score across the concatenated acquisition data
            mean = np.mean(acq_data_concat, axis=1, keepdims=True)
            std = np.std(acq_data_concat, axis=1, keepdims=True)
            acq_data_z = (acq_data_concat - mean) / std

            # Compute absolute value of z-scored data
            acq_data_abs = np.abs(acq_data_z)

            # Compute GFP for the current acquisition
            gfp = np.std(acq_data_z, axis=0)

            # Apply smoothing to the GFP signal if specified
            if smoothing_window is not None and smoothing_window > 1:
                gfp = np.convolve(gfp, np.ones(smoothing_window) / smoothing_window, mode='same')

            # Find GFP peaks for the entire acquisition
            peaks, _ = find_peaks(gfp, distance=peak_distance)

            # Plot and save the GFP with peaks
            plot_gfp(gfp, peaks, subject, session, gfp_dir)

            print(f"Acquisition {first_acq}: found {len(peaks)} peaks before selection.")
            # Select the top `n_peaks` GFP peaks if specified
            if n_peaks is not None and len(peaks) > n_peaks:
                gfp_amplitudes = gfp[peaks]
                top_peaks_indices = np.argsort(gfp_amplitudes)[-n_peaks:]
                peaks = peaks[top_peaks_indices]

            # Update session peak count
            session_peak_count += len(peaks)

            # Extract data corresponding to the peaks
            peak_data = acq_data_abs[:, peaks]
            #peak_data = acq_data_z[:, peaks]
            session_peak_data.append(peak_data)

        # Concatenate all peaks and data for the session
        if len(session_peak_data) == 0:
            continue  # Skip if no valid data for this session
        session_peak_data = np.concatenate(session_peak_data, axis=1)  # shape: (n_channels, total_peaks_session)

        # Append selected peaks data for the session to the global list
        all_selected_data.append(session_peak_data)

        # Track session limits
        start_index = current_index
        end_index = current_index + session_peak_data.shape[1] - 1
        session_limits[session] = (start_index, end_index)
        current_index = end_index + 1

        print(f"Session {session}: extracted {session_peak_count} peaks.")

    if len(all_selected_data) == 0:
        print(f"No valid data found for subject {subject}. Skipping...")
        return

    # Concatenate all selected peak data
    all_selected_data = np.concatenate(all_selected_data, axis=1)  # shape: (n_channels, total_peaks)

    # Save the peaks matrix and session limits in a single .npz file
    save_path = os.path.join(subject_save_dir, f"peaks_and_limits_sub-{subject}.npz")
    np.savez(save_path, Peaks=all_selected_data, SessionLimits=session_limits)
    print(f"Peaks and session limits saved for subject {subject} at {save_path}")


def concatenate_peaks(
    subject_list,
    input_dir,
    save_dir,
    sessions=['01', '02', '03']
):
    """
    Concatenate peak matrices from all subjects, reorder by session, and save the concatenated matrix.

    Parameters
    ----------
    subject_list : list of str
        List of subject IDs.
    input_dir : str
        Directory where subject peak matrices are stored.
    save_dir : str
        Directory where the concatenated matrix will be saved.
    sessions : list of str
        List of session IDs.

    Returns
    -------
    str
        Path to the saved concatenated matrix file.
    """
    all_selected_data = []  # To store concatenated data
    session_limits = {}  # To track session limits in the concatenated matrix
    current_index = 0
    channel_names = None

    # Step 1: Read and concatenate data for each session
    for session in sessions:
        session_data = []  # To store data for the current session across all subjects

        for subject in subject_list:
            subject_file = os.path.join(input_dir, f"sub-{subject}", f"peaks_and_limits_sub-{subject}.npz")
            if not os.path.exists(subject_file):
                print(f"File not found for subject {subject}: {subject_file}")
                continue

            # Load the subject's peaks and session limits
            data = np.load(subject_file, allow_pickle=True)
            peaks = data["Peaks"]
            limits = data["SessionLimits"].item()  # Convert back to dictionary

            # Extract the data for the current session
            if session in limits:
                start, end = limits[session]
                session_data.append(peaks[:, start:end + 1])  # Add session-specific data

                # Save channel names from the first subject
                if channel_names is None:
                    channel_names = peaks.shape[0]

        # Concatenate all subjects' data for the current session
        if session_data:
            session_data_concat = np.concatenate(session_data, axis=1)  # Horizontal concatenation
            all_selected_data.append(session_data_concat)

            # Track session limits in the concatenated matrix
            start_index = current_index
            end_index = current_index + session_data_concat.shape[1] - 1
            session_limits[session] = (start_index, end_index)
            current_index = end_index + 1

    # Step 2: Concatenate all sessions' data
    if not all_selected_data:
        print("No data found for any session or subject.")
        return None

    all_selected_data = np.concatenate(all_selected_data, axis=1)  # Final concatenated matrix

    # Save the reordered peaks matrix and session limits
    out_dir = os.path.join(save_dir, "concatenated_peaks")
    os.makedirs(out_dir, exist_ok=True)
    reordered_save_path = os.path.join(out_dir, "reordered_peaks_and_limits.npz")
    np.savez(reordered_save_path, Peaks=all_selected_data, SessionLimits=session_limits)
    print(f"Reordered peaks matrix and session limits saved to {reordered_save_path}")

    return reordered_save_path


def segment_microstates_peaks(
    concatenated_file,
    save_dir,
    n_microstates=10,
    method='kmod',
    sampling_rate=1024,
    random_state=42
):
    """
    Load the concatenated matrix, perform microstate segmentation, and save the results.

    Parameters
    ----------
    concatenated_file : str
        Path to the concatenated matrix file.
    save_dir : str
        Directory where the segmentation results will be saved.
    n_microstates : int
        Number of microstates to compute.
    method : str
        Clustering method for NeuroKit2 microstates_segment ('kmeans', 'pca', etc.).
    sampling_rate : int
        Sampling rate of the data.
    random_state : int
        Random seed.

    Returns
    -------
    None
    """
    # Load the concatenated matrix and session limits
    data = np.load(concatenated_file, allow_pickle=True)
    all_selected_data = data["Peaks"]
    session_limits = data["SessionLimits"].item()  # Convert back to dictionary

    # Step 1: Compute microstate segmentation
    print("Computing microstate segmentation...")
    microstates_results = nk.microstates_segment(
        all_selected_data,
        n_microstates=n_microstates,
        train=np.arange(all_selected_data.shape[1]),  # Use all selected peaks for training
        method=method,
        sampling_rate=sampling_rate,
        gfp_method='l2',
        random_state=random_state,
    )
    """
    silhouette, gap_statistic = compute_clustering_metrics(data["Peaks"].T, 
                                                          microstates_results["Sequence"],
                                                          n_clusters=n_microstates,
                                                          random_state=random_state)

    """
    # Step 2: Save the results
    out_dir = os.path.join(save_dir, f"{n_microstates}_microstates_segmentation")
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "microstates_all_subjects_results.npz")
    np.savez(
        results_path,
        Microstates=microstates_results["Microstates"],
        Sequence=microstates_results["Sequence"],
        GFP=microstates_results["GFP"],
        GEV=microstates_results["GEV"],
        SessionLimits=session_limits
    )
    print(f"Microstate results saved to {results_path}")


def backfit_microstates_peaks(
    microstate_maps,
    subject_file,
    save_dir
):
    """
    Backfit microstates to a subject's peaks matrix by computing the maximum dot product.

    Parameters
    ----------
    microstate_maps : np.ndarray
        2D array of microstate maps (rows: ROIs, columns: microstates).
    subject_file : str
        Path to the subject's .npz file containing the peaks matrix and session limits.
    save_dir : str
        Directory where the backfitting results will be saved.
    sessions : list of str
        List of session IDs.

    Returns
    -------
    None
    """
    # Load the subject's peaks matrix and session limits
    if not os.path.exists(subject_file):
        print(f"Subject file not found: {subject_file}")
        return

    data = np.load(subject_file, allow_pickle=True)
    peaks_matrix = data["Peaks"]  # Subject's peaks matrix
    session_limits = data["SessionLimits"].item()  # Session limits dictionary

    # Initialize an array to store the backfitted microstate labels
    backfitted_labels = np.zeros(peaks_matrix.shape[1], dtype=int)

    # Perform backfitting for each peak
    activation = microstate_maps.dot(peaks_matrix)
    backfitted_labels = np.argmax(np.abs(activation), axis=0)
    # Save the backfitted labels and session limits
    subject_id = os.path.basename(subject_file).split('-')[1].split('.')[0] # Extract subject ID from filename
    backfit_save_path = os.path.join(save_dir, f"backfitted_labels_sub-{subject_id}.npz")

    gev, gev_all = _cluster_quality_gev(peaks_matrix.T,
                                        microstate_maps,
                                        backfitted_labels,
                                        n_clusters=microstate_maps.shape[0]
    )

    """
    silhouette, gap = compute_clustering_metrics(peaks_matrix.T,
                                                 sequence=backfitted_labels,
                                                 n_clusters=microstate_maps.shape[0],
                                                       )
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    np.savez(
        backfit_save_path,
        Sequence=backfitted_labels,
        SessionLimits=session_limits,
        GEV=gev,
        GEV_all=gev_all,
        #Silhouette=silhouette,
        #GapStatistic=gap
    )
    print(f"Backfitted microstate labels saved to {backfit_save_path}")




def compute_gap_statistic(data, labels, n_clusters, random_state=42):
    """
    Compute the gap statistic for clustering quality.

    Parameters:
    ----------
    data : np.ndarray
        The data points (shape: n_samples x n_features).
    labels : np.ndarray
        The cluster assignments for each data point.
    n_clusters : int
        The number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns:
    -------
    float
        The gap statistic value.
    """
    # Compute intra-cluster distances for the actual data
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    kmeans.cluster_centers_ = np.array([data[labels == i].mean(axis=0) for i in range(n_clusters)])
    intra_cluster_distances = np.sum(
        np.min(np.linalg.norm(data[:, None] - kmeans.cluster_centers_, axis=2), axis=1)
    )

    # Generate random data within the same bounds as the original data
    random_data = np.random.uniform(
        np.min(data, axis=0), np.max(data, axis=0), size=data.shape
    )

    # Compute intra-cluster distances for the random data
    random_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    random_kmeans.fit(random_data)
    random_intra_cluster_distances = np.sum(
        np.min(np.linalg.norm(random_data[:, None] - random_kmeans.cluster_centers_, axis=2), axis=1)
    )

    # Compute the gap statistic
    return np.log(random_intra_cluster_distances) - np.log(intra_cluster_distances)

def compute_clustering_metrics(data, sequence, n_clusters, random_state=42):
    """
    Compute clustering quality metrics (gap statistic and silhouette score) and save them.

    Parameters:
    ----------
    data : np.ndarray
        The data points (shape: n_samples x n_features).
    sequence : np.ndarray
        The cluster assignments for each data point.
    n_clusters : int
        The number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns:
    -------
    silhouette : float
        The silhouette score.
    gap_statistic : float
        The gap statistic value.
    """
    print("Computing clustering quality metrics...")

    # Compute silhouette score
    silhouette = silhouette_score(data, sequence)
    print(f"Silhouette Score: {silhouette:.4f}")

    # Compute gap statistic
    gap_statistic = compute_gap_statistic(data, sequence, n_clusters, random_state=random_state)
    print(f"Gap Statistic: {gap_statistic:.4f}")

    return silhouette, gap_statistic