"""
Cross-validation workflow for microstate analysis using the PyCrostates API.
"""

import os
import glob
import numpy as np
import pandas as pd
import mne
from pycrostates.io import ChData, read_cluster
from pycrostates.cluster import ModKMeans
from scipy.stats import permutation_test, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

from micros_based_stats.microstate_analysis_pycrostate import cohens_d_statistic


def perform_clustering_cv(subjects, gfp_peaks_dir, out_dir, n_microstates, infoStub):
    """
    Perform clustering on GFP peaks from a subset of subjects (training set).

    Parameters:
    - subjects (list of str): List of subject IDs to include.
    - gfp_peaks_dir (str): Directory containing per-subject GFP peak files.
    - out_dir (str): Directory where clustering results will be saved.
    - n_microstates (int): Number of microstates to compute.
    - infoStub (mne.Info): MNE Info object for creating ChData.
    """
    all_peaks = []
    for sub in subjects:
        peak_files = glob.glob(os.path.join(gfp_peaks_dir, f'{sub}*_gfp_peaks.npz'))
        for pf in peak_files:
            data = np.load(pf)
            all_peaks.append(data['peak_data'])

    if not all_peaks:
        raise ValueError("No GFP peak files found for the given subjects.")

    combined_peaks = np.concatenate(all_peaks, axis=1)
    peaks_ch = ChData(data=combined_peaks, info=infoStub)

    clustering = ModKMeans(n_clusters=n_microstates, random_state=42)
    clustering.fit(peaks_ch, picks='all')

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{n_microstates}_clustering.fif")
    clustering.save(save_path)
    print(f"CV clustering ({n_microstates} states) saved to {save_path}")


def backfit_data_cv(clustering_f,
                    preprocessed_data_dir,
                    subjects,
                    fold,
                    factor=0,
                    half_window_size=10,
                    min_segment_length=5,
                    out_dir=None):
    """
    Backfit microstates to preprocessed data for a subset of subjects,
    recording the fold number.

    Parameters:
    - clustering_f (str): Path to the clustering .fif file.
    - preprocessed_data_dir (str): Directory containing preprocessed .fif files.
    - subjects (list of str): Subject IDs to backfit (e.g. ['sub-01', 'sub-02']).
    - fold (int): Fold number to record in the output.
    - factor (int): Smoothing factor for prediction.
    - half_window_size (int): Half window size for prediction.
    - min_segment_length (int): Minimum segment length for prediction.
    - out_dir (str): Directory to save backfitting results.

    Returns:
    - measures_df (pd.DataFrame): DataFrame with backfit metrics including fold column.
    """
    clustering = read_cluster(clustering_f)
    n_clusters = clustering.n_clusters

    metric_names = ['mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs']
    columns = (['sub', 'ses', 'fold', 'n_clusters', 'entropy', 'unlabeled'] +
               [f'{metric}_{i}' for i in range(n_clusters) for metric in metric_names])
    measures_df = pd.DataFrame(columns=columns)

    # Build file list filtered to the requested subjects
    all_files = glob.glob(os.path.join(preprocessed_data_dir, '*.fif'))
    file_list = [f for f in all_files
                 if any(sub in os.path.basename(f) for sub in subjects)]

    if not file_list:
        print(f"No preprocessed files found for fold {fold} subjects.")
        return measures_df

    for f in file_list:
        print(f"Backfitting (fold {fold}): {os.path.basename(f)}")
        raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
        segmentation = clustering.predict(raw,
                                          factor=factor,
                                          half_window_size=half_window_size,
                                          min_segment_length=min_segment_length,
                                          reject_edges=True,
                                          reject_by_annotation=True)
        subject, session, _ = raw.filenames[0].name.split('_')[:-1]
        measures = segmentation.compute_parameters()
        e_h = segmentation.entropy(ignore_repetitions=False)

        row = {
            'sub': subject,
            'ses': session,
            'fold': fold,
            'n_clusters': n_clusters,
            'entropy': e_h,
            'unlabeled': measures.get('unlabeled', 0),
        }
        for i in range(n_clusters):
            for metric in metric_names:
                row[f'{metric}_{i}'] = measures.get(f'{i}_{metric}', np.nan)

        measures_df = pd.concat([measures_df, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    file_out = os.path.join(out_dir,
                            f'backfit_cv_fold_{fold}_{n_clusters}_microstates_measures.csv')
    measures_df.to_csv(file_out, index=False)
    print(f"CV backfit metrics (fold {fold}, k={n_clusters}) saved to {file_out}")
    return measures_df


def permutation_test_metrics_cv(backfit_cv_dir,
                                participants_tsv,
                                metric='gev',
                                n_perms=1000,
                                out_dir=None):
    """
    Perform permutation tests on a chosen backfit metric (per microstate)
    comparing case vs control, across all folds and n_clusters values.

    Parameters:
    - backfit_cv_dir (str): Directory containing backfit_cv CSV files.
    - participants_tsv (str): Path to participants.tsv with 'participant_id' and 'case_ctrl'.
    - metric (str): Metric to test. One of 'mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs'.
    - n_perms (int): Number of permutation resamples.
    - out_dir (str): Directory to save results.

    Returns:
    - results_df (pd.DataFrame): DataFrame with permutation test results.
    """
    part_df = pd.read_csv(participants_tsv, sep='\t')
    part_df = part_df.rename(columns={'participant_id': 'sub'})

    csv_files = sorted(glob.glob(os.path.join(backfit_cv_dir, 'backfit_cv_fold_*_measures.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No backfit_cv CSV files found in {backfit_cv_dir}")

    all_results = []
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        fold = df['fold'].iloc[0]
        n_clusters = int(df['n_clusters'].iloc[0])
        merged = pd.merge(df, part_df[['sub', 'case_ctrl']], on='sub', how='left')

        # Test the chosen metric for each microstate
        for i in range(n_clusters):
            col = f'{metric}_{i}'
            if col not in merged.columns:
                continue
            group_ctrl = merged[merged['case_ctrl'] == 0][col].dropna()
            group_case = merged[merged['case_ctrl'] == 1][col].dropna()

            if len(group_ctrl) < 2 or len(group_case) < 2:
                continue

            res = permutation_test(
                (group_ctrl, group_case),
                cohens_d_statistic,
                permutation_type='independent',
                n_resamples=n_perms,
                alternative='two-sided'
            )

            all_results.append({
                'fold': fold,
                'n_clusters': n_clusters,
                'microstate': i,
                'metric': metric,
                'statistic': res.statistic,
                'p_value': res.pvalue
            })

    results_df = pd.DataFrame(all_results)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'permtest_cv_{metric}.csv')
        results_df.to_csv(out_file, index=False)
        print(f"CV permutation test results for '{metric}' saved to {out_file}")
    return results_df


def ttest_metrics_cv(backfit_cv_dir,
                     participants_tsv,
                     metric='gev',
                     out_dir=None):
    """
    Perform independent-samples t-tests on a chosen backfit metric (per microstate)
    comparing case vs control, across all folds and n_clusters values.

    Parameters:
    - backfit_cv_dir (str): Directory containing backfit_cv CSV files.
    - participants_tsv (str): Path to participants.tsv with 'participant_id' and 'case_ctrl'.
    - metric (str): Metric to test. One of 'mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs'.
    - out_dir (str): Directory to save results.

    Returns:
    - results_df (pd.DataFrame): DataFrame with t-test results.
    """
    part_df = pd.read_csv(participants_tsv, sep='\t')
    part_df = part_df.rename(columns={'participant_id': 'sub'})

    csv_files = sorted(glob.glob(os.path.join(backfit_cv_dir, 'backfit_cv_fold_*_measures.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No backfit_cv CSV files found in {backfit_cv_dir}")

    all_results = []
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        fold = df['fold'].iloc[0]
        n_clusters = int(df['n_clusters'].iloc[0])
        merged = pd.merge(df, part_df[['sub', 'case_ctrl']], on='sub', how='left')

        for i in range(n_clusters):
            col = f'{metric}_{i}'
            if col not in merged.columns:
                continue
            group_ctrl = merged[merged['case_ctrl'] == 0][col].dropna()
            group_case = merged[merged['case_ctrl'] == 1][col].dropna()

            if len(group_ctrl) < 2 or len(group_case) < 2:
                continue

            t_stat, p_value = ttest_ind(group_ctrl, group_case, equal_var=False)

            all_results.append({
                'fold': fold,
                'n_clusters': n_clusters,
                'microstate': i,
                'metric': metric,
                't_statistic': t_stat,
                'p_value': p_value
            })

    results_df = pd.DataFrame(all_results)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'ttest_cv_{metric}.csv')
        results_df.to_csv(out_file, index=False)
        print(f"CV t-test results for '{metric}' saved to {out_file}")
    return results_df


def ttest_validation_cv(validation_dir,
                        participants_tsv,
                        metric='occurrences',
                        out_dir=None):
    """
    Perform independent-samples t-tests on a chosen validation metric
    (per microstate) comparing case vs control, across all n_clusters
    values within a single fold's validation results.

    Parameters:
    - validation_dir (str): Directory containing validation CSV files
      (validation_fold_{fold}_{n_clusters}_microstates.csv).
    - participants_tsv (str): Path to participants.tsv with 'participant_id' and 'case_ctrl'.
    - metric (str): Metric to test. One of 'mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs'.
    - out_dir (str): Directory to save results.

    Returns:
    - results_df (pd.DataFrame): DataFrame with t-test results.
    """
    part_df = pd.read_csv(participants_tsv, sep='\t')
    part_df = part_df.rename(columns={'participant_id': 'sub'})

    csv_files = sorted(glob.glob(os.path.join(validation_dir,
                                              'validation_fold_*_microstates.csv')))
    if not csv_files:
        raise FileNotFoundError(
            f"No validation CSV files found in {validation_dir}")

    all_results = []
    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        fold = df['fold'].iloc[0]
        n_clusters = int(df['n_clusters'].iloc[0])
        merged = pd.merge(df, part_df[['sub', 'case_ctrl']], on='sub', how='left')

        for i in range(n_clusters):
            col = f'{metric}_{i}'
            if col not in merged.columns:
                continue
            group_ctrl = merged[merged['case_ctrl'] == 0][col].dropna()
            group_case = merged[merged['case_ctrl'] == 1][col].dropna()

            if len(group_ctrl) < 2 or len(group_case) < 2:
                continue

            t_stat, p_value = ttest_ind(group_ctrl, group_case, equal_var=False)

            all_results.append({
                'fold': fold,
                'n_clusters': n_clusters,
                'microstate': i,
                'metric': metric,
                't_statistic': np.abs(t_stat),
                'p_value': p_value
            })

    results_df = pd.DataFrame(all_results)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'ttest_validation_{metric}.csv')
        results_df.to_csv(out_file, index=False)
        print(f"Validation t-test results for '{metric}' saved to {out_file}")
    return results_df


def validate_results(clustering_dir,
                     preprocessed_data_dir,
                     test_subjects,
                     fold,
                     n_microstates,
                     factor=0,
                     half_window_size=10,
                     min_segment_length=5,
                     out_dir=None):
    """
    Validate the clustering solution on held-out test subjects by backfitting
    and computing metrics.

    Parameters:
    - clustering_dir (str): Directory containing the fold's clustering files.
    - preprocessed_data_dir (str): Directory containing preprocessed .fif files.
    - test_subjects (list of str): Subject IDs for the test set.
    - fold (int): Fold number.
    - n_microstates (int): Number of microstates (selects which clustering file).
    - factor (int): Smoothing factor for prediction.
    - half_window_size (int): Half window size for prediction.
    - min_segment_length (int): Minimum segment length for prediction.
    - out_dir (str): Directory to save validation results.

    Returns:
    - validation_df (pd.DataFrame): DataFrame with validation metrics.
    """
    clustering_f = os.path.join(clustering_dir, f"{n_microstates}_clustering.fif")
    clustering = read_cluster(clustering_f)
    n_clusters = clustering.n_clusters

    metric_names = ['mean_corr', 'gev', 'occurrences', 'timecov', 'meandurs']
    columns = (['sub', 'ses', 'fold', 'n_clusters', 'set', 'entropy', 'unlabeled'] +
               [f'{metric}_{i}' for i in range(n_clusters) for metric in metric_names])
    validation_df = pd.DataFrame(columns=columns)

    all_files = glob.glob(os.path.join(preprocessed_data_dir, '*.fif'))
    file_list = [f for f in all_files
                 if any(sub in os.path.basename(f) for sub in test_subjects)]

    if not file_list:
        print(f"No preprocessed files found for test subjects in fold {fold}.")
        return validation_df

    for f in file_list:
        print(f"Validating (fold {fold}, test): {os.path.basename(f)}")
        raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
        segmentation = clustering.predict(raw,
                                          factor=factor,
                                          half_window_size=half_window_size,
                                          min_segment_length=min_segment_length,
                                          reject_edges=True,
                                          reject_by_annotation=True)
        subject, session, _ = raw.filenames[0].name.split('_')[:-1]
        measures = segmentation.compute_parameters()
        e_h = segmentation.entropy(ignore_repetitions=False)

        row = {
            'sub': subject,
            'ses': session,
            'fold': fold,
            'n_clusters': n_clusters,
            'set': 'test',
            'entropy': e_h,
            'unlabeled': measures.get('unlabeled', 0),
        }
        for i in range(n_clusters):
            for metric in metric_names:
                row[f'{metric}_{i}'] = measures.get(f'{i}_{metric}', np.nan)

        validation_df = pd.concat([validation_df, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    file_out = os.path.join(out_dir,
                            f'validation_fold_{fold}_{n_clusters}_microstates.csv')
    validation_df.to_csv(file_out, index=False)
    print(f"Validation metrics (fold {fold}, k={n_clusters}) saved to {file_out}")
    return validation_df


def main_workflow_cv(participants_tsv,
                     bids_root,
                     gfp_peaks_dir,
                     preprocessed_data_dir,
                     base_out_dir,
                     n_folds=5,
                     n_microstates_list=None,
                     infoStub=None,
                     factor=0,
                     half_window_size=10,
                     min_segment_length=5,
                     metric='gev',
                     n_perms=1000):
    """
    Main cross-validation workflow for microstate analysis.

    For each fold:
      1. Split subjects into balanced train/test (equal case/control).
      2. Cluster GFP peaks from training subjects for each n_microstates.
      3. Backfit training subjects and save metrics with fold number.
      4. Run t-tests on chosen metric (case vs control).
    Then across all folds:
      5. Order t-test results by descending |t_statistic|.
      6. Reorder clustering maps accordingly.
      7. Validate on held-out test subjects using reordered clusters.

    Parameters:
    - participants_tsv (str): Path to participants.tsv (columns: participant_id, case_ctrl).
    - bids_root (str): Path to BIDS root directory.
    - gfp_peaks_dir (str): Directory with per-subject GFP peak .npz files.
    - preprocessed_data_dir (str): Directory with preprocessed .fif files.
    - base_out_dir (str): Base directory where all CV results are saved.
    - n_folds (int): Number of cross-validation folds.
    - n_microstates_list (list of int): List of n_microstates values to evaluate.
    - infoStub (mne.Info): MNE Info object for ChData creation.
    - factor (int): Smoothing factor for backfit prediction.
    - half_window_size (int): Half window size for backfit prediction.
    - min_segment_length (int): Minimum segment length for backfit prediction.
    - metric (str): Metric for permutation testing ('gev', 'mean_corr', etc.).
    - n_perms (int): Number of permutations for the permutation test.
    """
    if n_microstates_list is None:
        n_microstates_list = list(range(2, 13))

    # Load participants
    part_df = pd.read_csv(participants_tsv, sep='\t')
    part_df = part_df.rename(columns={'participant_id': 'sub'})

    cases = part_df[part_df['case_ctrl'] == 1]['sub'].values
    controls = part_df[part_df['case_ctrl'] == 0]['sub'].values

    # Shuffle reproducibly
    rng = np.random.default_rng(seed=42)
    rng.shuffle(cases)
    rng.shuffle(controls)

    case_folds = np.array_split(cases, n_folds)
    control_folds = np.array_split(controls, n_folds)

    os.makedirs(base_out_dir, exist_ok=True)

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1} / {n_folds}")
        print(f"{'='*60}")

        # Split into train / test
        test_cases = list(case_folds[fold])
        test_controls = list(control_folds[fold])
        train_cases = [s for i, cf in enumerate(case_folds) if i != fold for s in cf]
        train_controls = [s for i, cf in enumerate(control_folds) if i != fold for s in cf]

        train_subjects = train_cases + train_controls
        test_subjects = test_cases + test_controls

        print(f"  Train: {len(train_subjects)} subjects "
              f"({len(train_cases)} case, {len(train_controls)} ctrl)")
        print(f"  Test:  {len(test_subjects)} subjects "
              f"({len(test_cases)} case, {len(test_controls)} ctrl)")

        fold_dir = os.path.join(base_out_dir, f"fold_{fold}")
        clustering_dir = os.path.join(fold_dir, "clustering")
        backfit_dir = os.path.join(fold_dir, "backfit_train")
        validation_dir = os.path.join(fold_dir, "validation_test")
        test_dir = os.path.join(fold_dir, "t_test")

        ttest_file = os.path.join(test_dir, f'ttest_cv_{metric}.csv')
        if os.path.exists(ttest_file):
            print(f"  Results already exist ({ttest_file}), skipping fold {fold}.")
            continue

        for n in n_microstates_list:
            # Step 1: Cluster GFP peaks from training subjects
            perform_clustering_cv(train_subjects, gfp_peaks_dir,
                                  clustering_dir, n, infoStub)

            # Step 2: Backfit training subjects
            backfit_data_cv(
                clustering_f=os.path.join(clustering_dir, f"{n}_clustering.fif"),
                preprocessed_data_dir=preprocessed_data_dir,
                subjects=train_subjects,
                fold=fold,
                factor=factor,
                half_window_size=half_window_size,
                min_segment_length=min_segment_length,
                out_dir=backfit_dir
            )

        # Step 3: T-test on training backfit metrics
        ttest_metrics_cv(
            backfit_cv_dir=backfit_dir,
            participants_tsv=participants_tsv,
            metric=metric,
            out_dir=test_dir
        )

    # Step 4: Order t-test files and reorder clusters
    order_ttest_cv_files(base_out_dir, metric)
    reorder_clusters_cv(base_out_dir, metric)

    # Step 5: Validate on test subjects using reordered clusters
    for fold in range(n_folds):
        test_cases = list(case_folds[fold])
        test_controls = list(control_folds[fold])
        test_subjects = test_cases + test_controls

        fold_dir = os.path.join(base_out_dir, f"fold_{fold}")
        ordered_dir = os.path.join(fold_dir, "cluster_ordered")
        validation_dir = os.path.join(fold_dir, "validation_test")

        for n in n_microstates_list:
            validate_results(
                clustering_dir=ordered_dir,
                preprocessed_data_dir=preprocessed_data_dir,
                test_subjects=test_subjects,
                fold=fold,
                n_microstates=n,
                factor=factor,
                half_window_size=half_window_size,
                min_segment_length=min_segment_length,
                out_dir=validation_dir
            )

        # Step 6: T-test on validation metrics
        ttest_validation_cv(
            validation_dir=validation_dir,
            participants_tsv=participants_tsv,
            metric=metric,
            out_dir=os.path.join(fold_dir, "t_test_validation")
        )

    print(f"\nCross-validation complete. Results in {base_out_dir}")


def plot_ttest_cv_swarm(base_out_dir, metric='occurrences', stat='t_statistic',
                        dataset='train'):
    """
    Collect t-test CSV files from all folds and create a swarm plot organized
    by n_clusters with fold as hue.

    Parameters:
    - base_out_dir (str): Base CV output directory containing fold_* subdirectories.
    - metric (str): Metric name used in the t-test (matches the CSV filename).
    - stat (str): Which column to plot on the y-axis ('t_statistic' or 'p_value').
    - dataset (str): 'train' to use t_test/ttest_cv_{metric}_ordered.csv,
      'validation' to use t_test_validation/ttest_validation_{metric}.csv.
    """
    if dataset == 'train':
        pattern = os.path.join(base_out_dir, 'fold_*', 't_test',
                               f'ttest_cv_{metric}_ordered.csv')
    elif dataset == 'validation':
        pattern = os.path.join(base_out_dir, 'fold_*', 't_test_validation',
                               f'ttest_validation_{metric}.csv')
    else:
        raise ValueError(f"dataset must be 'train' or 'validation', got '{dataset}'")

    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No t-test CSV files found matching {pattern}")

    dfs = [pd.read_csv(f) for f in csv_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined['fold'] = combined['fold'].astype(int)
    combined['n_clusters'] = combined['n_clusters'].astype(int)

    fig, ax = plt.subplots(figsize=(max(12, len(combined['n_clusters'].unique()) * 0.6), 6))
    sns.swarmplot(data=combined, x='n_clusters', y=stat, hue='fold',
                  palette='tab10', dodge=True, size=4, ax=ax)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel(stat.replace('_', ' ').title())
    ax.set_title(f'{dataset.capitalize()} t-test {stat.replace("_", " ")} '
                 f'for "{metric}" across CV folds')
    ax.legend(title='Fold', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(base_out_dir,
                            f'swarm_ttest_{dataset}_{metric}_{stat}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Swarm plot saved to {out_path}")

    csv_out = os.path.join(base_out_dir,
                           f'combined_ttest_{dataset}_{metric}.csv')
    combined.to_csv(csv_out, index=False)
    print(f"Combined dataframe saved to {csv_out}")

    plt.show()
    return combined


def reorder_clusters_cv(base_out_dir, metric='occurrences'):
    """
    Reorder clustering maps for each fold based on the descending |t_statistic|
    order stored in the per-fold ttest_cv_{metric}_ordered.csv files.

    Assumes order_ttest_cv_files has already been run to produce the _ordered files.

    For each (fold, n_clusters), reads the original clustering file, reorders
    the cluster maps according to unordered_microstate, and saves the result
    in a new 'cluster_ordered' folder within each fold directory.

    Parameters:
    - base_out_dir (str): Base CV output directory containing fold_* subdirectories.
    - metric (str): Metric name used in the t-test (matches the CSV filename).
    """
    pattern = os.path.join(base_out_dir, 'fold_*', 't_test',
                           f'ttest_cv_{metric}_ordered.csv')
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No ttest_cv_{metric}_ordered.csv files found in "
            f"{base_out_dir}/fold_*/t_test/")

    combined = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined['fold'] = combined['fold'].astype(int)
    combined['n_clusters'] = combined['n_clusters'].astype(int)
    combined['unordered_microstate'] = combined['unordered_microstate'].astype(int)

    for fold, fold_df in combined.groupby('fold'):
        fold_dir = os.path.join(base_out_dir, f"fold_{fold}")
        clustering_dir = os.path.join(fold_dir, "clustering")
        ordered_dir = os.path.join(fold_dir, "cluster_ordered")
        os.makedirs(ordered_dir, exist_ok=True)

        for n_clusters, grp in fold_df.groupby('n_clusters'):
            # Sort by new microstate index (0, 1, 2, …) to get the reorder mapping
            grp = grp.sort_values('microstate')
            new_order = grp['unordered_microstate'].tolist()

            clustering_f = os.path.join(clustering_dir, f"{n_clusters}_clustering.fif")
            if not os.path.exists(clustering_f):
                print(f"Warning: {clustering_f} not found, skipping.")
                continue

            clustering = read_cluster(clustering_f)
            clustering.reorder_clusters(order=new_order)

            save_path = os.path.join(ordered_dir, f"{n_clusters}_clustering.fif")
            clustering.save(save_path)
            print(f"Reordered clustering (fold {fold}, k={n_clusters}) saved to {save_path}")


def order_ttest_cv_files(base_out_dir, metric='occurrences'):
    """
    For each fold, read the ttest_cv_{metric}.csv, sort microstates by
    descending absolute t-statistic within each n_clusters, add an
    'unordered_microstate' column with the original index, renumber
    'microstate' as 0, 1, 2, …, and save as ttest_cv_{metric}_ordered.csv.

    Parameters:
    - base_out_dir (str): Base CV output directory containing fold_* subdirectories.
    - metric (str): Metric name used in the t-test (matches the CSV filename).
    """
    pattern = os.path.join(base_out_dir, 'fold_*', 't_test', f'ttest_cv_{metric}.csv')
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No ttest_cv_{metric}.csv files found in {base_out_dir}/fold_*/t_test/")

    for csv_f in csv_files:
        df = pd.read_csv(csv_f)
        df['t_statistic'] = np.abs(df['t_statistic'].astype(float))
        df = df.sort_values(['n_clusters', 't_statistic'], ascending=[True, False])
        df['unordered_microstate'] = df['microstate']
        df['microstate'] = df.groupby('n_clusters').cumcount()

        out_file = csv_f.replace(f'ttest_cv_{metric}.csv',
                                 f'ttest_cv_{metric}_ordered.csv')
        df.to_csv(out_file, index=False)
        print(f"Ordered t-test saved to {out_file}")


def cluster_correlation_across_folds(base_out_dir, n_microstates_list=None):
    """
    For each n_clusters value, load the reordered clustering from each fold
    and compute a polarity-invariant correlation matrix across all
    (fold, cluster) combinations. Save the matrix as an imshow figure.

    For example, with 5 folds and 4 states the matrix is 20×20: rows/columns
    are labelled fold0_map0, fold0_map1, …, fold4_map3.

    Parameters:
    - base_out_dir (str): Base CV output directory containing fold_* subdirectories,
      each with a 'cluster_ordered' folder.
    - n_microstates_list (list of int or None): Which n_clusters values to process.
      If None, auto-detected from fold_0/cluster_ordered/*.fif files.
    """
    fold_dirs = sorted(glob.glob(os.path.join(base_out_dir, 'fold_*', 'cluster_ordered')))
    if not fold_dirs:
        raise FileNotFoundError(
            f"No cluster_ordered directories found in {base_out_dir}/fold_*/")

    n_folds = len(fold_dirs)

    # Auto-detect n_microstates from fold_0 if not provided
    if n_microstates_list is None:
        fif_files = glob.glob(os.path.join(fold_dirs[0], '*_clustering.fif'))
        n_microstates_list = sorted(
            int(os.path.basename(f).split('_')[0]) for f in fif_files
        )

    out_dir = os.path.join(base_out_dir, 'cluster_correlations')
    os.makedirs(out_dir, exist_ok=True)

    for n_clusters in n_microstates_list:
        # Collect all cluster centers across folds
        all_maps = []
        labels = []
        for fold_idx, fd in enumerate(fold_dirs):
            clustering_f = os.path.join(fd, f"{n_clusters}_clustering.fif")
            if not os.path.exists(clustering_f):
                print(f"Warning: {clustering_f} not found, skipping.")
                continue
            clustering = read_cluster(clustering_f)
            centers = clustering._cluster_centers_  # shape (n_clusters, n_channels)
            for k in range(n_clusters):
                all_maps.append(centers[k])
                labels.append(f"f{fold_idx}_m{k}")

        if not all_maps:
            print(f"No maps found for n_clusters={n_clusters}, skipping.")
            continue

        maps_matrix = np.array(all_maps)  # (n_folds * n_clusters, n_channels)
        n = maps_matrix.shape[0]

        # Polarity-invariant similarity: cosine similarity (L2-normalized dot product)
        norms = np.linalg.norm(maps_matrix, axis=1, keepdims=True)
        maps_norm = maps_matrix / norms
        corr_matrix = np.abs(maps_norm @ maps_norm.T)

        # Save the correlation matrix as CSV
        corr_df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
        csv_path = os.path.join(out_dir, f'cluster_corr_{n_clusters}_states.csv')
        corr_df.to_csv(csv_path)

        # Plot
        fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(7, n * 0.4)))
        im = ax.imshow(corr_matrix, vmin=np.min(corr_matrix), vmax=1, cmap='RdYlBu_r',
                        interpolation='nearest')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=max(6, 10 - n // 10))
        ax.set_yticklabels(labels, fontsize=max(6, 10 - n // 10))
        ax.set_title(f'Polarity-invariant map correlation ({n_clusters} states, '
                     f'{n_folds} folds)')

        # Draw grid lines to separate folds
        for i in range(1, n_folds):
            pos = i * n_clusters - 0.5
            ax.axhline(pos, color='black', linewidth=1)
            ax.axvline(pos, color='black', linewidth=1)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('|Cosine similarity|')
        plt.tight_layout()

        fig_path = os.path.join(out_dir, f'cluster_corr_{n_clusters}_states.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Correlation matrix ({n_clusters} states) saved to {fig_path}")
        plt.close(fig)
