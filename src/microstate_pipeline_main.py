import os
from microstate_analysis import (extract_peaks,
                                 concatenate_peaks,
                                 segment_microstates,
                                 backfit_microstates
                                )
import numpy as np

from plot_F import plot_microstate_Fstatistics
from repeated_anova_backfit import process_microstate_results


if __name__ == "__main__":
    # Define constants and directories
    aal_labels = [
        'Rectus_L', 'Olfactory_L', 'Frontal_Sup_Orb_L', 'Frontal_Med_Orb_L', 'Frontal_Mid_Orb_L',
        'Frontal_Inf_Orb_L', 'Frontal_Sup_L', 'Frontal_Mid_L', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Tri_L',
        'Frontal_Sup_Medial_L', 'Supp_Motor_Area_L', 'Paracentral_Lobule_L', 'Precentral_L', 'Rolandic_Oper_L',
        'Postcentral_L', 'Parietal_Sup_L', 'Parietal_Inf_L', 'SupraMarginal_L', 'Angular_L', 'Precuneus_L',
        'Occipital_Sup_L', 'Occipital_Mid_L', 'Occipital_Inf_L', 'Calcarine_L', 'Cuneus_L', 'Lingual_L',
        'Fusiform_L', 'Heschl_L', 'Temporal_Sup_L', 'Temporal_Mid_L', 'Temporal_Inf_L', 'Temporal_Pole_Sup_L',
        'Temporal_Pole_Mid_L', 'ParaHippocampal_L', 'Cingulum_Ant_L', 'Cingulum_Mid_L', 'Cingulum_Post_L',
        'Insula_L', 'Rectus_R', 'Olfactory_R', 'Frontal_Sup_Orb_R', 'Frontal_Med_Orb_R', 'Frontal_Mid_Orb_R',
        'Frontal_Inf_Orb_R', 'Frontal_Sup_R', 'Frontal_Mid_R', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_R',
        'Frontal_Sup_Medial_R', 'Supp_Motor_Area_R', 'Paracentral_Lobule_R', 'Precentral_R', 'Rolandic_Oper_R',
        'Postcentral_R', 'Parietal_Sup_R', 'Parietal_Inf_R', 'SupraMarginal_R', 'Angular_R', 'Precuneus_R',
        'Occipital_Sup_R', 'Occipital_Mid_R', 'Occipital_Inf_R', 'Calcarine_R', 'Cuneus_R', 'Lingual_R',
        'Fusiform_R', 'Heschl_R', 'Temporal_Sup_R', 'Temporal_Mid_R', 'Temporal_Inf_R', 'Temporal_Pole_Sup_R',
        'Temporal_Pole_Mid_R', 'ParaHippocampal_R', 'Cingulum_Ant_R', 'Cingulum_Mid_R', 'Cingulum_Post_R',
        'Insula_R', 'Hippocampus_L', 'Hippocampus_R', 'Amygdala_L', 'Amygdala_R', 'Caudate_L', 'Caudate_R',
        'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R'
    ]
    n_subjects = 2 # change according to your dataset
    subject_list = [f"{i:02d}" for i in range(1, n_subjects)]  # Subject IDs from '01'
    # bids root directory
    bids_root = '/Users/matte/Desktop/Naples_data/bids_deprivation' 
    n_peaks = 1000  # Number of GFP peaks to select from each acquisition of session of each subject
    peak_distance = 10  # Minimum distance between peaks in samples
    method = 'kmod'  # Specify the clustering method (e.g., 'kmeans', 'pca')
    minlenE = 2048  # Minimum number of time points required for a run to be included
    l_freq = 2.0
    h_freq = 30.0
    smoothing_window = 5
    sampling_rate = 1024
    start_microstates = 2
    end_microstates = 5
    # output directory
    base_output_dir = './reports/microstate_results/deprivation/'
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    # Define sessions
    sessions = ['00', '01', '02', '03']
    
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    # Directory to save individual subject peaks
    save_dir_peaks = os.path.join(base_output_dir, 'individual_subj_peaks/')

     # Save parameters to a .txt file
    parameters_file = os.path.join(base_output_dir, "parameters.txt")
    with open(parameters_file, "w") as f:
        f.write("Parameters:\n")
        f.write(f"n_peaks: {n_peaks}\n")
        f.write(f"peak_distance: {peak_distance}\n")
        f.write(f"method: {method}\n")
        f.write(f"minlenE: {minlenE}\n")
        f.write(f"l_freq: {l_freq}\n")
        f.write(f"h_freq: {h_freq}\n")
        f.write(f"smoothing_window: {smoothing_window}\n")
        f.write(f"sampling_rate: {sampling_rate}\n")
        f.write(f"start_microstates: {start_microstates}\n")
        f.write(f"end_microstates: {end_microstates}\n")
    print(f"Parameters saved to {parameters_file}")


    # Extract peaks for each subject if not already done
    if not os.path.exists(save_dir_peaks) or len(os.listdir(save_dir_peaks)) == 0:
        for subj in subject_list:
            extract_peaks(
                subject=subj,
                bids_root=bids_root,
                sessions=sessions,
                save_dir=save_dir_peaks,
                l_freq=l_freq,
                h_freq=h_freq,
                n_peaks=n_peaks,
                min_run_length=minlenE,
                peak_distance=peak_distance,
                smoothing_window=smoothing_window,
            )

    # Concatenate the Peaks for all subjects and sessions
    if not os.path.exists(os.path.join(base_output_dir, 'concatenated_peaks')):
        concatenate_peaks(
            subject_list,
            input_dir=save_dir_peaks,
            save_dir=base_output_dir,
            sessions=sessions
        )

    # Classify the peaks to obtain the microstates
    concatenate_peaks_file = os.path.join(base_output_dir, 'concatenated_peaks', 'reordered_peaks_and_limits.npz')

    for n_microstates in range(start_microstates, end_microstates):  # Loop from 2 to 15 microstates
        segmentation_file = os.path.join(base_output_dir, f'{n_microstates}_microstates_segmentation', 'microstates_all_subjects_results.npz')
        if not os.path.exists(segmentation_file):
            print(f"Segmenting microstates for {n_microstates} microstates...")
            segment_microstates(
                concatenated_file=concatenate_peaks_file,
                save_dir=base_output_dir,
                n_microstates=n_microstates,
                method=method,
                random_state=42
            )
        else:
            print(f"Segmentation already exists for {n_microstates} microstates.")

        # Load the microstate maps
        data = np.load(segmentation_file, allow_pickle=True)
        microstate_maps = data['Microstates']
        out_backfit_dir = os.path.join(base_output_dir, f'{n_microstates}_backfitted_microstates')

        # Backfit the microstates to each subject and plot the results
        for subj in subject_list:
            subject_file = os.path.join(save_dir_peaks, f'sub-{subj}', f'peaks_and_limits_sub-{subj}.npz')
            if not os.path.exists(os.path.join(out_backfit_dir, f'backfitted_microstates_sub-{subj}')):
                backfit_microstates(
                    microstate_maps,
                    subject_file,
                    out_backfit_dir
                )

    process_microstate_results(base_folder=base_output_dir)
    plot_microstate_Fstatistics(base_folder=base_output_dir,alpha=0.5)
    