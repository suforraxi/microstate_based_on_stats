from nilearn import datasets
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_img_on_surf
import numpy as np
import os

def map_microstate_to_aal(microstate_maps, selected_row, aal_atlas, th_percentile=50):
    """
    Maps a selected row of microstate data to the AAL atlas and visualizes it.

    Parameters:
    - microstate_maps: numpy array, the microstate maps matrix.
    - selected_row: int, the row index of the microstate data to map.
    - my_aal_regions: list, list of AAL region names.
    - aal_atlas: dict, the AAL atlas fetched using nilearn.datasets.fetch_atlas_aal().

    Returns:
    - mapped_aal_nii: Nifti1Image, the mapped AAL atlas as a NIfTI image.
    """
    my_aal_regions = [
        'Rectus_L', 'Olfactory_L', 'Frontal_Sup_Orb_L', 'Frontal_Med_Orb_L', 'Frontal_Mid_Orb_L',
        'Frontal_Inf_Orb_L', 'Frontal_Sup_L', 'Frontal_Mid_L', 'Frontal_Inf_Oper_L', 'Frontal_Inf_Tri_L',
        'Frontal_Sup_Medial_L', 'Supp_Motor_Area_L', 'Paracentral_Lobule_L', 'Precentral_L', 'Rolandic_Oper_L',
        'Postcentral_L', 'Parietal_Sup_L', 'Parietal_Inf_L', 'SupraMarginal_L', 'Angular_L',
        'Precuneus_L', 'Occipital_Sup_L', 'Occipital_Mid_L', 'Occipital_Inf_L', 'Calcarine_L',
        'Cuneus_L', 'Lingual_L', 'Fusiform_L', 'Heschl_L', 'Temporal_Sup_L',
        'Temporal_Mid_L', 'Temporal_Inf_L', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Mid_L', 'ParaHippocampal_L',
        'Cingulum_Ant_L', 'Cingulum_Mid_L', 'Cingulum_Post_L', 'Insula_L', 'Rectus_R',
        'Olfactory_R', 'Frontal_Sup_Orb_R', 'Frontal_Med_Orb_R', 'Frontal_Mid_Orb_R', 'Frontal_Inf_Orb_R',
        'Frontal_Sup_R', 'Frontal_Mid_R', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_R', 'Frontal_Sup_Medial_R',
        'Supp_Motor_Area_R', 'Paracentral_Lobule_R', 'Precentral_R', 'Rolandic_Oper_R', 'Postcentral_R',
        'Parietal_Sup_R', 'Parietal_Inf_R', 'SupraMarginal_R', 'Angular_R', 'Precuneus_R',
        'Occipital_Sup_R', 'Occipital_Mid_R', 'Occipital_Inf_R', 'Calcarine_R', 'Cuneus_R',
        'Lingual_R', 'Fusiform_R', 'Heschl_R', 'Temporal_Sup_R', 'Temporal_Mid_R',
        'Temporal_Inf_R', 'Temporal_Pole_Sup_R', 'Temporal_Pole_Mid_R', 'ParaHippocampal_R', 'Cingulum_Ant_R',
        'Cingulum_Mid_R', 'Cingulum_Post_R', 'Insula_R', 'Hippocampus_L', 'Hippocampus_R',
        'Amygdala_L', 'Amygdala_R', 'Caudate_L', 'Caudate_R', 'Putamen_L',
        'Putamen_R', 'Pallidum_L', 'Pallidum_R', 'Thalamus_L', 'Thalamus_R',
    ]
    # Create a dictionary mapping region names to their indices in my_aal_regions
    #my_region_to_index = {region: idx for idx, region in enumerate(my_aal_regions)}

    # Debugging: Print the dictionary to verify
    #print("Region to Index Mapping:", my_region_to_index)

    # Load the AAL atlas labels
    aal_labels = aal_atlas['labels']

    # Create a dictionary mapping AAL labels to their corresponding indices in aal_atlas['indices']
    aal_label_to_code = {label: int(idx) for label, idx in zip(aal_labels, aal_atlas['indices'])}

    # Debugging: Print the dictionary to verify
    #print("AAL Label to Code Mapping (as integers):", aal_label_to_code)

    # Get the 3D data from the AAL atlas
    aal_data = nib.load(aal_atlas.maps).get_fdata()

    # Select a row from microstate_maps
    microstate_data = np.abs(microstate_maps[selected_row, :])
    #microstate_data = microstate_maps[selected_row, :]
    if th_percentile is not None:
        threshold_value = np.percentile(microstate_data, th_percentile)
    else:
        threshold_value = None

    #print(f"Selected microstate data (row {selected_row}):", microstate_data)

    # Map the reordered data to the AAL atlas
    mapped_aal_data = np.zeros_like(aal_data)
    for i, value in enumerate(microstate_data):
        # Skip regions with no data
        roi_name = my_aal_regions[i]
        region_mask = (aal_data == aal_label_to_code[roi_name])  # AAL regions are 1-indexed
        mapped_aal_data[region_mask] = value

    # Save the mapped AAL atlas as a new NIfTI image
    mapped_aal_nii = nib.Nifti1Image(mapped_aal_data, affine=nib.load(aal_atlas.maps).affine)
    print(f'min and max of maps {selected_row}: {np.min(microstate_data)}, {np.max(microstate_data)}')
    # Visualize the mapped AAL atlas using plot_img_on_surf
    plot_img_on_surf(
        stat_map=mapped_aal_nii,  # Use the mapped AAL atlas volume as the stat map
        #surf_mesh='fsaverage',
        views=["lateral", "medial"],
        hemispheres=["left", "right"],
        title=f"Microstate Map {selected_row}",
        bg_on_data=True,
        symmetric_cmap=None,
        vmin=np.min(microstate_data), vmax=np.max(microstate_data),
        colorbar=True,  # Add a colorbar to verify the color mapping
        cmap='viridis',
        cbar_tick_format="",
        threshold=threshold_value,  # Disable threshold to show all values
        darkness=None
    )
    plt.gcf().suptitle("", fontsize=24)  
    #plt.show()

    return mapped_aal_nii


def plot_brains(file_maps, output_dir):
   
    os.makedirs(output_dir, exist_ok=True)
    # Path to the .npz file
    file_path = file_maps
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)
    # Extract the microstate_maps matrix
    if "Microstates" in data:
        microstate_maps = data["Microstates"]
        print(f"Loaded microstate_maps with shape: {microstate_maps.shape}")
    else:
        raise KeyError("The key 'Microstates' was not found in the .npz file.")

    # Example usage
    aal_atlas = datasets.fetch_atlas_aal()


    for i in range(microstate_maps.shape[0]):   
        print(f"Processing microstate map {i}...")
        _ = map_microstate_to_aal(microstate_maps, i, aal_atlas, th_percentile=None)

        # Save the plot as an image with higher resolution
        output_image_path = os.path.join(output_dir, f"microstate_map_{i:02d}.png")
        plt.savefig(output_image_path, dpi=600)  # Set DPI to 300 for better resolution
        print(f"Saved plot for microstate map {i} to {output_image_path}")
