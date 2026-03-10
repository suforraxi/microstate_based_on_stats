from pycrostates.io import read_cluster
from micros_based_stats.examples.plot_brain_aal import map_microstate_to_aal
from nilearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import os

n_microstates = 15  # Change this to the desired number of microstates
cluster_f = f'/Volumes/CrucialX6/matteo/bids_als/derivatives/clustering_ordered/{n_microstates}_clustering.fif'
output_dir = '/Volumes/CrucialX6/matteo/bids_als/derivatives/brains_aal/'
clustering = read_cluster(cluster_f)

microstate_maps = clustering._cluster_centers_

aal_atlas = datasets.fetch_atlas_aal()


for i in range(microstate_maps.shape[0]):   
    print(f"Processing microstate map {i}...")
    _ = map_microstate_to_aal(microstate_maps, i, aal_atlas, th_percentile=None)

        # Save the plot as an image with higher resolution
    output_image_path = os.path.join(output_dir, str(n_microstates), f"microstate_map_{i:02d}.png")
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=600)  # Set DPI to 300 for better resolution
    print(f"Saved plot for microstate map {i} to {output_image_path}")