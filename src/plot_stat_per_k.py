import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from statsmodels.stats.multitest import multipletests  # Import for FDR correction


# Apply FDR correction per N_Microstate
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

def plot_microstate_Fstatistics(base_folder, alpha=0.05):
    """
    Generate and save plots for microstate F-statistics, significant states, and GEV.

    Parameters:
    ----------
    base_folder : str
        Path to the main folder containing the analysis results, including the "final_anova_results.csv" file.
    alpha : float, optional
        Significance level for statistical tests (default is 0.05).

    Description:
    ------------
    This function performs the following tasks:
    1. Loads the "final_anova_results.csv" file from the specified base folder.
    2. Computes the maximum F value for each number of microstates (N_Microstate).
    3. Applies Bonferroni correction and FDR correction to p-values to determine significant states.
    4. Counts the number of significant states after FDR correction for each N_Microstate.
    5. Extracts Global Explained Variance (GEV) from *_microstates_segmentation folders.
    6. Generates and saves the following plots:
       - Maximum F values across different numbers of microstates.
       - Number of significant states after FDR correction.
       - GEV as a function of N_Microstate.
       - Scatter plot of significant states for each N_Microstate after FDR correction.

    Outputs:
    --------
    The function saves the following plots in the base folder:
    - "Max_F_Values_Plot.png": Line plot of maximum F values.
    - "Significant_States_FDR_Plot.png": Bar plot of the number of significant states after FDR correction.
    - "GEV_Plot.png": Line plot of GEV across N_Microstate.
    - "Significant_States_Scatter_Plot.png": Scatter plot of significant states after FDR correction.

    Notes:
    ------
    - The function assumes that the "final_anova_results.csv" file and *_microstates_segmentation folders
      are present in the specified base folder.
    - The GEV data is extracted from "microstates_all_subjects_results.npz" files within *_microstates_segmentation folders.
    - The function uses Bonferroni and FDR corrections to adjust p-values for multiple comparisons.
    """

    # Load the final_anova_results.csv file
    file_path = os.path.join(base_folder, "final_anova_results.csv")
    df = pd.read_csv(file_path)

    # Compute the maximum F value for each N_Microstate
    summary_df = df.groupby("N_Microstate").agg(
        max_F=("F", "max")
    ).reset_index()
    # Apply Bonferroni correction and count significant states
    df["Bonferroni_p"] = alpha / df["N_Microstate"]  # Adjust p-values using Bonferroni correction
    df["Significant_Bonferroni"] = df['p-value'] < df["Bonferroni_p"]  # Determine significance after correction

    df = df.groupby("N_Microstate").apply(apply_fdr_correction).reset_index(drop=True)  # Reset index to avoid ambiguity

    # Count the number of significant states for each N_Microstate
    significance_summary = df.groupby("N_Microstate")["Significant_FDR"].sum().reset_index()
    significance_summary.rename(columns={"Significant_FDR": "Num_Significant_States_FDR"}, inplace=True)

    # Extract GEV from *_microstates_segmentation folders
    gev_results = []
    for folder_name in os.listdir(base_folder):
        if folder_name.endswith("_microstates_segmentation"):
            n_microstates = int(folder_name.split("_")[0])  # Extract the number of microstates
            npz_file = os.path.join(base_folder, folder_name, "microstates_all_subjects_results.npz")
            if os.path.exists(npz_file):
                data = np.load(npz_file, allow_pickle=True)
                gev = data.get("GEV", None)  # Extract the GEV variable
                if gev is not None:
                    gev_results.append({"N_Microstate": n_microstates, "GEV": gev})

    # Create a DataFrame for GEV results
    gev_df = pd.DataFrame(gev_results).sort_values("N_Microstate")

    # Subplot 1: Maximum F values
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["N_Microstate"], summary_df["max_F"], marker="o", label="Max F", color="blue")
    plt.ylabel("F Value", fontsize=12)
    plt.xlabel("Number of k Microstates", fontsize=12)
    plt.title("Maximum F-Values across different number of microstates", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    output_path_max_f = os.path.join(base_folder, "Max_F_Values_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_max_f, dpi=300)
    print(f"Max F values plot saved to {output_path_max_f}")
    #plt.show()

    # Subplot 2: Number of significant states (FDR correction)
    plt.figure(figsize=(10, 6))
    plt.bar(significance_summary["N_Microstate"], significance_summary["Num_Significant_States_FDR"], color="purple", alpha=0.7)
    plt.ylabel("Number of Significant States (FDR)", fontsize=12)
    plt.title("Number of Significant States After FDR Correction", fontsize=14)
    plt.grid(alpha=0.3)
    output_path_significant_states = os.path.join(base_folder, "Significant_States_FDR_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_significant_states, dpi=300)
    print(f"Significant states plot saved to {output_path_significant_states}")
    #plt.show()

    # Subplot 3: GEV as a function of N_Microstate
    plt.figure(figsize=(10, 6))
    plt.plot(gev_df["N_Microstate"], gev_df["GEV"], marker="d", label="GEV", color="red")
    plt.ylabel("GEV", fontsize=12)
    plt.title("Global Explained Variance (GEV) Across N_Microstate", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    output_path_gev = os.path.join(base_folder, "GEV_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_gev, dpi=300)
    print(f"GEV plot saved to {output_path_gev}")
    #plt.show()

    # Subplot 4: Significant states for each N_Microstate (FDR correction)
    plt.figure(figsize=(10, 6))
    for n_microstate in df["N_Microstate"].unique():
        significant_states = df[(df["N_Microstate"] == n_microstate) & (df["Significant_FDR"])]
        plt.scatter(
            [n_microstate] * len(significant_states),
            significant_states["State"],
            label=f"N_Microstate={n_microstate}" if len(significant_states) > 0 else "",
            alpha=0.7
        )
    plt.xlabel("Number of Microstates (N_Microstate)", fontsize=12)
    plt.ylabel("State", fontsize=12)
    plt.title("Significant States After FDR Correction", fontsize=14)
    plt.grid(alpha=0.3)
    output_path_significant_states_scatter = os.path.join(base_folder, "Significant_States_Scatter_Plot.png")
    plt.tight_layout()
    plt.savefig(output_path_significant_states_scatter, dpi=300)
    print(f"Significant states scatter plot saved to {output_path_significant_states_scatter}")
    #plt.show()

def plot_microstate_Tstatistics(base_folder, alpha=0.05):
    """
    Generate and save plots for microstate T-statistics, significant states, and GEV.

    Parameters:
    ----------
    base_folder : str
        Path to the main folder containing the analysis results, including the "final_t_results.csv" file.
    alpha : float, optional
        Significance level for statistical tests (default is 0.05).

    Description:
    ------------
    This function performs the following tasks:
    1. Loads the "final_t_results.csv" file from the specified base folder.
    2. Computes the maximum T value for each number of microstates (N_Microstate).
    3. Applies Bonferroni correction and FDR correction to p-values to determine significant states.
    4. Counts the number of significant states after FDR correction for each N_Microstate.
    5. Extracts Global Explained Variance (GEV) from *_microstates_segmentation folders.
    6. Generates and saves the following plots:
       - Maximum T values across different numbers of microstates.
       - Number of significant states after FDR correction.
       - GEV as a function of N_Microstate.
       - Scatter plot of significant states for each N_Microstate after FDR correction.

    Outputs:
    --------
    The function saves the following plots in the base folder:
    - "Max_T_Values_Plot.png": Line plot of maximum T values.
    - "Significant_States_FDR_Plot_T.png": Bar plot of the number of significant states after FDR correction.
    - "GEV_Plot_T.png": Line plot of GEV across N_Microstate.
    - "Significant_States_Scatter_Plot_T.png": Scatter plot of significant states after FDR correction.

    Notes:
    ------
    - The function assumes that the "final_t_results.csv" file and *_microstates_segmentation folders
      are present in the specified base folder.
    - The GEV data is extracted from "microstates_all_subjects_results.npz" files within *_microstates_segmentation folders.
    - The function uses Bonferroni and FDR corrections to adjust p-values for multiple comparisons.
    """

    # Load the final_t_results.csv file
    file_path = os.path.join(base_folder, "final_t_results.csv")
    df = pd.read_csv(file_path)

    # Compute the maximum absolute T value for each N_Microstate, but save the original value
    summary_df = df.loc[df.groupby("N_Microstate")['T-Statistic'].apply(lambda x: abs(x).idxmax())]
    summary_df = summary_df[['N_Microstate', 'T-Statistic']].rename(columns={'T-Statistic': 'max_T'})

    # Apply Bonferroni correction and count significant states
    df["Bonferroni_p"] = alpha / df["N_Microstate"]  # Adjust p-values using Bonferroni correction
    df["Significant_Bonferroni"] = df['p-value'] < df["Bonferroni_p"]  # Determine significance after correction

    df = df.groupby("N_Microstate").apply(apply_fdr_correction).reset_index(drop=True)  # Reset index to avoid ambiguity

    # Count the number of significant states for each N_Microstate
    significance_summary = df.groupby("N_Microstate")["Significant_FDR"].sum().reset_index()
    significance_summary.rename(columns={"Significant_FDR": "Num_Significant_States_FDR"}, inplace=True)

    # Extract GEV from *_microstates_segmentation folders
    gev_results = []
    for folder_name in os.listdir(base_folder):
        if folder_name.endswith("_microstates_segmentation"):
            n_microstates = int(folder_name.split("_")[0])  # Extract the number of microstates
            npz_file = os.path.join(base_folder, folder_name, "microstates_all_subjects_results.npz")
            if os.path.exists(npz_file):
                data = np.load(npz_file, allow_pickle=True)
                gev = data.get("GEV", None)  # Extract the GEV variable
                if gev is not None:
                    gev_results.append({"N_Microstate": n_microstates, "GEV": gev})

    # Create a DataFrame for GEV results
    gev_df = pd.DataFrame(gev_results).sort_values("N_Microstate")

    # Subplot 1: Maximum T values
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["N_Microstate"], summary_df["max_T"], marker="o", label="Max T", color="blue")
    plt.ylabel("T Value", fontsize=12)
    plt.xlabel("Number of k Microstates", fontsize=12)
    plt.title("Maximum T-Values across different number of microstates", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

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

    # Subplot 3: GEV as a function of N_Microstate
    plt.figure(figsize=(10, 6))
    plt.plot(gev_df["N_Microstate"], gev_df["GEV"], marker="d", label="GEV", color="red")
    plt.ylabel("GEV", fontsize=12)
    plt.title("Global Explained Variance (GEV) Across N_Microstate", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    output_path_gev = os.path.join(base_folder, "GEV_Plot_T.png")
    plt.tight_layout()
    plt.savefig(output_path_gev, dpi=300)
    print(f"GEV plot saved to {output_path_gev}")

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