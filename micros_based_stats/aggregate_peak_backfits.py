import os
import pandas as pd
import numpy as np
from scipy.stats import zscore, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import AnovaRM


def compute_microstate_occurrences(npz_file, n_microstates):
    """
    Compute the occurrences of each microstate class for each session, ensuring all microstates
    (from 0 to n_microstates-1) are included, even if they have 0 occurrences.

    Parameters:
    ----------
    npz_file : str
        Path to the .npz file containing the Sequence and SessionLimits.
    n_microstates : int
        Total number of microstates (e.g., 3, 5, 10).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns: 'Session', 'Microstate', 'Occurrences'.
    """
    # Load the .npz file
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"File not found: {npz_file}")
    
    data = np.load(npz_file, allow_pickle=True)
    sequence = data["Sequence"]  # Microstate associations
    session_limits = data["SessionLimits"].item()  # Session limits as a dictionary

    # Initialize a list to store the results
    results = []

    # Loop through each session
    for session, (start, end) in session_limits.items():
        # Extract the sequence for the current session
        session_sequence = sequence[start:end + 1]

        # Count occurrences of each microstate class
        unique, counts = np.unique(session_sequence, return_counts=True)
        session_counts = dict(zip(unique, counts))

        # Ensure all microstates (0 to n_microstates-1) are included, even if they have 0 occurrences
        for microstate in range(n_microstates):
            occurrences = session_counts.get(microstate, 0)  # Default to 0 if the microstate is missing
            results.append({
                "Session": session,
                "Microstate": microstate,
                "Occurrences": occurrences
            })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def compute_repeated_measures_anova(df, output_file=None):
    """
    Perform repeated measures ANOVA for amount_zscore across Phases for each State.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing 'amount_zscore', 'Phase', 'State', and 'sub'.
    output_file : str, optional
        Path to save the ANOVA results. If None, results are not saved.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns: 'State', 'F', 'p-value'.
    """
    # Ensure 'Phase', 'State', and 'sub' are treated as categorical variables
    df['Session'] = df['Session'].astype('category')
    df['Microstate'] = df['Microstate'].astype('category')
    df['sub'] = df['sub'].astype('category')

    # Open a file to save results if output_file is provided
    if output_file:
        anova_file = open(output_file, "w")
        anova_file.write("Repeated Measures ANOVA Results:\n\n")

    # List to store results for each state
    results = []

    # Perform repeated measures ANOVA for each State
    for state in df['Microstate'].unique():
        print(f"\nPerforming repeated measures ANOVA for State {state}...")
        if output_file:
            anova_file.write(f"Microstate{state}:\n")

        # Filter data for the current state
        state_data = df[df['Microstate'] == state]

        # Perform repeated measures ANOVA
        aov = AnovaRM(state_data, depvar='Occurrences_zscore', subject='sub', within=['Session'])
        res = aov.fit()

        # Perform post-hoc analysis using Tukey's HSD
        posthoc = pairwise_tukeyhsd(state_data['Occurrences_zscore'], state_data['Session'])
        print(posthoc)

        if output_file:
            anova_file.write("\nPost-hoc Analysis (Tukey's HSD):\n")
            anova_file.write(posthoc.summary().as_text())
            anova_file.write("\n\n")

        # Extract F-value and p-value
        f_value = res.anova_table.iloc[0, 0] # F value for 'Session'
        p_value = res.anova_table.iloc[0, 3]

        # Append the results to the list
        results.append({'State': state, 'F': f_value, 'p-value': p_value})

        # Print and save the results
        print(res)
        if output_file:
            anova_file.write(res.summary().as_text())
            anova_file.write("\n\n")

    # Close the file if it was opened
    if output_file:
        anova_file.close()
        print(f"ANOVA results saved to {output_file}")

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def aggregate_microstate_occurrences(backfit_dir, n_microstates):
    """
    Compute and aggregate microstate occurrences for all backfit files in a directory,
    keeping track of the subject in the 'sub' column.

    Parameters:
    ----------
    backfit_dir : str
        Path to the directory containing the backfitted microstate files.
    n_microstates : int
        Total number of microstates (e.g., 3, 5, 10).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with columns: 'sub', 'Session', 'Microstate', 'Occurrences'.
    """
    # Initialize a list to store the aggregated results
    aggregated_results = []

    # Loop through all .npz files in the backfit directory
    for file_name in os.listdir(backfit_dir):
        if file_name.endswith(".npz"):
            # Construct the full file path
            file_path = os.path.join(backfit_dir, file_name)

            # Extract the subject ID from the file name (e.g., "sub-01" from "backfitted_labels_sub-01.npz")
            subject = int(file_name.split('_')[-1].split('.')[0].split('-')[-1])  # Extract "01"

            # Compute occurrences for the current file
            occurrences_df = compute_microstate_occurrences(file_path, n_microstates)

            # Add the subject column to the DataFrame
            occurrences_df['sub'] = subject

            # Append the results to the aggregated list
            aggregated_results.append(occurrences_df)

    # Concatenate all results into a single DataFrame
    aggregated_df = pd.concat(aggregated_results, ignore_index=True)

    return aggregated_df

def process_microstate_results_Fstat(base_folder):
    """
    Process microstate results for a given base directory.
    Extract occurrences, compute z-scores, perform repeated measures ANOVA,
    and aggregate results across different numbers of microstates.

    Parameters:
    ----------
    base_folder : str
        The base directory containing the backfitted microstates folders.
    """
    # Initialize a list to store the aggregated ANOVA results
    all_anova_results = []

    # Loop through all *_backfitted_microstates folders
    for folder_name in os.listdir(base_folder):
        if folder_name.endswith("_backfitted_microstates"):
            # Define the path to the current backfitted microstates folder
            backfit_dir = os.path.join(base_folder, folder_name)

            # Extract the number of microstates from the folder name (e.g., "3" from "3_backfitted_microstates")
            n_microstates = int(folder_name.split("_")[0])

            print(f"Processing folder: {folder_name} (n_microstates = {n_microstates})")

            # Aggregate microstate occurrences
            aggregated_df = aggregate_microstate_occurrences(backfit_dir, n_microstates)

            # Compute z-scores for the 'Occurrences' column grouped by 'sub' and 'Session'
            aggregated_df['Occurrences_zscore'] = aggregated_df.groupby(['sub'])['Occurrences'].transform(lambda x: zscore(x, ddof=0))

            # Save the aggregated DataFrame to a CSV file
            output_csv = os.path.join(backfit_dir, 'combined', "aggregated_microstate_occurrences.csv")
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            aggregated_df.to_csv(output_csv, index=False)
            print(f"Aggregated occurrences saved to {output_csv}")

            # Perform repeated measures ANOVA
            anova_output_file = os.path.join(backfit_dir, 'combined', "repeated_measures_anova_results.txt")
            anova_results_df = compute_repeated_measures_anova(aggregated_df, output_file=anova_output_file)

            # Add the number of microstates to the ANOVA results
            anova_results_df['N_Microstate'] = n_microstates

            # Append the results to the aggregated list
            all_anova_results.append(anova_results_df)

            # Save the ANOVA results to a CSV file
            anova_results_csv = os.path.join(backfit_dir, 'combined', "anova_results.csv")
            anova_results_df.to_csv(anova_results_csv, index=False)
            print(f"ANOVA results saved to {anova_results_csv}")

    # Combine all ANOVA results into a single DataFrame
    final_anova_results_df = pd.concat(all_anova_results, ignore_index=True)

    # Save the final aggregated ANOVA results to a CSV file in the base directory
    final_output_csv = os.path.join(base_folder, "final_anova_results.csv")
    final_anova_results_df.to_csv(final_output_csv, index=False)
    print(f"Final aggregated ANOVA results saved to {final_output_csv}")

def process_microstate_results_ttest(base_folder, case_ctrl):
    """
    Process microstate results for a given base directory.
    Extract occurrences, perform t-tests for two independent groups,
    and aggregate results across different numbers of microstates.

    Parameters:
    ----------
    base_folder : str
        The base directory containing the backfitted microstates folders.
    case_ctrl : str
        File .csv containing the two groups for t-tests
    """
    # Read the case-control participant file
    participants_df = pd.read_csv(case_ctrl, sep='\t')
    # Transform the 'sub' column in participants_df to match the format in aggregated_df
    participants_df['sub'] = participants_df['participant_id'].apply(lambda x: int(x.split('-')[-1]))

    # Initialize a list to store the aggregated t-test results
    all_ttest_results = []

    # Loop through all *_backfitted_microstates folders
    for folder_name in os.listdir(base_folder):
        if folder_name.endswith("_backfitted_microstates"):
            # Define the path to the current backfitted microstates folder
            backfit_dir = os.path.join(base_folder, folder_name)

            # Extract the number of microstates from the folder name (e.g., "3" from "3_backfitted_microstates")
            n_microstates = int(folder_name.split("_")[0])

            print(f"Processing folder: {folder_name} (n_microstates = {n_microstates})")

            # Aggregate microstate occurrences
            aggregated_df = aggregate_microstate_occurrences(backfit_dir, n_microstates)

           # Save the aggregated DataFrame to a CSV file
            output_csv = os.path.join(backfit_dir, 'combined', "aggregated_microstate_occurrences.csv")
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            aggregated_df.to_csv(output_csv, index=False)
            print(f"Aggregated occurrences saved to {output_csv}")

            # Perform t-tests for each microstate
            for microstate in aggregated_df['Microstate'].unique():
                print(f"Performing t-test for Microstate {microstate}...")

                # Filter data for the current microstate
                microstate_data = aggregated_df[aggregated_df['Microstate'] == microstate]

                # Merge with participants_df to get group information
                microstate_data = microstate_data.merge(participants_df, left_on='sub', right_on='sub')

                # Split data into two groups based on 'case_ctrl'
                group1 = microstate_data[microstate_data['case_ctrl'] == 0]['Occurrences']
                group2 = microstate_data[microstate_data['case_ctrl'] == 1]['Occurrences']

                # Perform t-test
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

                # Append the results to the list
                all_ttest_results.append({
                    'Microstate': microstate,
                    'T-Statistic': t_stat,
                    'p-value': p_value,
                    'N_Microstate': n_microstates
                })

    # Combine all t-test results into a single DataFrame
    final_ttest_results_df = pd.DataFrame(all_ttest_results)

    # Save the final aggregated t-test results to a CSV file in the base directory
    final_output_csv = os.path.join(base_folder, "final_t_results.csv")
    final_ttest_results_df.to_csv(final_output_csv, index=False)
    print(f"Final aggregated t-test results saved to {final_output_csv}")


