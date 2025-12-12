from matplotlib import pyplot as plt
import os

def plot_gfp(gfp, gfp_peaks, subject, session, outdir):
    """
    Plot the Global Field Power (GFP) and its peaks, and save the plot as an image.

    Parameters:
    -----------
    gfp : array-like
        The Global Field Power (GFP) values to be plotted over time.
    gfp_peaks : array-like
        Indices of the GFP peaks to be highlighted on the plot.
    subject : str or int
        Identifier for the subject being analyzed.
    session : str or int
        Identifier for the session being analyzed.
    outdir : str
        Directory where the resulting plot image will be saved.

    Saves:
    -------
    A PNG image of the GFP plot with peaks highlighted, named as 
    'sub-{subject}_ses-{session}_gfp.png' in the specified output directory.
    """
    plt.figure(figsize=(15, 4))
    plt.plot(gfp, label='GFP')
    plt.scatter(gfp_peaks, gfp[gfp_peaks], color='red', s=10, label='GFP Peaks')
    plt.title(f'Subject {subject} - Global Field Power')
    plt.xlabel('Time (samples)')
    plt.ylabel('GFP (a.u.)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'sub-{subject}_ses-{session}_gfp.png'))
    plt.close()