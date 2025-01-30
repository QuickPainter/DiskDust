import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm as tqdm
from astropy.io import ascii
import emcee as emcee
from astropy import constants, units
import os
import corner
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import gaussian_kde


def check_convergence(sampler, threshold=50, discard=100):
    try:
        # Estimate the autocorrelation time
        autocorr_time = sampler.get_autocorr_time(tol=0)

        print(autocorr_time)
        
        # Check if the chain length is sufficiently larger than the autocorrelation time
        converged = sampler.chain.shape[1] > threshold * np.max(autocorr_time)

        return converged, autocorr_time
    except emcee.autocorr.AutocorrError:
        # AutocorrError is raised if chain length is too short to estimate autocorrelation time
        print("Chain is too short to reliably estimate autocorrelation time.")
        return False, None



## plotting MCMC chains
def plot_chains(sampler,labels,ndim):

    chains_fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axvline(x=.15*len(samples[:, :, i]),color='red',ls='--',alpha=.6)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.show()

    return chains_fig