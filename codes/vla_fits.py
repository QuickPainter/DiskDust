import os, sys, time
import numpy as np
from astropy.io import fits
from fit_tools import *
import corner
from deproject_vis import deproject_vis
import matplotlib.pyplot as plt


def main():

    ### User Controls

    # datafiles to fit
    visdir = '/Users/calebpainter/Downloads/Research/THESIS/data/vis_stuff/iter_vis/'
    vis_files = np.sort(os.listdir(visdir))
    target = 'MWC480'
    # location of visibilities
    # visdir = '/Users/calebpainter/Downloads/Research/THESIS/data/vis_stuff/pool/asha0/SCIENCE/VLA_SEDs/DR/VLA/MWC480/iter_vis/'

    # location of posteriors subdirectory
    postdir = '/Users/calebpainter/Downloads/Research/THESIS/codes/outputs/'
    plotdir = '/Users/calebpainter/Downloads/Research/THESIS/plots/vis/'

    # MCMC parameters
    nsteps, ninit = 10000, 500

    # model type
    mtype = 'gaussian'
    plbls = ['flux', 'dx', 'dy', 'sigmax', 'sigmay', 'theta']
    punits = ['uJy', 'arcsec', 'arcsec', 'arcsec', 'arcsec', 'degr']

    # mtype = 'point'
    # plbls = ['flux', 'dx', 'dy']
    # punits = ['uJy', 'arcsec', 'arcsec']

    incl, PA = 36.5, 147.5

    # (u, v) distance bin width for visibility profile plots (klambda)
    duv = 2

    #########################################################

    # start loop through fits
    for i in range(len(vis_files)):

        band_cal = '.'.join(vis_files[i].split('.')[-4:-2])
        print('Running on disk band/calibration', band_cal)
        ### Load the data
        _ = np.load(visdir+vis_files[i])
        u, v, vis, wgt = _['u'], _['v'], _['Vis'], _['Wgt']

        ### Perform Flux Inference
        # run the MCMC
        t0 = time.time()
        postfile = postdir+band_cal+'.'+mtype+'.post.npz'
        samples, tau = fit_vis((u, v, vis, wgt), outfile=postfile, nsteps=nsteps, 
                            ninit=ninit, nwalk=12, nthread=6, mtype=mtype)
        t1 = time.time()

        # progress tracker
        print('Inference stored at '+postdir+band_cal+\
            '.'+mtype+'.post.npz in %.1f s' % (t1 - t0))
        samples = np.load(postdir+band_cal+'.'+mtype+'.post.npz')['chain']

        ### Inference Metrics
        # make a simple corner plot
        samples[:,0] *= 1e6
        fig = corner.corner(samples, levels=(1-np.exp(-0.5*(np.array([1, 2, 3])))), 
                            labels=plbls)
        plt.savefig(plotdir+band_cal+'.'+mtype+'.corner.png')
        fig.clf()

        # simple estimates of marginalized posterior summaries
        clevs = [15.85, 50., 84.15]
        CI = np.percentile(samples, clevs, axis=0)
        lbls, uts = plbls, punits
        for j in range(len(lbls)):
            if j == 0:
                print('%s = %.1f +%.1f / -%.1f %s' % \
                    (lbls[j], CI[1,j], CI[2,j]-CI[1,j], CI[1,j]-CI[0,j], uts[j]))
            else:
                print('%s = %.2f +%.2f / -%.2f %s' % \
                    (lbls[j], CI[1,j], CI[2,j]-CI[1,j], CI[1,j]-CI[0,j], uts[j]))
        print(' ')

        # random draws on visibility profile
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 7.5), 
                            constrained_layout=True,
                            gridspec_kw={'height_ratios':[2, 1], 'hspace':0.10})
        uvdist = np.sqrt(u**2 + v**2) / 1e3
        uvbins = np.arange(1, 125, duv)

        class Visibility:
            def __init__(self, vis, u, v, wgt):
                self.vis = vis
                self.u = u
                self.v = v
                self.wgt = wgt

        vp = deproject_vis(Visibility(vis, u, v, wgt), uvbins, incl=0, PA=0,
                        offx=CI[1,1], offy=CI[1,2])

        ndraws = 500
        # drawing random values from the samples chain to plot many paths
        rix = np.random.randint(0, samples.shape[0], ndraws)
        for j in range(ndraws):
            if mtype == 'point':
                mdraw = ptsrc_model(samples[rix[j],:], u, v)
            elif mtype == 'gaussian':
                mdraw = gaussian_model(samples[rix[j],:], u, v)
            mvp = deproject_vis(Visibility(mdraw, u, v, wgt), uvbins, incl=0, PA=0,
                                offx=CI[1,1], offy=CI[1,2])
            ax[0].plot(1e-3*mvp.rho_uv, mvp.vis_prof.real, '-C1', alpha=0.03)
            ax[1].plot(1e-3*mvp.rho_uv, mvp.vis_prof.imag, '-C1', alpha=0.03)

        ax[0].axhline(y=0, linestyle=':', color='darkslategray')
        ax[0].errorbar(1e-3 * vp.rho_uv, 1e6 * vp.vis_prof.real, 
                    yerr=1e6 * vp.err_std.real, fmt='o', color='k', zorder=0)
        ax[0].set_xlim([0, 1.1*uvbins.max()])
        ax[0].set_ylim([-0.1*1e6*vp.vis_prof.real.max(), 
                        1.25*1e6*vp.vis_prof.real.max()])
        ax[0].set_ylabel('real visibilities (uJy)')
        ax[0].set_xlabel('deprojected baseline length (k_lambda)')

        ax[1].axhline(y=0, linestyle=':', color='darkslategray')
        ax[1].errorbar(1e-3 * vp.rho_uv, 1e6 * vp.vis_prof.imag, 
                    yerr=1e6 * vp.err_std.imag, fmt='o', color='k', zorder=0)
        ax[1].set_xlim([0, 1.1*uvbins.max()])
        ax[1].set_ylim([-1.25*1e6*vp.vis_prof.imag.max(),
                        1.25*1e6*vp.vis_prof.imag.max()])
        ax[1].set_ylabel('imag visibilities (uJy)')
        ax[1].set_xlabel('deprojected baseline length (k_lambda)')

        fig.savefig(plotdir+band_cal+'.'+mtype+'.visprof.png')
        fig.clf()

if __name__ == '__main__':
    main()