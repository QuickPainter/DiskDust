import os
import sys
import numpy as np
from deproject_vis import deproject_vis
from fit_tools import *
import matplotlib.pyplot as plt

targ = 'MWC480'
vfile = '_K.contp3'
offs = [-0.02, -0.70]
incl, PA = 36.5, 147.5

# location of visibilities (whatever you like)
visdir = '/Users/calebpainter/Downloads/Research/THESIS/data/vis_stuff/pool/asha0/SCIENCE/VLA_SEDs/DR/VLA/MWC480/iter_vis/'

# (u, v) distance bin width for visibility profile plots (klambda)
duv = 0.5

### Load the data
_ = np.load(visdir+targ+vfile+'.vis.npz')
u, v, vis, wgt = _['u'], _['v'], _['Vis'], _['Wgt']

# random draws on visibility profile
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5.5, 7.5),
                       constrained_layout=True,
                       gridspec_kw={'height_ratios':[1, 1], 'hspace':0.10})
uvdist = np.sqrt(u**2 + v**2) / 1e3
uvbins = np.arange(1, 125, duv)

class Visibility:
    def __init__(self, vis, u, v, wgt):
        self.vis = vis
        self.u = u
        self.v = v
        self.wgt = wgt

vp = deproject_vis(Visibility(vis, u, v, wgt), uvbins, incl=incl, PA=PA,
                   offx=offs[0], offy=offs[1])

ax[0].axhline(y=0, linestyle=':', color='darkslategray')
ax[0].errorbar(1e-3 * vp.rho_uv, 1e6 * vp.vis_prof.real,
               yerr=1e6 * vp.err_std.real, fmt='o', color='k', zorder=0,
               alpha=0.7, rasterized=True)
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-500, 3500])
ax[0].set_ylabel(r'real visibilities ($\\mu$Jy)')
ax[0].set_xlabel(r'deprojected baseline length (k$\lambda$)')

ax[1].axhline(y=0, linestyle=':', color='darkslategray')
ax[1].errorbar(1e-3 * vp.rho_uv, 1e6 * vp.vis_prof.imag,
               yerr=1e6 * vp.err_std.imag, fmt='o', color='k', zorder=0,
               alpha=0.7, rasterized=True)
ax[1].set_xlim([0, 100])
ax[1].set_ylim([-1500, 1500])
ax[1].set_ylabel(r'imag visibilities ($\\mu$Jy)')
ax[1].set_xlabel(r'deprojected baseline length (k$\lambda$)')

fig.savefig('figs/'+targ+vfile+'.visprof.png')
fig.clf()
