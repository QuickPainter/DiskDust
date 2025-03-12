import os
import sys
import time
import importlib
import numpy as np
import scipy.constants as sc
import emcee
from multiprocessing import Pool
from mcmc_tools import *
import corner
import matplotlib.pyplot as plt

# style setups (always deployed)
_ = importlib.import_module('plot_setups')
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])


# target
targ = 'MWC480'

# model type and priors information
mtype = 'low'
pri_type = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform']
pri_pars = [[0, 10], [0, 1], [-3, 3], [0, 5], [-10, 5]]

mtype = 'trans'
pri_type = ['uniform', 'uniform', 'uniform', 'uniform', 'uniform', 
            'normal', 'uniform', 'uniform']
pri_pars = [[0, 10], [0, 1], [-3, 3], [0, 5], [0, 5], [180, 50], [0, 1], 
            [-10, 5]]

# MCMC parameters
append = False
nsteps, ninit = 100000, 200
nwalk, nthread = 64, 8
maxtau = 1500
burnfactor = 10
thinfactor = 0.5
cutfactor = 50

# data location
specdir = 'final_spec/'

# posteriors subdirectory
postdir = 'spec_posteriors/'


""" ======================================================================= """

# prior evaluators
def uniform_prior(par, p):
    if np.logical_and((par >= p[0]), (par <= p[1])):
        return 0
    else:
        return -np.inf

def normal_prior(par, p):
    return -0.5 * ((par - p[0]) / p[1])**2


if mtype == 'low':
    # dimensionality
    ndim = 5

    # notation
    plbls = ['Snu0', 'X', 'alpc', 'alpd', 'logf']
    punits = ['mJy', '', '', '', '']

    # spectrum model
    def spec_model(pars, nu):
        f = (1. - pars[1]) * pars[0] * (nu / 33)**pars[3] + \
            pars[1] * pars[0] * (nu / 33)**pars[2]
        return f

elif mtype == 'trans':
    # dimensionality
    ndim = 8

    # notation
    plbls = ['Snu0', 'X', 'alpc', 'alpd_lo', 'alpd_hi', 'nu_t', 'scl', 'logf']
    punits = ['mJy', '', '', '', '', 'GHz', '', '']

    # spectrum model
    def spec_model(pars, nu):
        alp_dust = pars[3] - (pars[3] - pars[4]) / \
                   (1 + np.exp(-pars[6] * (nu - pars[5])))
        f = (1. - pars[1]) * pars[0] * (nu / 33)**alp_dust + \
            pars[1] * pars[0] * (nu / 33)**pars[2]
        return f

else:
    print('I do not know this model.  Exiting.')
    sys.exit()



""" Probability functions """
# log-prior
def log_prior(pars):
    lnT = 0
    for ii in range(len(pars)):
        cmd = pri_type[ii]+'_prior(pars['+str(ii)+'], '+str(pri_pars[ii])+')'
        lnT += eval(cmd)
    return lnT

# log-likelihood
def log_likelihood(pars, nu, Snu, eSnu):
    model_spec = spec_model(pars, nu)
    if mtype == 'low':
        var = eSnu**2 + model_spec**2 * np.exp(2 * pars[4])
    elif mtype == 'trans':
        var = eSnu**2 + model_spec**2 * np.exp(2 * pars[7])
    return -0.5 * np.sum((Snu - model_spec)**2 / var + np.log(var))

# log-posterior
def log_posterior(pars, nu, Snu, eSnu):
    if np.isfinite(log_prior(pars)):
        return log_likelihood(pars, nu, Snu, eSnu) + log_prior(pars)
    else:
        return -np.inf


""" Inference """
# caution with internal multithreading
if (nthread > 1): os.environ["OMP_NUM_THREADS"] = "1"

# Load the visibility data
nu_, Snu_, eSnu_, grp_ = np.loadtxt(specdir+targ+'.spectrum.txt').T
ok = nu_ <= 970
nu = nu_[ok]
Snu = Snu_[ok]
eSnu = eSnu_[ok]
grp = grp_[ok]


# Assign the output filename prefix
outfile = targ+'.'+mtype

# Initialize the walkers, starting from the previous run
if append:
    if os.path.exists(postdir+outfile+'.post.npz'):
        pre_samples = np.load(postdir+outfile+'.post.npz')['samples']
        pre_logpost = np.load(postdir+outfile+'.post.npz')['logpost']
        p00 = pre_samples[-1,:,:]
    else:
        print('I cannot find the file to append samples.  Exiting')
        sys.exit()
# Initialize the walkers, starting from random posterior draws
else:
    p0 = np.empty((nwalk, ndim))
    for ip in range(ndim):
        _ = 'np.random.'+pri_type[ip]+'('+str(pri_pars[ip][0])+', '
        _ += str(pri_pars[ip][1])+', '+str(nwalk)+')'
        p0[:,ip] = eval(_)

    # Quick initial run to mitigate stray walkers
    with Pool(processes=nthread) as pool:
        isampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior,
                                         pool=pool, args=(nu, Snu, eSnu))
        isampler.run_mcmc(p0, ninit, progress=False)
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
    p00 = np.reshape(p00, p0.shape)

# Full MCMC run
with Pool(processes=nthread) as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_posterior,
                                    pool=pool, args=(nu, Snu, eSnu))
    sampler.run_mcmc(p00, nsteps, progress=True)
samples = sampler.get_chain()
logpost = sampler.get_log_prob()
if append:
    samples = np.concatenate((pre_samples, samples))
    logpost = np.concatenate((pre_logpost, logpost))

samples_ = mcmc_out(samples, logpost, maxtau=maxtau, cutfactor=cutfactor,
                    burnfactor=burnfactor, thinfactor=thinfactor)

# Save the outputs
postfile = postdir+outfile+'.post.npz'
print('Posterior samples were saved to '+postfile)
np.savez(postfile, samples=samples, chain=samples_, logpost=logpost)



""" Diagnostics """
### plot the walker traces
# identify outlier walkers (based on lnprob)
nstep, nwalk, ndim = samples.shape
ncut = round(nstep / cutfactor)
dev_ = (np.median(logpost[ncut:,:], axis=0) - \
        np.median(logpost[ncut:,:])) / np.std(logpost[ncut:,:])
out_ix = np.where(np.abs(dev_) >= 2)
_samples  = np.delete(samples, out_ix, axis=1)
_logposts = np.delete(logpost, out_ix, axis=1)
_nwalk = _samples.shape[1]

fig, ax = plt.subplots(nrows=ndim+1, ncols=1, figsize=(5., 8.),
                       constrained_layout=True, sharex=True)
blob = np.dstack((_samples, np.reshape(_logposts, (nstep, _nwalk, 1))))
steps = np.arange(nstep)
for ip in range(ndim+1):
    for iw in range(_nwalk):
        ax[ip].plot(steps, blob[:,iw,ip], '-k', alpha=0.05)
    if ip < ndim:
        ax[ip].set_ylabel(plbls[ip])
    else:
        ax[ip].set_ylabel('log(prob)')
fig.savefig('spec_figs/'+outfile+'.traces.png')
fig.clf()


print(_samples.shape, samples_.shape)
### plot the pairwise covariances
fig = corner.corner(samples_,
                    levels=(1-np.exp(-0.5*(np.array([1, 2, 3]))**2)),
                    labels=plbls)
plt.savefig('spec_figs/'+outfile+'.corner.png')
fig.clf()


### print simple marginalized posterior summaries
clevs = [15.85, 50., 84.15]
CI = np.percentile(samples_, clevs, axis=0)
print(' ')
for j in range(len(plbls)):
    print('%s = %.3f +%.3f / -%.3f %s' % \
          (plbls[j], CI[1,j], CI[2,j]-CI[1,j], CI[1,j]-CI[0,j], punits[j]))
print(' ')


### spectrum + posterior draws
fig, ax = plt.subplots(nrows=2, figsize=(7., 10.), constrained_layout=True,
                       gridspec_kw={'height_ratios':[2, 1], 'hspace':0.10})

# data
ax[0].errorbar(nu_, Snu_, yerr=eSnu_, fmt='o', ms=4, color='k')

# load and plot historical SED
wl_h, Snu_h, eSnu_h, sys_eSnu_h = np.loadtxt('data/'+targ+'.SED.txt',
                                             usecols=(0,1,2,3)).T
nu_h = (sc.c / (1e-6 * wl_h)) / 1e9
eSnu_h = np.sqrt(eSnu_h**2 + (sys_eSnu_h * Snu_h)**2)
ax[0].errorbar(nu_h, 1e3 * Snu_h, yerr=1e3 * eSnu_h, fmt='o', ms=4, 
            color='xkcd:gray', zorder=0, alpha=0.7)

ax[1].axhline(0, linestyle='dashed', color='xkcd:coral pink')

ndraws = 500
mnu = np.linspace(0, 1000, 2048)
rix = np.random.randint(0, samples_.shape[0], ndraws)
for j in range(ndraws):
    mdraw = spec_model(samples_[rix[j],:], mnu)
    ax[0].plot(mnu, mdraw, '-', color='xkcd:coral pink', alpha=0.02, 
               rasterized=True)

    ddraw = spec_model(samples_[rix[j],:], nu_)
    ax[1].plot(nu_, (Snu_ - ddraw) / eSnu_, 'o', ms=4, color='k', alpha=0.03,
               rasterized=True)

ax[0].set_xlim([1, 1000])
ax[0].set_xscale('log')
ax[0].set_ylim([0.005, 10000])
ax[0].set_yscale('log')
ax[0].set_xlabel('$\\nu$  (GHz)')
ax[0].set_ylabel('$S_\\nu$  (mJy)')

ax[1].set_xlim([1, 1000])
ax[1].set_xscale('log')
ax[1].set_ylim([-10, 10])
ax[1].set_xlabel('$\\nu$  (GHz)')
ax[1].set_ylabel('residual SNR')

plt.savefig('spec_figs/'+outfile+'.spectrum.png')
fig.clf()
