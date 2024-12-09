import os, sys, time
import numpy as np
import emcee
from multiprocessing import Pool


### Point-source Visibility Model
def ptsrc_model(pars, u, v):

    # model visibilities at phase center (constant!)
    mvis = pars[0] * np.ones_like(u) + 1j*np.zeros_like(u)

    # phase shift to treat offsets
    dx, dy = -pars[1] * np.pi / (180 * 3600), -pars[2] * np.pi / (180 * 3600)
    phase_shift = np.exp(-2 * np.pi * 1.0j * (u * dx + v * dy))
    mvis *= phase_shift

    return mvis


### Point-source Visibility Model
def gaussian_model(pars, u, v):

    # model visibilities at phase center (constant!)
    thet = np.radians(pars[5])
    uu = (u * np.cos(thet) + v * np.sin(thet)) * pars[3] 
    vv = (-u * np.sin(thet) + v * np.cos(thet)) * pars[4]
    uuvv = (uu**2 + vv**2) / (180 * 3600 / np.pi)**2
    mvis = pars[0] * np.exp(-2 * np.pi**2 * uuvv) + 1j * np.zeros_like(uuvv)

    # phase shift to treat offsets
    dx, dy = -pars[1] * np.pi / (180 * 3600), -pars[2] * np.pi / (180 * 3600)
    phase_shift = np.exp(-2 * np.pi * 1.0j * (u * dx + v * dy))
    mvis *= phase_shift

    return mvis


### (Log-)Likelihood Function for point-source model
def lnL_point(pars, u, v, vis, wgt):

    # model visibilities at phase center (constant!)
    model_vis = ptsrc_model(pars, u, v)

    # return the log-likelihood
    return -0.5 * np.sum(wgt * np.absolute(vis - model_vis)**2)


### (Log-)Likelihood Function for Gaussian model
def lnL_gaussian(pars, u, v, vis, wgt):

    # model visibilities at phase center (constant!)
    model_vis = gaussian_model(pars, u, v)

    # return the log-likelihood
    return -0.5 * np.sum(wgt * np.absolute(vis - model_vis)**2)


### (Log-)Prior Function for point-source model
def lnT_point(pars):

    # uniform in flux density (lies between 0 and 2 Jy)
    if np.logical_or((pars[0] < 0), (pars[0] > 2)):
        return -np.inf

    # Gaussian priors for offsets (0.0 +/- 0.2 arcsec from phase center)
    lnTx = -0.5 * ((pars[1] - 0.0) / 0.2)**2
    lnTy = -0.5 * ((pars[2] - 0.0) / 0.2)**2

    # return the log-prior
    return lnTx + lnTy


### (Log-)Prior Function for gaussian model
def lnT_gaussian(pars):

    # uniform in flux density (lies between 0 and 2 Jy)
    if np.logical_or((pars[0] < 0), (pars[0] > 2)):
        return -np.inf

    # Gaussian in sizes, with peak at zero (prefer point-like emission)
    if np.logical_or((pars[3] <= 0), (pars[4] <= 0)):
        return -np.inf
    else:
        lnTa = -0.5 * ((pars[3] - 0.0) / 0.2)**2
        lnTb = -0.5 * ((pars[4] - 0.0) / 0.2)**2

    # uniform in position angle (between 0 and 180)
    if np.logical_or((pars[5] < 0), (pars[5] > 180)):
        return -np.inf

    # Gaussian priors for offsets (0.0 +/- 0.2 arcsec from phase center)
    lnTx = -0.5 * ((pars[1] - 0.0) / 0.2)**2
    lnTy = -0.5 * ((pars[2] - 0.0) / 0.2)**2

    # return the log-prior
    return lnTx + lnTy + lnTa + lnTb



### (Log-)Posterior Function for point-source model
def lnP_point(pars, u, v, vis, wgt):

    # check prior
    if not np.isfinite(lnT_point(pars)):
        return -np.inf

    # return log-posterior
    return lnL_point(pars, u, v, vis, wgt) + lnT_point(pars)


### (Log-)Posterior Function for gaussian model
def lnP_gaussian(pars, u, v, vis, wgt):

    # check prior
    if not np.isfinite(lnT_gaussian(pars)):
        return -np.inf

    # return log-posterior
    return lnL_gaussian(pars, u, v, vis, wgt) + lnT_gaussian(pars)


### Inference Wrapper
def fit_vis(data, outfile=None, nwalk=64, nthread=6, nsteps=10000, 
                  ninit=200, maxtau=None, return_full=False, mtype='point'):

    # careful with internal numpy multithreading 
    if (nthread > 1): os.environ["OMP_NUM_THREADS"] = "1"

    # initialization
    if mtype == 'point':
        ndim = 3
        p0 = np.array([0.1, 0, 0]) + 0.1 * np.random.randn(nwalk, ndim)
    elif mtype == 'gaussian':
        ndim = 6
        p0 = np.array([0.1, 0, 0, 0.2, 0.2, 20]) + \
             0.1 * np.random.randn(nwalk, ndim)

    # early sampling
    if mtype == 'point':
        with Pool(processes=nthread) as pool:
            isampler = emcee.EnsembleSampler(nwalk, ndim, lnP_point, 
                                             pool=pool, args=data) 
            isampler.run_mcmc(p0, ninit)
    elif mtype == 'gaussian':
        with Pool(processes=nthread) as pool:
            isampler = emcee.EnsembleSampler(nwalk, ndim, lnP_gaussian,
                                             pool=pool, args=data)
            isampler.run_mcmc(p0, ninit)

    # reset initialization to mitigate stray walkers
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]

    # MCMC sampling
    if mtype == 'point':
        with Pool(processes=nthread) as pool:
            sampler = emcee.EnsembleSampler(nwalk, ndim, lnP_point, 
                                            pool=pool, args=data)
            sampler.run_mcmc(p00, nsteps, progress=True)
    elif mtype == 'gaussian':
        with Pool(processes=nthread) as pool:
            sampler = emcee.EnsembleSampler(nwalk, ndim, lnP_gaussian, 
                                            pool=pool, args=data)
            sampler.run_mcmc(p00, nsteps, progress=True)

    # compute the autocorrelation times
    tau_ = sampler.get_autocorr_time(quiet=True)
    if maxtau is not None:
        tau = np.concatenate((tau_, np.array([maxtau])))
    else:
        tau = 1. * tau_

    # assign burn, thin
    print('tau',tau.max())
    try:
        burn = int(round(10 * tau.max(), -2))
        thin = int(np.round(tau.max()))
    except:
        burn = 1000
        thin = 100

    # full set of posterior samples (nstep, nwalk, ndim)
    samples = sampler.get_chain()

    # burned, thinned, flattened set of posterior samples (npost, ndim)
    samples_ = sampler.get_chain(discard=burn, thin=thin, flat=True)

    # save the posteriors if requested
    if outfile is not None:
        np.savez(outfile, samples=samples, chain=samples_, tau=tau, 
                 burn=burn, thin=thin)

    # return the posterior samples and autocorrelation times
    if return_full:
        return samples, tau
    else:
        return samples_, tau 
