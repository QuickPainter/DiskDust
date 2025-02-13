
import numpy as np
import astropy 
from astropy import constants as c
from astropy import units as u
import scipy.integrate as integrate
import matplotlib.pyplot as plt 
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
import sys

# load data

## load data

# define data directories

def get_data_version(file_type, archive,file_number,non_vla_freq_index_cutoff, lower_freq_cutoff=0, disk='MWC480'):
    dir_name = '/Users/calebpainter/Downloads/Research/THESIS/data'
    image_files = os.listdir(dir_name + '/image')

    nu_, Snu_, eSnu_, grp_ = np.loadtxt(dir_name+f'/{disk}.spectrum.txt').T

    # non-VLA archival
    non_vla = ascii.read(dir_name+'/MWC480.spec.txt')  
    lambda_non_vla, snu_non_vla, snu_err_non_vla1, snu_err_non_vla2  = non_vla['col1'], non_vla['col2'], non_vla['col3'], non_vla['col4']

    snu_err_non_vla = np.sqrt(snu_err_non_vla1**2 + (snu_non_vla * snu_err_non_vla2)**2)

    nu_non_vla = np.array([(constants.c/(units.micron * x)).to(units.GHz).value for x in lambda_non_vla])

    # set what data you want (vis vs imge, include archival or no, which image data you want, when you want vla_freq cutoff to be)

    # record as the data files
    data_version = {
        'File Type:': file_type,
        'Non-VLA Archive and Cutoff': (archive, non_vla_freq_index_cutoff),
        'File Number (if multiple)': file_number
    }


    if file_type == 'img':
        image_np= np.load(dir_name+f'/image/{image_files[file_number]}')
        nu_img, Snu_img, err_Snu_img = image_np['nu'], image_np['Snu'], image_np['eSnu']

    
        if archive == True:

            nu_hz_combined_img = np.array(list(nu_img*(10**9)) + list(nu_non_vla*(10**9))[non_vla_freq_index_cutoff:])
            snu_uJy_combined_img = np.array(list(Snu_img*(10**6)) + list(snu_non_vla*(10**6))[non_vla_freq_index_cutoff:])
            snu_err_uJy_combined_img = np.array(list(err_Snu_img*(10**6)) + list(snu_err_non_vla*(10**6))[non_vla_freq_index_cutoff:])

            mcmc_nu = nu_hz_combined_img
            mcmc_Snu = snu_uJy_combined_img
            mcmc_Snu_err= snu_err_uJy_combined_img

        elif archive == False:
            mcmc_nu = nu_img
            mcmc_Snu = Snu_img
            mcmc_Snu_err= err_Snu_img

    elif file_type == 'vis':
        # visibility-based spectrum
        _ = np.load(dir_name+'/MWC480.vis_fluxes.npz')
        nu_vis, Snu_vis, err_Snu_vis = _['nu'], _['Fnu'], _['eFnu']

        if archive == True:
            
            nu_hz_combined_vis = np.array(list(nu_vis) + list(nu_non_vla*(10**9))[non_vla_freq_index_cutoff:])
            snu_uJy_combined_vis = np.array(list(Snu_vis) + list(snu_non_vla*(10**6))[non_vla_freq_index_cutoff:])
            snu_err_uJy_combined_vis = np.array(list(err_Snu_vis) + list(snu_err_non_vla*(10**6))[non_vla_freq_index_cutoff:])

            mcmc_nu = nu_hz_combined_vis
            mcmc_Snu = snu_uJy_combined_vis
            mcmc_Snu_err= snu_err_uJy_combined_vis

        elif archive == False:
            mcmc_nu = nu_vis
            mcmc_Snu = Snu_vis
            mcmc_Snu_err= err_Snu_vis

    ## take final values
    if True:
        mcmc_nu = nu_ 
        mcmc_Snu = Snu_ 
        mcmc_Snu_err = eSnu_



        if archive == True:

            mcmc_nu = np.array(list(mcmc_nu) + list(nu_non_vla)[:non_vla_freq_index_cutoff])
            mcmc_Snu = np.array(list(mcmc_Snu) + list(snu_non_vla*(10**3))[:non_vla_freq_index_cutoff])
            mcmc_Snu_err = np.array(list(mcmc_Snu_err) + list(snu_err_non_vla*(10**3))[:non_vla_freq_index_cutoff])


            ## convert to microjansky
            mcmc_Snu = mcmc_Snu * 10**3
            mcmc_Snu_err = mcmc_Snu_err * 10**3

    else:
        ## convert all frequencies to GHz
        mcmc_nu = mcmc_nu/(10**9)


    # add calibration errors as list
    flux_scale_errors = np.zeros(len(mcmc_nu))

    indices_17Ghz = np.where(mcmc_nu < 17)[0] 
    indices_50Ghz = np.where((mcmc_nu < 50) &(mcmc_nu > 17) )[0] 
    indices_100Ghz = np.where(mcmc_nu > 50)[0] 

    flux_scale_errors[indices_17Ghz] = .03
    flux_scale_errors[indices_50Ghz] = .05
    flux_scale_errors[indices_100Ghz] = .1

    noema = True
    if noema == False:
        ## add in NOEMA data
        image_np= np.load(dir_name+'/MWC480_NOEMA.imf_fluxes.npz')
        nu_noema, Snu_noema, err_Snu_noema = image_np['nu'], image_np['Snu'], image_np['eSnu']

        NOEMA_flux_scale_errors = np.zeros(len(nu_noema))

        indices_lower = np.where(nu_noema < 110)[0] 
        indices_upper = np.where(nu_noema > 110)[0] 

        NOEMA_flux_scale_errors[indices_lower] = .08
        NOEMA_flux_scale_errors[indices_upper] = .1

        mcmc_nu = np.array(list(mcmc_nu) + list(nu_noema))
        mcmc_Snu = np.array(list(mcmc_Snu) + list(Snu_noema*(10**6)))
        mcmc_Snu_err = np.array(list(mcmc_Snu_err) + list(err_Snu_noema*(10**6)))
        flux_scale_errors = np.array(list(flux_scale_errors) + list(NOEMA_flux_scale_errors))
    


    flux_scale_errors_scaled = flux_scale_errors * mcmc_Snu



    # sort by frequencies
    sorted_indices = np.argsort(mcmc_nu)

    mcmc_nu = mcmc_nu[sorted_indices]
    mcmc_Snu = mcmc_Snu[sorted_indices]
    mcmc_Snu_err = mcmc_Snu_err[sorted_indices]
    flux_scale_errors = flux_scale_errors[sorted_indices]
    flux_scale_errors_scaled = flux_scale_errors_scaled[sorted_indices]

    try:
        lower_index_cutoff = np.where(mcmc_nu < lower_freq_cutoff)[-1][-1]
    except:
        lower_index_cutoff = 0

    print(lower_index_cutoff)
    mcmc_nu = mcmc_nu[lower_index_cutoff:]
    mcmc_Snu = mcmc_Snu[lower_index_cutoff:]
    mcmc_Snu_err = mcmc_Snu_err[lower_index_cutoff:]
    flux_scale_errors = flux_scale_errors[lower_index_cutoff:]
    flux_scale_errors_scaled = flux_scale_errors_scaled[lower_index_cutoff:]

    return data_version, mcmc_nu, mcmc_Snu, mcmc_Snu_err, flux_scale_errors, flux_scale_errors_scaled



def logistic_model(v):
    chi, tot_flux, alpha_cont, alpha_dust_ceiling, alpha_dust_floor, alpha_dust_slope, alpha_dust_freq_center, log_f = .18, 2, .5, 3, 2.5, -.05, 11, -2

    alpha_dust = alpha_dust_floor + (alpha_dust_ceiling - alpha_dust_floor) / (1 + np.exp(-alpha_dust_slope * (v - alpha_dust_freq_center)))

    # normalize to Ghz
    model = (1-chi)*tot_flux*(v/33)**alpha_dust + (chi)*tot_flux*(v/33)**alpha_cont

    return model, alpha_dust


## THERMAL MODEL CODE


# Constants
pi = np.pi

# Function to calculate B_v(T) (Planck function approximation)
def B_v(T, v):

    h = 6.626e-34  # Planck's constant (Joule-second)
    k = 1.381e-23  # Boltzmann constant (Joule/Kelvin)
    c = 3e8  # Speed of light (m/s) 
    

    return (2 * h * v**3 / c**2) / (np.exp(h * v / (k * T)) - 1)

# Function to calculate tau_v
def tau_v(v, v_0, beta, Sigma):
    return (v / v_0)**beta * Sigma

# Function to calculate the integrand B_v(T) * (1 - exp(-tau_v))
def integrand(r, T0, v, v_0, beta, Sigma_function):

    ## in grams / cm^2
    Sigma_r = Sigma_function(r)

    ## dimensionless
    tau = tau_v(v, v_0, beta, Sigma_r)

    # in kelvin
    temp = T0 * (r)**(-3/7)

    Bv  =  B_v(temp, v)

    Bv_scaled_to_au = Bv * (1/6.68459e-12)**2

    return Bv_scaled_to_au * (1 - np.exp(-tau)) * 2 * pi * r

# Define Sigma(r) as a simple function of r (e.g., power law or exponential decay)
def Sigma_function(r, tau0, A1=3, sigma1=15, A2=.2, r_bump=100, sigma2=8):
    gaussian1 = A1 * np.exp(-r**2 / (2 * sigma1**2))
    gaussian2 = A2 * np.exp(-(r - r_bump)**2 / (2 * sigma2**2))
    baseline = 10**(-2)

    full = gaussian1 + gaussian2 + baseline

    return full * tau0

# Main function to compute S_v
def S_v(i, d, R, T0, v, v_0, beta, tau0, Sigma_function, num_points=1000):

    sigma_func = lambda r: Sigma_function(r, tau0)
    integrand_func = lambda r: integrand(r, T0, v, v_0, beta, sigma_func)
    
    # Use quad for integration over r from 0 to R
    integral, error = integrate.quad(integrand_func, 0, R)
    
    cos_i = np.cos(i)
    return (cos_i / d**2) * integral




def model_wrapper(T=80, beta=1.4, Tau0= 1, vs = np.linspace(10, 5*999, int((1000-1)/0.99))):

    # convert to hz
    vs = vs * 10**9

    i = np.radians(45)  # 45 degrees inclination
    d = 450 * 9.461e+15  # Distance in meters
    L_star = 22 # in solar lums
    M_star = 2 # in solar mass


    R = 100  # Radius limit in arbitrary units

    T0 = T * (L_star/0.28)**(2/7) / (M_star/0.8)**(1/7)  # Temperature in Kelvin

    v_0 = 33*10**9  # Reference frequency in Hz

    # vs in Hz
    Sv = []

    
    for v in vs:
    
        # Compute S_v
        S_v_value = S_v(i, d, R, T0, v, v_0, beta, Tau0, Sigma_function)

        ## convert to microjansky
        S_v_value = S_v_value / 10**(-32)


        Sv.append(S_v_value)
    
    return np.array(Sv), vs


## check outputs for different parameters
def plot_model(ax, params, vs = np.linspace(10, 5*999, int((1000-1)/0.99))):

    Tau0, beta, T, logf = params

    
    Sv, new_vs = model_wrapper(T=T, beta= beta, Tau0 = Tau0, vs=vs)

    ax.plot(vs, Sv)

    text = (
        r"$\Sigma_0$= " + f'{np.round(Tau0,2)}\n' + 
        r'$\beta=$' + f'{np.round(beta,2)}\n' + 
        r'$T_0$=' + f'{np.round(T,2)}\n' )

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)  # Customize the appearance
    ax.text(1.04, .8, text, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(alpha=.5,ls='--')


## we want to fit the data with maybe two power laws to start, one from contamination and one from true emission

# first we need to decide on parameters

def log_likelihood_two_component(single_contam_theta, v, Sv, Sv_err):

    ## chi = fraction of emission from contamination from brehmstralung
    chi, tot_flux, alpha_dust, alpha_cont, log_f = single_contam_theta 

    # normalize to Ghz
    model = (1-chi)*tot_flux*(v)**alpha_dust + (chi)*tot_flux*(v)**alpha_cont

    sigma2 = Sv_err**2 + model**2 * np.exp(2*log_f)
    # log_model = np.log(model)

    log_likelihood = -0.5 * np.sum((Sv - model) ** 2 / sigma2 + np.log(sigma2))

    return log_likelihood

# Define the prior: flat priors in this example
def log_prior_two_component(single_contam_theta):
    chi, tot_flux, alpha_dust, alpha_cont, log_f = single_contam_theta
    if 0 < chi <= 1 and 0 < tot_flux and 2 < alpha_dust < 6 and -3 < alpha_cont < 3:
        return 0.0  # log(1) = 0 for flat prior
    return -np.inf  # log(0) = -inf for values outside the bounds

# Define the posterior probability function (log-posterior)
def log_posterior_two_component(single_contam_theta, x, y, sigma):
    lp = log_prior_two_component(single_contam_theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_two_component(single_contam_theta, x, y, sigma)


## logistic function model  
def log_likelihood_logistic(logistic_params, v, Sv, Sv_err):

    chi, tot_flux, alpha_cont, alpha_dust_ceiling, alpha_dust_floor, alpha_dust_slope, alpha_dust_freq_center, log_f = logistic_params

    alpha_dust = alpha_dust_floor + (alpha_dust_ceiling - alpha_dust_floor) / (1 + np.exp(-alpha_dust_slope * (v - alpha_dust_freq_center)))

    # normalize to Ghz
    model = (1-chi)*tot_flux*(v/33)**alpha_dust + (chi)*tot_flux*(v/33)**alpha_cont

    sigma2 = Sv_err**2 + model**2 * np.exp(2*log_f)
    # log_model = np.log(model)

    log_likelihood = -0.5 * np.sum((Sv - model) ** 2 / sigma2 + np.log(sigma2))

    return log_likelihood

# Define the prior: flat priors in this example
def log_prior_logistic(logistic_params):
    chi, tot_flux, alpha_cont, alpha_dust_ceiling, alpha_dust_floor, alpha_dust_slope, alpha_dust_freq_center, log_f = logistic_params
    if 0 < chi <= 1 and 0 < tot_flux and 2 < alpha_dust_ceiling < 5 and 0 < alpha_cont < 2 and 0< alpha_dust_floor < alpha_dust_ceiling and 0 > alpha_dust_slope > -0.1 and 0 < alpha_dust_freq_center < 400:
        # Gaussian prior for alpha_dust_freq_center
        mu = 180
        sigma = 50
        gaussian_prior = -0.5 * ((alpha_dust_freq_center - mu) / sigma) ** 2
        
        return gaussian_prior  # Add log prior term
    
    return -np.inf  # Outside the bounds
# Define the posterior probability function (log-posterior)
def log_posterior_logistic(logistic_params, x, y, sigma):
    lp = log_prior_logistic(logistic_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_logistic(logistic_params, x, y, sigma)

def log_likelihood_three_power_law(single_contam_theta, v, Sv, Sv_err):

    ## chi = fraction of emission from contamination from brehmstralung
    f1, f2, alpha1, alpha2, alpha3, switch, log_f = single_contam_theta 

    f3 = f2 * switch**(alpha2-alpha3)

    # normalize to Ghz
    baseline = f1 * v ** alpha1
    dust = np.where(v < switch, f2 * v ** alpha2, f3 * v ** alpha3)
    model = baseline + dust
    
    sigma2 = Sv_err**2 + model**2 * np.exp(2*log_f)
    # log_model = np.log(model)

    log_likelihood = -0.5 * np.sum((Sv - model) ** 2 / sigma2 + np.log(sigma2))

    return log_likelihood

# Define the prior: flat priors in this example
def log_prior_three_power_law(single_contam_theta):
    f1, f2, alpha1, alpha2, alpha3, switch, log_f = single_contam_theta
    if 0 < f1 < 3000 and 0 < f2 <3000 and -4 < alpha1 < 2 and 2 < alpha2 < 4 and 0 < alpha3 < 4 and 3 < switch < 10:
        return 0.0  # log(1) = 0 for flat prior
    return -np.inf  # log(0) = -inf for values outside the bounds

# Define the posterior probability function (log-posterior)
def log_posterior_three_power_law(single_contam_theta, x, y, sigma):
    lp = log_prior_three_power_law(single_contam_theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_three_power_law(single_contam_theta, x, y, sigma)


def log_likelihood_three_component(single_contam_theta, v, log_Sv, log_Sv_err):

    ## chi = fraction of emission from contamination from brehmstralung
    Fdust, Fbrehm, Fspin, alpha_dust, alpha_brehm, spin_center = single_contam_theta 

    # normalize to Ghz
    model = Fdust*(v)**alpha_dust + Fbrehm*(v)**alpha_brehm + Fspin*(v/spin_center)**2*np.exp((1-(v/spin_center)**2 ))

    log_likelihood = -0.5 * np.sum(((log_Sv - model) / log_Sv_err) ** 2 + np.log(2 * np.pi * log_Sv_err**2))

    return log_likelihood

# Define the prior: flat priors in this example
def log_prior_three_component(single_contam_theta):
    Fdust, Fbrehm, Fspin, alpha_dust, alpha_brehm, spin_center = single_contam_theta
    if  1 < Fdust and 1 < Fbrehm and 1 < Fspin <100  and 0 < alpha_dust < 6 and -10 < alpha_brehm < 10 and 0 < spin_center < 1:
        return 0.0  # log(1) = 0 for flat prior
    return -np.inf  # log(0) = -inf for values outside the bounds

# Define the posterior probability function (log-posterior)
def log_posterior_three_component(single_contam_theta, x, y, sigma):
    lp = log_prior_three_component(single_contam_theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_three_component(single_contam_theta, x, y, sigma)




## with thermal model

def log_likelihood_thermal(thermal_params, v, Sv, Sv_err):

    cont_flux, alpha_cont, Tau0, beta0, T0, log_f = thermal_params

    dust_termal_model, model_vs = model_wrapper(T=T0,beta=beta0, Tau0 = Tau0, vs = v)

    model = dust_termal_model + cont_flux*(v/33)**alpha_cont

    sigma2 = Sv_err**2 + model**2 * np.exp(2*log_f)

    log_likelihood = -0.5 * np.sum((Sv - model) ** 2 / sigma2 + np.log(sigma2))

    return log_likelihood

# Define the prior: flat priors in this example
def log_prior_thermal(thermal_params):
    cont_flux, alpha_cont, Tau0, beta0, T0, log_f = thermal_params 
    if 0 < cont_flux and 0 < alpha_cont < 2 and 0 < Tau0 < 2 and 0 < beta0  < 2 and 10 < T0 < 200 and -50 < log_f < 50:
        # Gaussian prior for some parameters
        mu = 400
        sigma = 100
        gaussian_prior_cont_flux = -0.5 * ((cont_flux - mu) / sigma) ** 2

        mu = .6
        sigma = .3
        gaussian_prior_alpha_cont = -0.5 * ((alpha_cont - mu) / sigma) ** 2

        mu = .2
        sigma = .05
        gaussian_prior_tau0 = -0.5 * ((Tau0 - mu) / sigma) ** 2

        mu = 1.4
        sigma = .2
        gaussian_prior_beta = -0.5 * ((beta0 - mu) / sigma) ** 2

        mu = 60
        sigma = 20
        gaussian_prior_T0 = -0.5 * ((T0 - mu) / sigma) ** 2

        return gaussian_prior_cont_flux + gaussian_prior_alpha_cont + gaussian_prior_tau0 + gaussian_prior_beta + gaussian_prior_T0

    return -np.inf  # Outside the bounds


# Define the posterior probability function (log-posterior)
def log_posterior_thermal_model(thermal_params, x, y, sigma):
    lp = log_prior_thermal(thermal_params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_thermal(thermal_params, x, y, sigma)


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

    chains_fig, axes = plt.subplots(ndim, figsize=(10, 20), sharex=True)
    samples = sampler.get_chain()
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.axvline(x=.15*len(samples[:, :, i]),color='red',ls='--',alpha=.6)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    return chains_fig

## set up MCMC sampler
def run_mcmc(ndim, nu, Snu, Snu_err,nsteps=100000,walkers_scaling=20,scaling=33, burn_fraction=.15,plot=False, disk='MWC80', model_type='default'):

    # Set up the MCMC sampler
    nwalkers = ndim*walkers_scaling  # number of walkers

    # Initialize walkers close to plausible values
    if ndim == 5:
        labels = ['chi','Ftot','alpha dust','alpha cont','log_f']
        initial_guess = np.array([.5, 1000, 2,1,100])  # initial guesses for two component model

    if ndim ==7:
        labels = [r'$S_v^{cont}$',r'$S_v^{dust1}$',r'$\alpha_{cont}$',r'$\alpha_{dust1}$',r'$\alpha_{dust2}$',r'$v_{\rm switch}$','log_f']
        initial_guess = np.array([1000,300,1,3,2,7, 10]) # Fdust, Fbrehm, Fspin, alpha_dust, alpha_brehm, spin_center GHz

    if ndim == 8:
       labels = ['chi', 'tot_flux', 'alpha_cont', 'alpha_dust_ceiling', 'alpha_dust_floor', 'alpha_dust_slope', 'alpha_dust_freq_center', 'log_f']
       initial_guess = np.array([.5, 2, 1, 3.5, 2.5, -.01, 180, -2])

    if ndim == 6:
        print("Running a Thermal Model")
        labels = ['cont_flux', 'alpha_cont', 'Tau0', 'beta0', 'T0', 'log_f']
        initial_guess = np.array([400, .6, 0.2, 1.4, 60, -2])


    initial_position = initial_guess + 0.1 * np.random.randn(nwalkers, ndim) * initial_guess

    # normalize Nu Vis to Ghz
    # norm_nu_vis = nu / (scaling)
    norm_nu_vis = nu

    # Set up the emcee sampler
    if ndim == 5:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_two_component, args=(norm_nu_vis, Snu, Snu_err),a=2.5)
    if ndim == 7:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_three_power_law, args=(norm_nu_vis, Snu, Snu_err))
    if ndim == 8:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_logistic, args=(norm_nu_vis, Snu, Snu_err))
    if ndim == 6:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_thermal_model, args=(norm_nu_vis, Snu, Snu_err))

    # Run the MCMC
    sampler.run_mcmc(initial_position, int(nsteps/2), progress=True)
    # tau = sampler.get_autocorr_time()

    samples = sampler.chain

    # Step 1: Calculate the overall mean of each parameter across all walkers and steps
    overall_mean = np.mean(samples, axis=(0, 1))  # Shape should be (n_dim,)

    # Step 2: Calculate the mean for each walker (average over n_steps for each walker)
    walker_means = np.mean(samples, axis=1)  # Shape should be (n_walkers, n_dim)

    # Step 3: Calculate the distance of each walker mean from the overall mean
    deviations = np.linalg.norm(walker_means - overall_mean, axis=1)

    print('deviations', deviations)
    
    # Step 4: Identify outlier walkers (e.g., threshold as 2 standard deviations from the mean deviation)
    threshold = 1 * np.std(deviations)

    outliers = deviations > threshold

    # Step 5: Filter out outlier walkers and set up new initial positions
    filtered_walkers = walker_means[~outliers]  # Keeps only non-outlier walkers

    # Step 6: Restart the emcee sampler with the filtered walkers as the new starting positions
    nwalkers_filtered, ndim = filtered_walkers.shape
    

    if ndim == 5:
        sampler = emcee.EnsembleSampler(nwalkers_filtered, ndim, log_posterior_two_component, args=(norm_nu_vis, Snu, Snu_err),a=2.5)
    if ndim == 7:
        sampler = emcee.EnsembleSampler(nwalkers_filtered, ndim, log_posterior_three_power_law, args=(norm_nu_vis, Snu, Snu_err),a=2.5)
    if ndim == 8:
        sampler = emcee.EnsembleSampler(nwalkers_filtered, ndim, log_posterior_logistic, args=(norm_nu_vis, Snu, Snu_err),a=2.5)
    if ndim == 6:
        sampler = emcee.EnsembleSampler(nwalkers_filtered, ndim, log_posterior_thermal_model, args=(norm_nu_vis, Snu, Snu_err))


    sampler.run_mcmc(filtered_walkers, nsteps, progress=True)

    samples = sampler.get_chain(thin=15, flat=True)


    params = []
    # burn 1500
    for i in range(ndim):
        params.append(np.median(samples[int(np.round(len(samples)*burn_fraction)):, i]))

    # chi_emcee = np.mean(chi_emcee_samples)
    # tot_flux_emcee = np.mean(tot_flux_emcee_samples)
    # alpha_dust_emcee = np.mean(alpha_dust_emcee_samples)
    # alpha_cont_emcee = np.mean(alpha_cont_emcee_samples)



    samples = samples[int(np.round(len(samples)*burn_fraction)):, :]


    if plot:

        chains_fig = plot_chains(sampler,labels,ndim)
        chains_fig.save_fig(f'{disk}_chains_fig_{model_type}_{nwalkers}_walkers.png')
        chains_fig.close()
    # Assuming `samples` is your MCMC chain samples (shape: nsteps, nwalkers, ndim)
        flat_samples = samples.reshape((-1, ndim))  # Flatten if using an MCMC sampler like emcee

        # remove outliers

        mean_samples = np.mean(flat_samples, axis=0)  # Shape: (ndim,)
        std_samples = np.std(flat_samples, axis=0)    # Shape: (ndim,)

        # Identify samples within 2σ of the mean (per dimension)
        within_2sigma = np.all(np.abs(flat_samples - mean_samples) <= 2 * std_samples, axis=1)

        # Filter out samples beyond 2σ
        filtered_samples = flat_samples[within_2sigma]


        corner_labels = ["$\\chi$", "$S_{cm}$", "$\\alpha^d$", "$\\alpha^c$"] if ndim == 4 else labels
        corner_fig = corner.corner(filtered_samples, labels=corner_labels, show_titles=True)

        corner_fig.save_fig(f'{disk}_corner_fig_{model_type}_{nwalkers}_walkers.png')
        corner_fig.close()

        return samples, sampler, params, norm_nu_vis, labels, chains_fig, corner_fig

    else:
        return samples, sampler, params, norm_nu_vis, labels
    

def main(ndim, disk):

    ndim = 6
    scaling = 33
    archival = False

    if ndim ==8:
        model_type = 'logistic'
    if ndim ==7:
        model_type = 'three power law'
    if ndim == 5:
        model_type = 'two power law'
    if ndim == 6:
        model_type = 'thermal'
        archival = True

    print(model_type)

    data_version, mcmc_nu, mcmc_Snu, mcmc_Snu_err, flux_scale_errors, flux_scale_errors_scaled = get_data_version('vis', archival, 0, 4, 0, disk=disk)

 
    samples, sampler, params, norm_nu_vis, labels, chains_fig, corner_fig = run_mcmc(ndim, mcmc_nu,mcmc_Snu,mcmc_Snu_err,1000,16,scaling, plot=True, disk=disk, model_type=model_type)

    

if __name__ == '__main__':
    ndim = sys.argv[1]
    disk = sys.argv[2]
    main(ndim)