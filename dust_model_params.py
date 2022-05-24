import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from scipy.stats import truncnorm
from astropy.io import ascii
from copy import deepcopy
from duste.DustAttnCalc import DustAttnCalc, getTraceInfo, getMargSample
from scipy.interpolate import RegularGridInterpolator as RGIScipy

lsun = 3.846e33
pc = 3.085677581467192e18  # in cm

lightspeed = 2.998e18  # AA/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
jansky_mks = 1e-26
log_stellar_tot_ratio=-0.0969

#############
# DUST INFO
#############
d2n = DustAttnCalc(bv=1,eff=0)
traced2n, xtupd2n = d2n.getPostModelData()
ngrid, lwn, taugrid, lwt, _ = getTraceInfo(traced2n,bivar=True)
ngrid_med, taugrid_med = np.median(ngrid,axis=0), np.median(taugrid,axis=0)
wnmed, wtmed = np.exp(np.median(lwn)), np.exp(np.median(lwt))
d12 = DustAttnCalc(bv=0,eff=0)
d12.extratext, d12.indep_name = 'd1_d2', ['dust1']
traced12, xtupd12 = d12.getPostModelData()
d2grid, _, _, _, _ = getTraceInfo(traced12,bivar=False)
d2grid_med = np.median(d2grid,axis=0)

nint = RGIScipy(xtupd2n, ngrid_med, bounds_error=False, fill_value=None)
d2int = RGIScipy(xtupd2n, taugrid_med, bounds_error=False, fill_value=None)
d1int = RGIScipy(tuple([d2grid_med]), xtupd12[0], bounds_error=False, fill_value=None)

inc_sample = getMargSample(num=1000,bv=1,eff=0)[-1] # Need this for marginalizing over axis ratio

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'outfile': 'results/dust_pop_test',
              'nofork': True,
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_walks': 50, # MC walks
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 200, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'nbins_sfh': 7,
              'sigma': 0.3,
              'df': 2,
              'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'datdir': '',
              'runname': 'td_new',
              'objname':'AEGIS_13'
              }
# ------------------
# Observational Data
# ------------------

def build_obs(snr=10.0, filterset=["sdss_g0", "sdss_r0"],
              add_noise=True, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param snr:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock spectrum
    """
    from prospect.utils.obsutils import fix_obs

    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which bands (and wavelengths if doing
    # spectroscopy) in which to generate mock data.
    mock = {}
    mock['wavelength'] = None  # No spectrum
    mock['spectrum'] = None    # No spectrum
    mock['filters'] = observate.load_filters(filterset)

    # We need the models to make a mock
    sps = load_sps(**kwargs)
    mod = load_model(**kwargs)

    # Now we get the mock params from the kwargs dict
    params = {}
    for p in mod.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    mod.params.update(params)
    spec, phot, _ = mod.mean_model(mod.theta, mock, sps=sps)

    # Now store some ancillary, helpful info;
    # this information is not required to run a fit.
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = phot.copy()
    mock['mock_params'] = deepcopy(mod.params)
    mock['mock_snr'] = snr
    mock["phot_wave"] = np.array([f.wave_effective for f in mock["filters"]])

    # And store the photometry, adding noise if desired
    pnoise_sigma = phot / snr
    if add_noise:
        pnoise = np.random.normal(0, 1, len(phot)) * pnoise_sigma
        mock['maggies'] = phot + pnoise
    else:
        mock['maggies'] = phot.copy()
    mock['maggies_unc'] = pnoise_sigma
    mock['phot_mask'] = np.ones(len(phot), dtype=bool)

    # This ensures all required keys are present
    mock = fix_obs(mock)

    return mock, sps

##########################
# TRANSFORMATION FUNCTIONS
##########################
def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def massmet_to_logmass(massmet=None,**extras):
    return massmet[0]

def massmet_to_logzsol(massmet=None,**extras):
    return massmet[1]

def logmass_to_masses(massmet=None, logsfr_ratios=None, agebins=None, **extras):
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**massmet[0]) / coeffs.sum()
    return m1 * coeffs

def logmass_to_logsfr(massmet=None, logsfr_ratios=None, agebins=None, stellar_to_total=0.8, time_sfr=1.0e8, **extras):
    mass_in_bins = logmass_to_masses(massmet=massmet, logsfr_ratios=logsfr_ratios, agebins=agebins, **extras)
    return np.log10(mass_in_bins[:2].sum()*stellar_to_total/time_sfr)

def dustattn_to_dust2(dustattn=None, **extras):
    return dustattn[0]

def dustattn_to_n(dustattn=None, **extras):
    return dustattn[1]

def dustattn_to_dust1(dustattn=None, **extras):
    return dustattn[2]

#############
# MODEL_PARAMS
#############
model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'massmet', 'N': 2,
                        'isfree': True,
                        'init': np.array([10,-0.5]),
                        'prior': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': False,
                        'depends_on': massmet_to_logmass,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': None})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': -0.5,
                        'depends_on': massmet_to_logzsol,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': None})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

model_params.append({'name': 'mass', 'N': 1,
                     'isfree': False,
                     'depends_on': logmass_to_masses,
                     'init': 1.,
                     'units': r'M$_\odot$',})

model_params.append({'name': 'logsfr', 'N': 1,
                     'isfree': False,
                     'depends_on': logmass_to_logsfr,
                     'init': 1.,
                     'units': r'M$_\odot$/yr',})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior': None})

model_params.append({'name': 'logsfr_ratios', 'N': 6,
                        'isfree': True,
                        'init': [0]*6,
                        'units': '',
                        'prior': None})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'dustattn', 'N': 3,
                        'isfree': True, 'init': [1.0, 0.0, 1.1],
                        'prior': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'depends_on': dustattn_to_dust1,
                        'init': 1.1,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': False, 'init': 1.0, 
                        'depends_on': dustattn_to_dust2,
                        'init': 1.0, 'units': '', 'prior': None})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False, 'init': 0.0, 
                        'depends_on': dustattn_to_n,
                        'init': 0.0, 'units': '', 'prior': None})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': False,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'disp_floor': 0.15,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': False,
                        'init': 2.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=7.0)})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'nebemlineinspec', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'prior': None})

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1, # scale with sSFR?
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

##### AGN dust ##############
model_params.append({'name': 'add_agn_dust', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'fagn', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.03,
                        'disp_floor': 0.02,
                        'units': '',
                        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': True,
                        'init': 20.0,
                        'init_disp': 5,
                        'disp_floor': 2,
                        'units': '',
                        'prior': priors.LogUniform(mini=5.0, maxi=150.0)})

####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior': priors.TopHat(mini=0.0, maxi=0.5)})

####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

#### resort list of parameters for later display purposes
parnames = [m['name'] for m in model_params]
fit_order = ['massmet','logsfr_ratios', 'dustattn', 'fagn', 'agn_tau', 'gas_logz']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt('gallazzi_05_massmet.txt')
    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 2

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)

        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])

###### Class for Dust Params ######
class DustAttnPrior(priors.Prior):
    """A Gaussian prior designed to approximate the Nagaraj et al. 2022 
    dust attenuation curve population model.
    """

    prior_params = ['logsfr','logstmass','logzsol','zred','d1min','d1max','d2min','d2max','nmin','nmax']
    distribution = truncnorm

    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 3
        
    def scale_d12(self,tau2):
        return 0.0725*np.sqrt(0.172*(tau2-0.0119))+0.0548

    def loc_d2n(self,numrep=10):
        indep = np.repeat([[self.params['logstmass'],self.params['logsfr'],self.params['logzsol'],self.params['zred'],0.0]],numrep,axis=0)
        indep[:,-1] = np.random.choice(inc_sample,size=numrep)
        return np.average(d2int(indep),axis=0), np.average(nint(indep),axis=0)

    def loc_d12(self,tau2):
        if type(tau2)==np.float64 or type(tau2)==np.float32: 
            return d1int([tau2])[0]
        else: 
            try: # Array of values
                return d1int(tau2)
            except: # Weird case where it's a numpy 0-dim array
                return d1int([tau2])

    def get_args_d2n(self):
        locd2, locn = self.loc_d2n()
        a_d2 = (self.params['d2min'] - locd2) / wtmed
        b_d2 = (self.params['d2max'] - locd2) / wtmed
        a_n = (self.params['nmin'] - locn) / wnmed
        b_n = (self.params['nmax'] - locn) / wnmed
        return a_d2, b_d2, a_n, b_n, locd2, locn

    def get_args_d12(self,tau2):
        locd1 = self.loc_d12(tau2)
        wd1med = self.scale_d12(tau2)
        a_d1 = (self.params['d1min'] - locd1) / wd1med
        b_d1 = (self.params['d1max'] - locd1) / wd1med
        return a_d1, b_d1, locd1, wd1med

    @property
    def range(self):
        return ((self.params['d2min'], self.params['d2max']),\
                (self.params['nmin'], self.params['nmax']),
                (self.params['d1min'], self.params['d1max']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a_d2, b_d2, a_n, b_n, locd2, locn = self.get_args_d2n()
        a_d1, b_d1, locd1, wd1med = self.get_args_d12(x[...,0])
        p[...,0] = self.distribution.pdf(x[...,0], a_d2, b_d2, loc=locd2, scale=wtmed)
        p[...,1] = self.distribution.pdf(x[...,1], a_n, b_n, loc=locn, scale=wnmed)
        p[...,2] = self.distribution.pdf(x[...,2], a_d1, b_d1, loc=locd1, scale=wd1med)
        with np.errstate(invalid='ignore'):
            p = np.log(p)
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        a_d2, b_d2, a_n, b_n, locd2, locn = self.get_args_d2n()
        tau2 = self.distribution.rvs(a_d2, b_d2, loc=locd2, scale=wtmed, size=nsample)
        n = self.distribution.rvs(a_n, b_n, loc=locn, scale=wnmed, size=nsample)
        a_d1, b_d1, locd1, wd1med = self.get_args_d12(tau2=tau2)
        tau1 = self.distribution.rvs(a_d1, b_d1, loc=locd1, scale=wd1med, size=nsample)

        return np.array([tau2, n, tau1])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        a_d2, b_d2, a_n, b_n, locd2, locn = self.get_args_d2n()
        tau2 = self.distribution.ppf(x[0], a_d2, b_d2, loc=locd2, scale=wtmed)
        n = self.distribution.ppf(x[1], a_n, b_n, loc=locn, scale=wnmed)
        a_d1, b_d1, locd1, wd1med = self.get_args_d12(tau2=tau2)
        tau1 = self.distribution.ppf(x[2], a_d1, b_d1, loc=locd1, scale=wd1med)
        return np.array([tau2,n,tau1])

###### Redefine SPS ######
class NebSFH(FastStepBasis):
    
    @property
    def emline_wavelengths(self):
        return self.ssp.emline_wavelengths

    @property
    def get_nebline_luminosity(self):
        """Emission line luminosities in units of Lsun per solar mass formed
        """
        return self.ssp.emline_luminosity/self.params['mass'].sum()

    def nebline_photometry(self,filters,z):
        """analytically calculate emission line contribution to photometry
        """
        emlams = self.emline_wavelengths * (1+z)
        elums = self.get_nebline_luminosity # Lsun / solar mass formed
        flux = np.empty(len(filters))
        for i,filt in enumerate(filters):
            # calculate transmission at nebular emission
            trans = np.interp(emlams, filt.wavelength, filt.transmission, left=0., right=0.)
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*emlams[idx]*elums[idx]).sum()/filt.ab_zero_counts
            else:
                flux[i] = 0.0
        return flux

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """Get a spectrum and SED for the given params.
        check for flag nebeminspec. if not true,
        add emission lines directly to photometry
        """

        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Redshifting + Wavelength solution
        # We do it ourselves.
        a = 1 + self.params.get('zred', 0)
        af = a
        b = 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = np.polynomial.chebyshev.chebval(x, c) / (lightspeed*1e-13)

        wa, sa = wave * (a + b), spectrum * af  # Observed Frame
        if outwave is None:
            outwave = wa
        
        spec_aa = lightspeed/wa**2 * sa # convert to perAA
        # Observed frame photometry, as absolute maggies
        if filters is not None:
            mags = observate.getSED(wa, spec_aa * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        ### if we don't have emission lines, add them
        if (not self.params['nebemlineinspec']) and self.params['add_neb_emission']:
            phot += self.nebline_photometry(filters,a-1)*to_cgs

        # Spectral smoothing.
        do_smooth = (('sigma_smooth' in self.params) and
                     ('sigma_smooth' in self.reserved_params))
        if do_smooth:
            # We do it ourselves.
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        elif outwave is not wa:
            # Just interpolate
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            # no interpolation necessary
            smspec = sa

        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = WMAP9.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / 1e3 / (3631*jansky_mks)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac

def load_sps(**extras):

    sps = NebSFH(**extras)
    return sps

def load_model(nbins_sfh=7, sigma=0.3, df=2, agelims=run_params['agelims'], objname=None, datdir=None, runname=None, zred=None,
               load_massmet=True, **extras):

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # first calculate redshift and corresponding t_universe
    # if no redshift is specified, read from file
    if zred is None:
        datname = datdir + objname.split('_')[0] + '_' + runname + '.dat'
        dat = ascii.read(datname)
        idx = dat['phot_id'] == int(objname.split('_')[-1])
        zred = float(dat['z_best'][idx])
    tuniv = WMAP9.age(zred).value*1e9

    # now construct the nonparametric SFH
    # current scheme: last bin is 15% age of the Universe, first two are 0-30, 30-100
    # remaining N-3 bins spaced equally in logarithmic space
    tbinmax = (tuniv*0.85)
    agelims = agelims[:2] + np.linspace(agelims[2],np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])

    # load nvariables and agebins
    model_params[n.index('agebins')]['N'] = nbins_sfh
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('mass')]['N'] = nbins_sfh
    model_params[n.index('logsfr_ratios')]['N'] = nbins_sfh-1
    model_params[n.index('logsfr_ratios')]['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params[n.index('logsfr_ratios')]['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0), scale=np.full(nbins_sfh-1,sigma), df=np.full(nbins_sfh-1,df))
    # set mass-metallicity prior
    # insert redshift into model dictionary
    if load_massmet:
        model_params[n.index('massmet')]['prior'] = MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=12.5)
    else:
        model_params[n.index('massmet')]['prior'] = priors.TopHat(mini=-1.98, maxi=0.19)
    model_params[n.index('zred')]['init'] = zred
    model_params[n.index('dustattn')]['prior'] = DustAttnPrior(d1min=0.0,d1max=6.0,d2min=0.0,d2max=4.0,nmin=-1.0,nmax=0.4,logstmass=model_params[n.index('logmass')]['init']+log_stellar_tot_ratio,logsfr=model_params[n.index('logsfr')]['init'],logzsol=model_params[n.index('logzsol')]['init'],zred=zred)
    return sedmodel.SedModel(model_params)

def build_noise(**extras):
    return None, None