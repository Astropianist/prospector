import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import prospect.io.read_results as reader
from prospect.plotting.utils import sample_posterior
import argparse as ap
import corner
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

def parse_args(argv=None):
    parser = ap.ArgumentParser(description="dust_model_run",
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fn", "--filename",
                        help='''Name of hdf5 output''',
                        type=str, default=None) 

    args = parser.parse_args(args=argv)
    if 'uda_1' in args.filename: 
        args.uda = 1
        import dust_model_params as dmp
    else: 
        args.uda = 0
        import td_delta_params as dmp

    return args

def plot_corner(chain,chain2,theta_labels,true_vals=None,fn='Results/sample_mcmc.h5',weights=None,weights2=None):
    indf = np.where(np.all(np.isfinite(chain),axis=1))[0]
    percentile_range = [0.95]*len(theta_labels)
    fig = corner.corner(chain[indf],labels=theta_labels,truths=true_vals,weights=weights[indf],truth_color='k',show_titles=True,quantiles=[0.16, 0.5, 0.84],bins=30,smooth=2.0,smooth1d=None, color='indigo',range=percentile_range)

    indf = np.where(np.all(np.isfinite(chain2),axis=1))[0]
    corner.corner(chain2[indf],weights=weights2[indf],quantiles=[0.16, 0.5, 0.84],bins=30,smooth=2.0,smooth1d=None, color='red',range=percentile_range, fig=fig)

    fig.savefig(fn.replace('_mcmc.h5','_corner.png'),bbox_inches='tight',dpi=200)

def make_seds(res,obs,mod,sps,n_seds=-1):
        """Generate and cache the best fit model spectrum and photometry.
        Optionally generate the spectrum and photometry for a number of
        posterior samples.

        Populates the attributes `*_best` and `*_samples` where `*` is:
        * spec
        * phot
        * sed
        * cal

        :param full: bool, optional
            If true, generate the intrinsic spextrum (`sed_*`) over the entire wavelength
            range.  The (restframe) wavelength vector will be given by
            `sps.wavelengths`
        """
        if n_seds > 0:
            spec_samples, phot_samples = draw_seds(res, obs, mod, sps, n_seds)
        else:
            # --- best sample ---
            ind_best = np.nanargmax(res["lnprobability"])
            xbest = res["chain"][ind_best, :]
            spec_best, phot_best, _ = mod.predict(xbest, obs=obs, sps=sps)
            spec_samples, phot_samples = np.atleast_2d(spec_best), np.atleast_2d(phot_best)

        return spec_samples, phot_samples

def draw_seds(res, obs, mod, sps, n_seds):
    """Draw a number of samples from the posterior chain, and generate
    spectra and photometry for them.

    :param n_seds: int
        Number of samples to draw and generate SEDs for

    :param dummy: dict, optional
        If given, use this dictionary as the obs dictionary when generating the intrinsic SED vector.
        Useful for generating an SED over a large wavelength range
    """
    chain = res['chain']
    indf = np.where(np.all(np.isfinite(chain),axis=1))[0]
    raw_samples = sample_posterior(chain[indf], res['weights'][indf], nsample=n_seds)
    spec, phot = [], []
    for x in raw_samples:
        s, p, _ = mod.predict(x, obs=obs, sps=sps)
        spec.append(s)
        phot.append(p)

    # should make this a named tuple
    return np.array(spec), np.array(phot)

def plot_sed(res,obs,mod,sps,fn='Results/sample_mcmc.h5',n_seds=-1):
    zred = float(mod.params["zred"])
    pmask = obs["phot_mask"]
    ophot, ounc = obs["maggies"][pmask], obs["maggies_unc"][pmask]
    owave = np.array([f.wave_effective for f in obs["filters"]])[pmask]
    if obs['wavelength'] is None:
        maxov = owave.max()
        if maxov>25000.0*(1+zred): ovm = max(maxov,150000.0*(1+zred))
        else: ovm = 25000.0*(1+zred)
        obs['wavelength'] = np.geomspace(min(owave.min(),1200.0*(1+zred)),ovm,1001)
    spec_sample, phot_sample = make_seds(res,obs,mod,sps,n_seds=n_seds)
    pwave, phot_sample = obs["phot_wave"][pmask], phot_sample[:,pmask]
    swave = obs.get("wavelength", None)
    if swave is None:
        swave = sps.wavelengths * (1 + zred)

    if n_seds>0:
        spec_best = np.median(spec_sample,axis=0)
        spec_err = np.std(spec_sample,axis=0)
        phot_best = np.median(phot_sample,axis=0)
        phot_err = np.std(phot_sample,axis=0)
    else:
        spec_best = spec_sample[0]
        phot_best = phot_sample[0]
        phot_err = np.zeros_like(phot_best)
    phot_resid = (phot_best-ophot)/np.sqrt(phot_err**2+ounc**2)
    figs, sedax = plt.subplots()
    # plot SED
    sedax.errorbar(pwave, phot_best, yerr=phot_err, marker="o", linestyle="",
                color='r', label="Best-fit photometry")
    if n_seds>0:
        sedax.fill_between(swave,spec_best-spec_err,spec_best+spec_err,color='k',alpha=0.1,label='')
        label_best = 'Median spectrum'
    else:
        label_best = 'Best-fit spectrum'
    sedax.plot(swave, spec_best, 'b-',
                label=label_best)
    sedax.errorbar(owave, ophot, yerr=ounc, marker='^', color='k', linestyle='', label='Observed photometry')
    sedax.set_xlim([swave.min(),swave.max()])
    sedax.set_ylim([spec_best.min(),spec_best.max()])
    sedax.set_xlabel(r'$\lambda_{\rm{obs}}$ ($\AA$)')
    sedax.set_ylabel('Flux')
    sedax.set_xscale('log')
    sedax.set_yscale('log')
    sedax.legend(loc='best')
    figs.savefig(fn.replace('_mcmc.h5','_sed.png'),bbox_inches='tight',dpi=300)

    return phot_resid.sum()/len(phot_resid)

def main(filename='',n_seds=200):
    import dust_model_params as dmp
    # args = parse_args()
    # if 'uda_1' in filename: 
    #     uda = 1
    #     import dust_model_params as dmp
    # else: 
    #     uda = 0
    #     import td_delta_params as dmp
    filename_td = filename.replace('uda_1','uda_0')
    res, obs, mod = reader.results_from(filename)
    res_td, obs_td, mod_td = reader.results_from(filename_td)
    logsfr = np.empty_like(res['chain'][:,0])
    logsfr_td = np.empty_like(res_td['chain'][:,0])
    for i in range(len(logsfr)):
        logsfr[i] = dmp.logmass_to_logsfr(massmet=res['chain'][i,0:2],logsfr_ratios=res['chain'][i,2:8],agebins=mod.params['agebins'])
    for i in range(len(logsfr_td)):
        logsfr_td[i] = dmp.logmass_to_logsfr(massmet=res_td['chain'][i,0:2],logsfr_ratios=res_td['chain'][i,2:8],agebins=mod_td.params['agebins'])
    inds = np.array([0,1,2,8,9,10])
    chain, chain_td = res['chain'][:,inds], res_td['chain'][:,inds]
    chain[:,2], chain_td[:,2] = logsfr, logsfr_td
    chain_td[:,-1] = chain_td[:,-1] * chain_td[:,-3] # Dust1_ratio to Dust1
    rp = res['run_params']
    true_sfr = dmp.logmass_to_logsfr(massmet=rp['massmet'],logsfr_ratios=rp['logsfr_ratios'],agebins=mod.params['agebins'])
    true_vals = [rp['logmass'],rp['logzsol'],true_sfr] + rp['dustattn']
    labels = [r'$\log({\rm M}_*)$',r'$\log(Z/Z_\odot)$',r'$\log({\rm SFR})$',r'$\tau_2$',r'$n$',r'$\tau_1$']

    plot_corner(chain,chain_td,labels,fn=filename,true_vals=true_vals,weights=res['weights'],weights2=res_td['weights'])

    fn_sed = filename.replace('_mcmc.h5','_sed.png')
    if exists(fn_sed):
        return

    sps = dmp.load_sps(**res['run_params'])
    phot_resid_avg = plot_sed(res,obs,mod,sps,fn=filename,n_seds=n_seds)
    phot_resid_avg_td = plot_sed(res_td,obs_td,mod_td,sps,fn=filename_td,n_seds=n_seds)
    with open('ProspResid.dat','a') as resid_file:
        resid_file.write(f'{filename}  {phot_resid_avg:.2f} \n')
        resid_file.write(f'{filename_td}  {phot_resid_avg_td:.2f} \n')

if __name__ == '__main__':
    main()