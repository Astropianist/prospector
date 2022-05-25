import pickle
import dust_model_params as dmp
import td_delta_params as pfile
import time
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.plotting import FigureMaker
import argparse as ap
filterset = ['sdss_r0','sdss_g0','sdss_u0','sdss_i0','sdss_z0','twomass_J','twomass_Ks','uvot_w2','uvot_m2','uvot_w1','spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4','spitzer_mips_24','herschel_pacs_70','herschel_pacs_100','herschel_pacs_160','herschel_spire_250','herschel_spire_350','herschel_spire_500']

def parse_args(argv=None):
    '''Parse arguments from commandline or a manually passed list
    Parameters
    ----------
    argv : list
        list of strings such as ['-f', 'input_file.txt', '-s', 'default.ssp']
    Returns
    -------
    args : class
        args class has attributes of each input, i.e., args.filename
        as well as astributes from the config file
    '''
    parser = ap.ArgumentParser(description="dust_model_run",
                               formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-snr", "--snr",
                        help='''Signal to noise ratio of mock data''',
                        type=float, default=10.0)  

    parser.add_argument("-z", "--zred",
                        help='''Redshift''',
                        type=float, default=1.0)

    parser.add_argument("-uda", "--use_dust_attn",
                        help='''Use dust attenuation population model as prior''',
                        action='count',default=0)

    parser.add_argument("-mmp", "--make_model_pickle",
                        help='''Make the model pickle file''',
                        action='count',default=0)
    
    parser.add_argument("-mf", "--make_figures",
                        help='''Make just the figures''',
                        action='count',default=0)

    parser.add_argument("-nf", "--num_filters",
                         help='''Number of filters to use''',
                         type=int, default=2) 

    parser.add_argument("-o","--outfile", type=str, default="prospector_test_run",
                        help="Root name (including path) of the output file(s).")

    parser.add_argument('-d2','--dust2', type=float, default=0.5,
                        help="Dust attenuation V band optical depth")
    parser.add_argument('-d1','--dust1', type=float, default=0.45,
                        help="Dust attenuation V band optical depth")
    parser.add_argument('-n','--dust_index', type=float, default=0.0,
                        help="Dust attenuation V band optical depth")
    parser.add_argument('-logZ','--logzsol', type=float, default=-0.5,
                        help="Metallicity of the mock; log(Z/Z_sun)")
    parser.add_argument('-logM','--logmass', type=float, default=10.0,
                        help="Log stellar mass of the mock; log solar masses formed")
    parser.add_argument('-lsr','--logsfr_ratios', type=float, nargs='*', default=[0.0]*6, help="Stellar mass of the mock; solar masses formed")

    # Initialize arguments
    args = parser.parse_args(args=argv)
    args.dustattn = [args.dust2,args.dust_index,args.dust1]
    args.massmet = [args.logmass,args.logzsol]
    args.dust1_ratio = args.dust1/args.dust2
    return args

def main():
    args = parse_args()
    run_params = dmp.run_params
    run_params.update(vars(args))
    run_params['filterset'] = filterset[:args.num_filters]
    hfile = "{0}_{1}_{2}_uda_{3}_args_{4}_{5}_{6}_{7}_{8}_{9}_{10}_mcmc.h5".format(args.outfile, args.num_filters, int(args.snr), int(args.use_dust_attn), str(args.logmass), str(args.logzsol), str(args.zred), str(args.logsfr_ratios), str(args.dust2), str(args.dust_index), str(args.dust1))
    mfile = hfile.replace('_mcmc.h5','_model')
    print("Finished creating run_params")
    if not args.make_figures:
        if args.use_dust_attn:
            mod = dmp.load_model(**run_params)
        else:
            mod = pfile.load_model(**run_params)
        print("Loaded fresh model for testing")
        modoutput = {}
        modoutput['powell'] = None
        modoutput['prospector_version'] = '1.1.0'
        modoutput['model'] = mod
        pickle.dump(modoutput,open(mfile,'wb'),protocol=4)
        if args.make_model_pickle: return
        obs, sps = dmp.build_obs(**run_params)
        print("Built mock obs and stored sps")

        output = fit_model(obs, mod, sps, **run_params)

        writer.write_hdf5(hfile, run_params, mod, obs,
                        output["sampling"][0], output["optimization"][0],
                        tsample=output["sampling"][1],
                        toptimize=output["optimization"][1],
                        sps=sps)

    #### Plotting stuff #######
    import SimplePlotting as SP
    SP.main(filename=hfile, n_seds=200)

    # show = ['dust2','dust_index','dust1','logmass','logsfr','logzsol']
    # show_labels = [r'$\tau_2$',r'$n$',r'$\tau_1$',r'log(M$_{\rm{st,tot}}$)','log(SFR)',r'$\log (Z/Z_\odot)$']
    # try:
    #     figobj = FigureMaker(results_file=hfile,show=show,show_labels=show_labels)
    #     figobj.plot_all()
    # except:
    #     pass

if __name__ == '__main__':
    main()