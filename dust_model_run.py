import numpy as np
import dust_model_params as dmp
import td_delta_params as pfile
import time
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect import prospect_args

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
    parser = prospect_args.get_parser()

    parser.add_argument("-snr", "--snr",
                        help='''Signal to noise ratio of mock data''',
                        type=float, default=10.0)  

    parser.add_argument("-z", "--zred",
                        help='''Redshift''',
                        type=float, default=1.0)

    # parser.add_argument("-uda", "--use_dust_attn",
    #                     help='''Use dust attenuation population model as prior''',
    #                     action='count',default=0)

    parser.add_argument("-nf", "--num_filters",
                         help='''Number of filters to use''',
                         type=int, default=2)               

    # Initialize arguments
    return parser.parse_args(args=argv)

def main():
    args = parse_args()
    obs = dmp.build_obs(snr=args.snr,filterset=filterset[:args.num_filters],**dmp.run_params)
    modda = dmp.load_model(zred=args.zred,**args,**dmp.run_params)
    modnoda = pfile.load_model(zred=args.zred,**args,**dmp.run_params)
    sps = pfile.load_sps(**args,**pfile.run_params)

    hfileda = "{0}_da_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    hfilenoda = "{0}_noda_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    outputda = fit_model(obs, modda, sps, None,**args, **dmp.run_params)
    outputnoda = fit_model(obs, modnoda, sps, None,**args, **dmp.run_params)

    writer.write_hdf5(hfileda, dmp.run_params, modda, obs,
                      outputda["sampling"][0], outputda["optimization"][0],
                      tsample=outputda["sampling"][1],
                      toptimize=outputda["optimization"][1],
                      sps=sps)

    writer.write_hdf5(hfilenoda, dmp.run_params, modnoda, obs,
                      outputnoda["sampling"][0], outputnoda["optimization"][0],
                      tsample=outputnoda["sampling"][1],
                      toptimize=outputnoda["optimization"][1],
                      sps=sps)

    try:
        hfileda.close()
        hfilenoda.close()
    except(AttributeError):
        pass

if __name__ == '__main__':
    main()