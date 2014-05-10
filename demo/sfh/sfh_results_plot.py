import matplotlib.pyplot as pl
import numpy as np
import pickle
import triangle



def results_plot(sample_file, outname = 'demo', nspec = 10, nsfh = 40, sps =None, start =0, thin =1):
    
    sample_results = pickle.load( open(sample_file, 'rb'))
    model = sample_results['model']
    parnames = theta_labels(model.theta_desc)
    flatchain = sample_results['chain'][start::thin,:]
    nchain, ndim = flatchain.shape
    #flatchain = chain.reshape(nchain, ndim)
    obs = sample_results['obs']
    sps.update(model.params)

#plot a triangle plot
    fig = triangle.corner(flatchain,labels = parnames,
                          quantiles=[0.16, 0.5, 0.84], verbose =False,
                          truths = sample_results['mock_input_theta'])
    fig.savefig('{0}_triangle.png'.format(outname))
    pl.close()

#plot SFHs
    pl.figure(1)
    pl.clf()
    rindex = np.random.uniform(0, nchain, nsfh).astype( int )
    tage = np.log10(model.params['tage'] * 1e9)

    pl.plot(tage, flatchain[rindex[0],:] * 0., color = 'gray', label = 'posterior sample')
    for i in range(nspec-1):
        pl.plot(tage, flatchain[rindex[i],:], color = 'gray', alpha = 0.5)
    pl.plot(tage, sample_results['mock_input_theta'], '-o',
            color = 'red', label = 'input/truth', linewidth = 2)
    pl.plot(tage, sample_results['initial_center'], '-',
            color = 'cyan', label = 'minimization result')
    

    pl.xlabel('log Age (years)')
    pl.ylabel(r'Stellar mass formed (M$_\odot$)')
    pl.legend(loc = 'upper left')
    pl.xlim(6.5, 10.5)
    pl.savefig('{0}_sfh.png'.format(outname))
    pl.close(1)
    
#plot spectrum
    pl.figure(1)
    rindex = np.random.uniform(0, nchain, nspec).astype( int )

    pl.plot(obs['wavelength'], obs['spectrum'],
            color = 'blue', linewidth = 1.5, label = 'observed, S/N=50')
    pl.plot(obs['wavelength'], model.model(flatchain[rindex[0],:], sps =sps)[0]*0.,
            color = 'green', alpha = 0.5, label = 'model ({0} samples)'.format(nspec))
    for i in range(nspec-1):
        pl.plot(obs['wavelength'],model.model(flatchain[rindex[i],:], sps =sps)[0],
                color = 'green', alpha = 0.5)

    pl.ylabel(r'L$_\odot/\AA$')
    pl.xlabel(r'wavelength ($\AA$)')
    pl.xlim(3000,10000)
    pl.ylim(0,0.05)
    pl.legend()
    pl.savefig('{0}_spectrum.png'.format(outname), dpi = 300)
    pl.close()

#plot components
    ncolors = len(sps.params['tage'])
    cm = pl.get_cmap('gist_rainbow')
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/ncolors) for i in range(ncolors)])

    ax.plot(obs['wavelength'],model.model(sample_results['mock_input_theta'], sps =sps)[0],
            label = 'total spectrum', color ='black')
    for i,(t,m) in enumerate(zip(sps.params['tage'], sample_results['mock_input_theta'])):
        color = cm(1.*(ncolors-i -1)/ncolors)
        ax.plot(sps.ssp.wavelengths, m * sps.basis_spec[i,:],
                label = 'log(Age) = {0:4.2f}'.format(np.log10(t*1e9)),
                color =color)
        
    ax.set_ylabel(r'L$_\odot/\AA$')
    ax.set_xlabel(r'wavelength ($\AA$)')
    ax.set_xlim(3000,10000)
    ax.legend(loc ='lower right', prop ={'size':8})
    ax.set_yscale('log')
    ax.set_ylim(1e-6,1e-1)
    ax.set_title('Components of the total spectrum')
    pl.savefig('{0}_components.png'.format(outname), dpi = 300)
    pl.close()

    return sample_results, flatchain, model

    
def theta_labels(desc):
    label, index = [], []
    for p in desc.keys():
        nt = desc[p]['N']
        name = p
        #if p is 'mass': name = 'm'
        if nt is 1:
            label.append(name)
            index.append(desc[p]['i0'])
        else:
            for i in xrange(nt):
                label.append(r'${1}_{{{0}}}$'.format(i+1, name))
                index.append(desc[p]['i0']+i)

    return [l for (i,l) in sorted(zip(index,label))]