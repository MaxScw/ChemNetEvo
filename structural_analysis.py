import numpy as np
import model as evo
from tqdm import tqdm
from load_evo_results import load_evo_results
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 12})

import argparse

parser = argparse.ArgumentParser(description="input specifics of evo run to be analysed",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n_pop", help="archive mode", type=int)
parser.add_argument("-n_chem", help="increase verbosity", type=int)
parser.add_argument("-n_react", help="checksum blocksize", type=int)
parser.add_argument("-n_epochs", help="skip files that exist", type=int)
parser.add_argument("-save_ep", help="files to exclude", type=int)
parser.add_argument("--appendix", help="Source location", 
                    default=[], nargs="*")
parser.add_argument("--save_appendix", help="Destination location", 
                    default=[], nargs="*")
parser.add_argument("--fpath", help="Destination location", 
                   default=[], nargs="*")
args = parser.parse_args()
config = vars(args)

n_pop = config['n_pop']
n_chem = config['n_chem']
n_react = config['n_react']
n_epochs = config['n_epochs']
save_ep = config['save_ep']
appendix = ''.join(config['appendix'])
save_appendix = ''.join(config['save_appendix'])
fpath = ''.join(config['fpath'])

run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}'
evo_results = load_evo_results(n_pop=n_pop,
                               n_chem=n_chem,
                               n_react=n_react,
                               n_epochs=n_epochs,
                               appendix=appendix,
                               path=fpath)
mut_pop = evo_results[0]
mut_fit = evo_results[1]
noMut_pop = evo_results[2]
noMut_fit = evo_results[3]

# choose evolution generations for which to sample buffering structure dist.
sample_eps = np.linspace(0, n_epochs, n_epochs//save_ep+1).astype(int)

n_buff_mut_hist = []
n_buff_noMut_hist = []

for ep in tqdm(sample_eps):
    ep = int(ep//save_ep)
    pop_out = []
    pop_in = []
    
    n_buff_mut = []
    n_buff_noMut = []

    sMats_mut = []
    sMats_noMut = []
    
    for pop_ind in range(n_pop):
        sMat_mut = evo.get_sMatrix(mut_pop[:, :, pop_ind, ep])
        sMat_noMut = evo.get_sMatrix(noMut_pop[:, :, pop_ind, ep])
        if sMat_mut[0]==False:
            plt.imshow(sMat_mut[1])
            n_buff_mut.append(np.unique(sMat_mut[1][:n_chem, :n_react],
                                    axis=1).shape[1])
            sMats_mut.append(sMat_mut)
            
        if sMat_noMut[0]==False:
            n_buff_noMut.append(np.unique(sMat_noMut[1][:n_chem, :n_react],
                                    axis=1).shape[1])
            sMats_noMut.append(sMat_noMut)
    
    n_buff_noMut_hist.append(n_buff_noMut)
    n_buff_mut_hist.append(n_buff_mut)
    
  
# plot heatmap of buffering structure distribution as a function of ev. epoch
buff_fig, buff_ax = plt.subplots(1, 2)
buff_fig.tight_layout()
buff_fig.set_size_inches(10, 5)

binrange = [1, n_react+1]
nbins = n_react
bins = np.linspace(binrange[0], binrange[1]-1, nbins)
buff_mut_histos = []
buff_noMut_histos = []

for sample_ind in range(len(sample_eps)):
    buff_mut_histos.append(np.histogram(
                                    n_buff_mut_hist[sample_ind], bins=nbins,
                                    range=binrange, density=True)[0])
    buff_noMut_histos.append(np.histogram(
                                    n_buff_noMut_hist[sample_ind], bins=nbins,
                                    range=binrange, density=True)[0])

buff_mut_plot = buff_ax[0].imshow(np.array(buff_noMut_histos), vmin=0, vmax=1)
buff_noMut_plot = buff_ax[1].imshow(np.array(buff_mut_histos), vmin=0, vmax=1)
cb = buff_fig.colorbar(buff_mut_plot, shrink=0.5)
cb.ax.set_title('relative frequency', fontsize=9)
cb = buff_fig.colorbar(buff_noMut_plot, shrink=0.5)
cb.ax.set_title('relative frequency', fontsize=9)

buff_ax[0].set_ylabel('generation')
buff_ax[0].set_xlabel('# indep. columns of chem. sens. matrix')
buff_ax[1].set_ylabel('generation')
buff_ax[1].set_xlabel('# indep. columns of chem. sens. matrix')
buff_ax[0].set_title('only selection')
buff_ax[1].set_title('selection + mutation')

buff_ax[0].set_xticks(np.concatenate((np.array([0]), bins[:-1]))[::2].astype(int))
buff_ax[0].set_xticklabels(bins[::2].astype(int))
buff_ax[1].set_xticks(np.concatenate((np.array([0]), bins[:-1]))[::2].astype(int))
buff_ax[1].set_xticklabels(bins[::2].astype(int))

buff_ax[0].set_yticks(np.arange(len(sample_eps))[::2])
buff_ax[0].set_yticklabels(sample_eps[::2])
buff_ax[1].set_yticks(np.arange(len(sample_eps))[::2])
buff_ax[1].set_yticklabels(sample_eps[::2])


# plot histograms of buffering structure distribution for all sample epochs
hist_fig, hist_ax = plt.subplots(1, 2)
hist_fig.tight_layout()
hist_fig.set_size_inches(10, 5)

cmap = matplotlib.colormaps.get_cmap('magma')
cs = []
hist_ax[0].set_ylabel('relative frequency in one generation')
hist_ax[0].set_xlabel('# indep. columns of chem. sens. matrix')
hist_ax[1].set_ylabel('relative frequency in one generation')
hist_ax[1].set_xlabel('# of indep. columns of chem. sens. matrix')

hist_ax[0].set_title('only selection')
hist_ax[1].set_title('selection + mutation')

for sample_ind in range(len(sample_eps)):
    c = cmap(1/(sample_ind+1))
    
    hist_ax[0].bar(bins, buff_noMut_histos[sample_ind],
                   alpha=0.5, fc=c, label=f'gen.{sample_eps[sample_ind]}')
    hist_ax[0].axvline(x=bins[np.argmax(
                       buff_noMut_histos[sample_ind])],
                       color=c, linestyle='--')
    
    hist_ax[1].bar(bins, buff_mut_histos[sample_ind],
                   alpha=0.5, fc=c, label=f'gen.{sample_eps[sample_ind]}')
    hist_ax[1].axvline(x=bins[np.argmax(
                       buff_mut_histos[sample_ind])],
                       color=c, linestyle='--')

hist_ax[0].set_xticks(bins[::2].astype(int))
hist_ax[0].set_xticklabels(bins[::2].astype(int))
hist_ax[0].legend()
hist_ax[0].set_ylim([0, 1])
hist_ax[1].set_xticks(bins[::2].astype(int))
hist_ax[1].set_xticklabels(bins[::2].astype(int))
hist_ax[1].legend()
hist_ax[1].set_ylim([0, 1])

buff_fig.savefig(f'buff_evol'+run_string+'_'+save_appendix+'.pdf', 
                 format='pdf')
hist_fig.savefig(f'buffDist_evol'+run_string+'_'+save_appendix+'.pdf', 
                 format='pdf')


# plot fitness distribution for initial and final population
fitDist_fig, fitDist_ax = plt.subplots(1, 2)
fitDist_fig.tight_layout()
fitDist_fig.set_size_inches(10, 5)

nbins = 30
binrange = [0.9, 1]
bin_width = (binrange[-1]-binrange[0])/nbins


noMut_fitVals0 = np.histogram(noMut_fit[:, 0], bins=nbins, range=binrange)
noMut_fitVals = np.histogram(noMut_fit[:, -1], bins=nbins, range=binrange)

fitDist_ax[0].bar(noMut_fitVals0[1][:-1]+0.5*bin_width, 
                  noMut_fitVals0[0]/np.sum(noMut_fitVals0[0]), 
                  width=bin_width, fc='red', label='initial population')
fitDist_ax[0].bar(noMut_fitVals[1][:-1]+0.5*bin_width, 
                  noMut_fitVals[0]/np.sum(noMut_fitVals[0]), 
                  width=bin_width, alpha=0.5, fc='green',
                  label='final evolved population')
fitDist_ax[0].set_xlabel('fitness (a.u.)')
fitDist_ax[0].set_ylabel('relative frequency in one generation')
fitDist_ax[0].set_xlim(binrange)
fitDist_ax[0].set_ylim([0, 1])
fitDist_ax[0].legend()
fitDist_ax[0].set_title('only selection')
print(max(mut_fit[:, -1]))
mut_fitVals0 = np.histogram(mut_fit[:, 0], bins=nbins, range=binrange)
mut_fitVals = np.histogram(mut_fit[:, -1], bins=nbins, range=binrange)
fitDist_ax[1].bar(mut_fitVals0[1][:-1]+0.5*bin_width, 
                  mut_fitVals0[0]/np.sum(mut_fitVals0[0]), 
                  width=bin_width, fc='red', label='initial population')
fitDist_ax[1].bar(mut_fitVals[1][:-1]+0.5*bin_width, 
                  mut_fitVals[0]/np.sum(mut_fitVals[0]), 
                  width=bin_width, alpha=0.5, fc='green',
                  label='final evolved population')
fitDist_ax[1].set_xlabel('fitness (a.u.)')
fitDist_ax[1].set_ylabel('relative frequency in population')
fitDist_ax[1].set_xlim(binrange)
fitDist_ax[1].set_ylim([0, 1])
fitDist_ax[1].legend()
fitDist_ax[1].set_title('selection + mutation')

fitDist_fig.savefig(f'fitnessDist_evol'+run_string+'_'+save_appendix+'.pdf', 
                    format='pdf')

plt.show()