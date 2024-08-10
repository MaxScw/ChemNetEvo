from matplotlib import pyplot as plt
from load_evo_results import load_evo_results
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib
import model as evo
from effect_size_distribution import get_effect_size_dist

matplotlib.rcParams.update({'font.size': 12})

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
mut_pop, mut_fit = load_evo_results(n_pop=n_pop,
                                                        n_chem=n_chem,
                                                        n_react=n_react,
                                                        n_epochs=n_epochs,
                                                        appendix=appendix,
                                                        path=fpath,
                                                        noMut=False)


target_smat = evo.np.zeros((n_chem, n_react))
target_smat[:, 0] = evo.np.array([1, 1, 1])
target_smat[0, 1] = 1
target_smat[1, 2] = 1
target_smat[:, 3] = evo.np.array([1, 1, 1])
target_smat[0, 4] = 1
target_smat[1, 4] = 1

# collect number of unique networks + network ranks

#noMut_div_hist = []
mut_div_hist = []

#noMut_avRank = []
mut_div_avRank = []

ref_fit_mean = []
ref_fit_max = []

avEffDist_hist = []

for ep in tqdm(range(n_epochs+1)):
    ref_fit = []
    if ep%save_ep==0:
        ep = ep//save_ep
        #noMut_div_hist.append(np.unique(noMut_pop[:, :, :, ep], axis=2).shape[-1])
        mut_div_hist.append(np.unique(mut_pop[:, :, :, ep], axis=2).shape[-1])

        #noMut_avRank.append(np.array([np.linalg.matrix_rank(noMut_pop[:, :, i, ep])
        #                            for i in range(n_pop)]))
        mut_div_avRank.append(np.array([np.linalg.matrix_rank(mut_pop[:, :, i, ep])
                                        for i in range(n_pop)]))
        
        
        # calculate reference fitness

        singular, sMatrix, aRank = evo.get_vector_sMatrix(mut_pop[:, :, :,ep])
        avEffDists = []
        for pop_ind in tqdm(range(n_pop)):
            if singular[pop_ind]==False:
                ref_fit.append(round(evo.get_fitness_pertRespSmat(
                                sMatrix[pop_ind][:n_chem, :n_react],
                                target_mat=target_smat), 8))
                avEffDists.append(get_effect_size_dist(mut_pop[:, :, pop_ind,ep], target_smat))
    
        avEffDist_hist.append(np.mean(avEffDists))
        ref_fit_mean.append(np.mean(ref_fit))
        ref_fit_max.append(max(ref_fit))

# plot evolution of number of networks + network rank

div_fig, div_ax = plt.subplots(1, 2)
div_fig.tight_layout()
div_fig.set_size_inches(10, 5)

ep_range = np.linspace(1, n_epochs, n_epochs//save_ep+1)
""" div_ax[0].plot(ep_range, noMut_div_hist, 'o',
                  c='red', label='only selection', alpha=0.5) """
div_ax[0].plot(ep_range, mut_div_hist, 'o',
                  c='blue', label='mutation+selection')
div_ax[0].legend()
div_ax[0].set_yscale('log')

""" div_ax[1].plot(ep_range, np.array(noMut_avRank).mean(axis=1),'o',
                  c='red', label='only selection', alpha=0.5) """
div_ax[1].plot(ep_range, np.array(mut_div_avRank).mean(axis=1),'o',
                  c='blue', label='mutation+selection')

div_ax[0].set_ylabel('log of number of unique networks')
div_ax[0].set_xlabel('evolutionary epochs')
div_ax[0].set_title('time development of diversity')
                 

div_ax[1].set_ylabel('rank of $\\nu$')
div_ax[1].set_xlabel('evolutionary epochs')
div_ax[1].set_title('time development of $\\nu$ rank')
                    
div_ax[1].legend()

div_fig.savefig(f'divergence_plot'+run_string+'_'+save_appendix+'.pdf', format='pdf')


# plot evolution of fitness
fit_fig, fit_axs = plt.subplots(1)
fit_fig.tight_layout()
fit_fig.set_size_inches(5, 5)

fit_fig2, fit_axs2 = plt.subplots(1)
fit_fig2.tight_layout()
fit_fig2.set_size_inches(5, 5)

fit_axs.plot(np.linspace(0, n_epochs, n_epochs+1)[::10],mut_fit.mean(axis=0)[::10], 'o',
            c='blue', label='$S^u_{target}$ mean fit.')
fit_axs.plot(np.linspace(0, n_epochs, n_epochs+1)[::10],mut_fit.max(axis=0)[::10], 
             c='blue', label='$S^u_{target}$ max fit.')
fit_axs.plot(np.linspace(0, n_epochs, n_epochs+1)[::100], ref_fit_mean, 'o',
            c='red', label='$S^u_{0}$ mean fit.', alpha=0.5)
fit_axs.plot(np.linspace(0, n_epochs, n_epochs+1)[::100], ref_fit_max, 
             c='red', label='$S^u_{0}$ max fit.', alpha=0.5)

fit_axs2.plot(np.linspace(0, n_epochs, n_epochs+1)[::100], avEffDist_hist, 'o',
             c='green', alpha=0.5)

fit_axs.set_ylim([0.5, 1.1])
fit_axs.set_ylabel('fitness (a.u.)')
fit_axs.set_xlabel('evolution epoch')
fit_axs.set_title('fitness population average')
fit_axs.legend(loc='lower right')

fit_axs2.set_ylim([-0.71, -0.1])
fit_axs2.set_ylabel('$\Delta F$')
fit_axs2.set_xlabel('evolution epoch')
fit_axs2.set_title('average mut. effect size')

fit_fig.tight_layout()
fit_fig2.tight_layout()
fit_fig.savefig(f'fitness_plot'+run_string+'_'+save_appendix+'.pdf', format='pdf')
fit_fig2.savefig(f'effSize_plot'+run_string+'_'+save_appendix+'.pdf', format='pdf')