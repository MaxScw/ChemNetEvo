import numpy as np
import model as evo
import os
from tqdm import tqdm
from load_evo_results import load_evo_results
from matplotlib import pyplot as plt
import matplotlib
import getGraph_fromStochMatrix as graphtool
from get_bs_hierarchy import get_bs_hierarchy
from effect_size_distribution import get_effect_size_dist
from scipy.stats import skew

matplotlib.rcParams.update({'font.size': 12})

n_pop = 1000
n_chem = 3
n_react = 5
n_epochs = 1000
save_ep = 100
fname = ''#'smatDistFit/'#'SmatDist_noiseComp_fb=fp/'
save_plots = False
show_plots = True

fp = 0.15
bp = 0.15

run_names = [f'_PR_fp{fp}_bp{bp}']#, '_PR_fp0.12_bp0.18', '_PR_fp0.15_bp0.15',
             #'_PR_fp0.17_bp0.13', '_PR_fp0.2_bp0.1']#['_fp0_bp0', '_fp0.001_bp0.002', '_fp0.01_bp0.02', 
             #'_fp0.05_bp0.1', '_fp0.1_bp0.2', '_fp0.5_bp1.0']#['fullCorr', 'fp1e5_bp1e5', 'fp51e5_bp51e5', 'fp7p51e5_bp7p51e5', 
nruns = 10

target_smat = np.array([1, 1, 0, 1, 1,
                        1, 0, 1, 1, 1,
                        1, 0, 0, 1, 0]).reshape(3, 5)

for run_name in run_names:
    effect_sizes_0 = []
    effect_sizes_f = []

    for i in range(nruns):
        appendix = run_name + f'_run{i+1}'#f'enzymeDup_run{i+1}'

        run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}'
        evo_results = load_evo_results(n_pop=n_pop,
                                    n_chem=n_chem,
                                    n_react=n_react,
                                    n_epochs=n_epochs,
                                    appendix=appendix,
                                    path=fname,
                                    noMut=False)
        mut_pop = evo_results[0]
        mut_fit = evo_results[1]

        singular_f, sMatrix_f, aRank_f = evo.get_vector_sMatrix(mut_pop[:, :, :,-1])
        singular_0, sMatrix_0, aRank_0 = evo.get_vector_sMatrix(mut_pop[:, :, :,0])

        for pop_ind in range(n_pop):
            if singular_0[pop_ind]==False:
                effect_sizes_0.append(get_effect_size_dist(
                                      sMatrix_0[pop_ind][:n_chem, :n_react], 
                                      target_smat))
            if singular_f[pop_ind]==False:
                effect_sizes_f.append(get_effect_size_dist(
                                      sMatrix_f[pop_ind][:n_chem, :n_react], 
                                      target_smat))
 
    binrange = [-0.1, 0.1]
    nbins = 30
    
    effect_size_f_dist = np.histogram(effect_sizes_f, bins=nbins, range=binrange)
    effect_size_0_dist = np.histogram(effect_sizes_0, bins=nbins, range=binrange)

    # skew_0 = skew(effect_size_0_dist[0]).mean()
    # skew_f = skew(effect_size_f_dist[0]).mean()

    skew_0 = skew(np.array(effect_sizes_0).flatten())
    skew_f = skew(np.array(effect_sizes_f).flatten())

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    fig.suptitle('$p_{forward}$='+f'{fp}'+', $p_{backward}$='+f'{bp}')

    axs[0].bar(effect_size_0_dist[1][:-1], effect_size_0_dist[0]/effect_size_0_dist[0].sum(), width=(binrange[-1]-binrange[0])/nbins)
    axs[0].set_title('starting population\n skew='+f'{round(skew_0, 3)}')
    axs[1].bar(effect_size_f_dist[1][:-1], effect_size_f_dist[0]/effect_size_0_dist[0].sum(), width=(binrange[-1]-binrange[0])/nbins)
    axs[1].set_title('evolved population\n skew='+f'{round(skew_f, 3)}')
    for ax in axs:
        ax.set_ylim([0, 1])
        ax.set_xlabel('$\Delta$F through single mut.')
        ax.set_ylabel('relative frequency')
    fig.tight_layout()

    if show_plots==True:
        plt.show()