#%%
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

plot_fitnessDistComp = True
topPerformingNets = False
bsSizeDist = False
bsHierarchy = False
effSizeDist = False
effSize_startAndFinish = False
effSize_onlyRef = True
fit2dHist = False

fps = 0
bps = 0

if os.path.isfile('maxMode.txt'):
    os.remove('maxMode.txt')
if os.path.isfile('symmDev.txt'):
    os.remove('symmDev.txt')


# run_names = ['fullCorr', 'fp1e5_bp21e5', 'fp1e4_bp21e4',
#              'fp1e3_bp21e3', 'fp1e2_bp21e2', 'fp1e1_bp21e1']


run_names = ['_fp0_bp0']#['_fp0.1_bp0.2', '_PR_fp0.12_bp0.18', '_PR_fp0.15_bp0.15',
             #'_PR_fp0.17_bp0.13', '_PR_fp0.2_bp0.1']#['_fp0_bp0', '_fp0.001_bp0.002', '_fp0.01_bp0.02', 
             #'_fp0.05_bp0.1', '_fp0.1_bp0.2', '_fp0.5_bp1.0']#['fullCorr', 'fp1e5_bp1e5', 'fp51e5_bp51e5', 'fp7p51e5_bp7p51e5', 
             #'fp1e4_bp1e4', 'fp1e3_bp1e3', 'fp1e2_bp1e2', 'fp1e1_bp1e1']

buff_hist_runs = []
cycle_hist_runs = []
hier_conn_comp_runs = []
hier_longest_path_runs = []

for run_name in run_names:
    run_fitness = []
    ref_fit = []
    top_inds = []
    best_net = []
    top_nets = []
    sMatrix_pops = []
    sMatrix_sing = []
    n_cycles = []
    n_buff = []
    conn_comp_lengths = []
    longest_paths = []
    high_fit = []
    high_final_fit = []
    fit_threshold = 0.99
    nruns = 1


    #target_smat = np.load(f"target_sMat_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}.npy")
    target_smat = evo.np.zeros((n_chem, n_react))
    target_smat[:, 0] = evo.np.array([1, 1, 1])
    target_smat[0, 1] = 1
    target_smat[1, 2] = 1
    target_smat[:, 3] = evo.np.array([1, 1, 1])
    target_smat[0, 4] = 1
    target_smat[1, 4] = 1

    pert = np.zeros(n_react)
    pert[0] = 1/2
    pert[4] = 1/2

    resp = np.zeros(n_chem)
    resp[0] = 1

    filtered_networks = []

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

        singular, sMatrix, aRank = evo.get_vector_sMatrix(mut_pop[:, :, :,-1])
        singular0, sMatrix0, aRank0 = evo.get_vector_sMatrix(mut_pop[:, :, :,0])

        for pop_ind in tqdm(range(n_pop)):
            if singular[pop_ind]==False:
                ref_fit.append(round(evo.get_fitness_pertRespSmat(
                                sMatrix[pop_ind][:n_chem, :n_react],
                                target_mat=target_smat), 8))   
                # ref_fit.append(round(evo.get_fitness_pertResp(
                #                 sMatrix[pop_ind][:n_chem, :n_react],
                #                 pert=pert, resp=resp), 8))   
                
                n_buff.append(np.unique(sMatrix[pop_ind][:n_chem, :n_react],
                                  axis=1).shape[1])
                
                graph, nodenames = get_bs_hierarchy(sMatrix[pop_ind][:n_chem, :n_react])
                longest_path = evo.nx.dag_longest_path_length(graph)

                graph, vertex_labels = graphtool.getGraph(mut_pop[:, :, pop_ind,-1], True)
                
                n_cycles.append(len(evo.nx.recursive_simple_cycles(graph)))

                longest_paths.append(longest_path)

                if ref_fit[-1]>fit_threshold:
                    filtered_networks.append(mut_pop[:, :, pop_ind,-1])
                    high_final_fit = high_final_fit + get_effect_size_dist(mut_pop[:, :, pop_ind,-1],
                                                                           target_smat)

            else:
                ref_fit.append(-1)

            if singular0[pop_ind]==False and mut_fit[pop_ind, 0]>fit_threshold:
                high_fit = high_fit + get_effect_size_dist(mut_pop[:, :, pop_ind,0],
                                                           target_smat)
        
        

        sMatrix_pops.append(sMatrix)
        sMatrix_sing.append(singular)
        # print(np.unique(mut_pop[:, :, :, -1], axis=2).shape)
        # print(f'diversity = {np.unique(mut_pop[:, :, :, -1], axis=2).shape[-1]}')
        # print(len(np.unique(mut_fit[:, -1])))
        #print(np.sort(mut_fit[:, -1])[-1])
        top_inds.append(np.argsort(mut_fit[:, -1])[-1])
        
        run_fitness.append(mut_fit)
        
        top_nets.append(mut_pop[:, :, top_inds[-1], -1])
        #print(evo.get_sMatrix(top_nets[-1])[1])
        # print(ref_fit[i*n_pop + top_inds[-1]])
        # print(round(evo.get_fitness_pertRespSmat(
        #                         sMatrix[top_inds[-1]][:n_chem, :n_react],
        #                         target_mat=target_smat), 8))
        #print(np.sum(abs(evo.get_sMatrix(top_nets[-1])[1][:n_chem, :n_react] - target_smat)))
    
    unMatrix_count = np.unique(np.array(filtered_networks), axis=0).shape[0]
    cycle_hist_runs.append(np.mean(n_cycles))
    hier_longest_path_runs.append(np.mean(longest_paths))

    # bin the fitness data into histograms for first and final epoch, ref. fitness

    top_inds = np.array(top_inds)

    run_fitness_f = np.array(run_fitness).reshape(nruns*n_pop, n_epochs+1)[:, -2]
    run_fitness_0 = np.array(run_fitness).reshape(nruns*n_pop, n_epochs+1)[:, 0]

    nbins = 30
    binrange = [0.7, 1]
    bin_width = (binrange[-1]-binrange[0])/nbins
    bins = np.linspace(0, nbins-1, nbins)
    bin_vals = np.linspace(binrange[0]+0.5*bin_width, 
                        binrange[1]-0.5*bin_width,
                        nbins)

    mut_fitVals0 = np.histogram(run_fitness_0, bins=nbins, range=binrange)
    mut_fitVals = np.histogram(run_fitness_f, bins=nbins, range=binrange)
    mut_revFitVals = np.histogram(ref_fit, bins=nbins, range=binrange)

    fitDist_2d = (np.stack([mut_fitVals[0] for i in range(nbins)])+
                    np.stack([mut_revFitVals[0] for i in range(nbins)]).T)
    max_n = np.argmax(fitDist_2d)
    max_i = max_n//nbins
    max_j = max_n - nbins*max_i

    fitDist_2d_symmDev = round(np.mean(abs(fitDist_2d - fitDist_2d.T)), 3)

    maxMode_f = open("maxMode.txt", "a")
    maxMode_f.write(f"{bin_vals[max_i]} {bin_vals[max_j]}\n")
    maxMode_f.close()

    symmDev_f = open("symmDev.txt", "a")
    symmDev_f.write(f"{fitDist_2d_symmDev}\n")
    symmDev_f.close()

    # bin the BS data into histograms

    bs_binrange = [1, n_react+1]
    bs_nbins = n_react
    bs_bins = np.linspace(bs_binrange[0], bs_binrange[1]-1, bs_nbins)
    bs_histo = np.histogram(np.array(n_buff), bins=bs_nbins, range=bs_binrange)
    buff_hist_runs.append(bs_histo[0])
#%%
    # bin the fitness of best-performing models for start versus finish

    hf_nbins = 60
    hf_binrange = [-2, 0.25]
    hf_binwidth = (hf_binrange[-1]-hf_binrange[0])/hf_nbins

    
    target_smat = evo.np.zeros((n_chem, n_react))
    target_smat[:, 0] = evo.np.array([1, 1, 1])
    target_smat[0, 1] = 1
    target_smat[1, 2] = 1
    target_smat[:, 3] = evo.np.array([1, 1, 1])
    target_smat[0, 4] = 1
    target_smat[1, 4] = 1

    target_net = np.array([1, -1, 0, 0, 1,
                           0, 1, -1, 0, 0,
                           0, 0, 1, -1, -1]).reshape(3, 5)
    
    high_fit = get_effect_size_dist(target_net, target_smat)
    #print(len(np.unique(high_fit)))
    #print(len(np.unique(high_final_fit)))
    smat = evo.get_sMatrix(target_net)[1]
    smat_fit = evo.get_fitness_pertRespSmat(smat[:n_chem, :n_react], 
                                            target_mat=target_smat)
    
    #print(np.mean(high_fit))
    #print(np.mean(high_final_fit))
    high_fit_dist = np.histogram(high_fit, bins=hf_nbins, range=hf_binrange)
    high_final_fit_dist = np.histogram(high_final_fit, bins=hf_nbins, range=hf_binrange)

    skew_0 = skew(high_fit)
    skew_f = skew(high_final_fit)
#%%
    # plot 2d-histogram of reference versus final target fitness for last epoch

    if fit2dHist==True:
        fitDist_fig, fitDist_ax = plt.subplots()
        fitDist_fig.tight_layout()
        fitDist_fig.set_size_inches(5, 5)
        fitDist_ax.imshow((np.stack([mut_fitVals[0] for i in range(nbins)])+
                        np.stack([mut_revFitVals[0] for i in range(nbins)]).T)/
                        (np.sum(mut_revFitVals[0])+np.sum(mut_fitVals[0])),
                        origin='lower')
        fitDist_ax.axhline(y=max_i, color='red', label='highest mode')
        fitDist_ax.axvline(x=max_j, color='red')
        fitDist_ax.plot(np.linspace(0, nbins-1, nbins), np.linspace(0, nbins-1, nbins),
                        color='blue', ls='dotted')
        tick_freq = nbins//10
        fitDist_ax.set_xlabel('fitness w.r.t. final target (a.u.)')
        fitDist_ax.set_ylabel('fitness w.r.t. start target (a.u.)')
        fitDist_ax.set_xticks(bins[::tick_freq])
        fitDist_ax.tick_params(axis='x', labelrotation=45)
        fitDist_ax.set_yticks(bins[::tick_freq])
        fitDist_ax.set_xticklabels(np.round(np.linspace(binrange[0], 
                                                        binrange[-1], 
                                                        nbins)[::tick_freq], 3))
        fitDist_ax.set_yticklabels(np.round(np.linspace(binrange[0], 
                                                        binrange[-1], 
                                                        nbins)[::tick_freq], 3))
        fitDist_ax.set_title(#'$p_f=10^{-1}$, $p_b=10^{-1}$\n'+#('fully correlated $S^u_{target}$\n'+
                            'dev. from symm. $\delta$='+f'{fitDist_2d_symmDev}')
        fitDist_ax.legend()
        fitDist_fig.tight_layout()
        if save_plots==True:
            fitDist_fig.savefig('combinedFitnessHist_'+run_name+'.pdf', format='pdf')
#%%
    # plot effect size distribution of starting versus final generation
    
    if effSizeDist==True and effSize_startAndFinish==True:
        highFit_fig, highFit_ax = plt.subplots(1, 2)
        highFit_fig.set_size_inches(10, 5)
        highFit_ax[0].bar(high_fit_dist[1][:-1]+0.5*hf_binwidth, 
                        high_fit_dist[0]/high_fit_dist[0].sum(),
                        width=hf_binwidth)
        highFit_ax[0].axvline(x=np.mean(high_fit), c='red',
                              label='$\langle \Delta F \\rangle$='+
                              f'{round(np.mean(high_fit), 3)}')
        highFit_ax[1].bar(high_final_fit_dist[1][:-1]+0.5*hf_binwidth, 
                        high_final_fit_dist[0]/high_final_fit_dist[0].sum(),
                        width=hf_binwidth)
        highFit_ax[1].axvline(x=np.mean(high_final_fit), c='red',
                              label='$\langle \Delta F \\rangle$='+
                              f'{round(np.mean(high_final_fit),3)}')
        
        highFit_ax[0].set_title('target matrix $S^u_0$\n skew='+f'{round(skew_0, 3)}')
        highFit_ax[1].set_title('evolved population\n skew='+f'{round(skew_f, 3)}'+f', {unMatrix_count} un. networks')
        for ax in highFit_ax:
            ax.legend()
            ax.set_ylim([0, 0.5])
            ax.set_xlabel('$\Delta$F through single mut.')
            ax.set_ylabel('relative frequency')
    elif effSizeDist==True:
        highFit_fig, highFit_ax = plt.subplots(1)
        highFit_fig.set_size_inches(5, 5)
        highFit_ax.bar(high_final_fit_dist[1][:-1]+0.5*hf_binwidth, 
                        high_final_fit_dist[0]/high_final_fit_dist[0].sum(),
                        width=hf_binwidth)
        highFit_ax.axvline(x=np.mean(high_final_fit), c='red',
                              label='$\langle \Delta F \\rangle$='+
                              f'{round(np.mean(high_final_fit),3)}')
        highFit_ax.set_title('evolved population\n skew='+f'{round(skew_f, 3)}'+f', {unMatrix_count} un. networks')
        
        highFit_ax.legend()
        highFit_ax.set_ylim([0, 0.5])
        highFit_ax.set_xlabel('$\Delta$F through single mut.')
        highFit_ax.set_ylabel('relative frequency')
    elif effSize_onlyRef==True:
        highFit_fig, highFit_ax = plt.subplots(1)
        highFit_fig.set_size_inches(5, 5)
        highFit_ax.bar(high_fit_dist[1][:-1]+0.5*hf_binwidth, 
                        high_fit_dist[0]/high_fit_dist[0].sum(),
                        width=hf_binwidth)
        highFit_ax.axvline(x=np.mean(high_fit), c='red',
                              label='$\langle \Delta F \\rangle$='+
                              f'{round(np.mean(high_fit), 3)}')
        highFit_ax.set_title('target matrix $S^u_0$\n skew='+f'{round(skew_0, 3)}')
        
        highFit_ax.legend()
        highFit_ax.set_ylim([0, 0.5])
        highFit_ax.set_xlabel('$\Delta$F through single mut.')
        highFit_ax.set_ylabel('relative frequency')
#%%
    # plot starting versus final fitness w.r.t. reference

    if plot_fitnessDistComp == True:
        fitDist_comp_fig, fitDist_comp_ax = plt.subplots()
        fitDist_comp_fig.tight_layout()
        fitDist_comp_fig.set_size_inches(5, 5)
        fitDist_comp_ax.bar(mut_fitVals0[1][:-1]+0.5*bin_width, 
                        mut_fitVals0[0]/np.sum(mut_fitVals0[0]), 
                        width=bin_width, fc='red', label='initial population')
        fitDist_comp_ax.bar(mut_fitVals[1][:-1]+0.5*bin_width, 
                        mut_fitVals[0]/np.sum(mut_fitVals[0]), 
                        width=bin_width, alpha=0.5, fc='green',
                        label='final evolved population')
        fitDist_comp_ax.bar(mut_revFitVals[1][:-1]+0.5*bin_width, 
                        mut_revFitVals[0]/np.sum(mut_revFitVals[0]), 
                        width=bin_width, alpha=0.5, fc='blue',
                        label='reference fitness')
        fitDist_comp_ax.set_xlabel('fitness (a.u.)')
        fitDist_comp_ax.set_ylabel('relative frequency in one generation')
        fitDist_comp_ax.set_xlim(binrange)
        fitDist_comp_ax.set_ylim([0, 1])
        fitDist_comp_ax.legend()
        fitDist_comp_fig.tight_layout()


    # plot top-performing networks for multiple runs

    if topPerformingNets == True:
        top_pop = [sMatrix_pops[run][top_inds[run]]  for run in range(nruns)]

        fig1, axs1 = plt.subplots(2, 5)
        fig1.set_size_inches(25, 25)
        for ind in range(10):
            if ind<5:
                row = 0
            else:
                row = 1
            # evo.nx.draw(graphtool.getGraph(mut_pop[:, :, 
            #     top10_inds[ind], -1], True)[0], connectionstyle='arc3, rad = 0.1',
            #     ax=axs[row, ind-5])
            axs1[row, ind-5*row].set_title('$F_{PRs}=$'+f'{round(run_fitness[ind][top_inds[ind], -1], 3)}\n'+
                                        '$F_{PRs, S_0}=$'+f'{round(ref_fit[ind*n_pop + top_inds[ind]], 3)}')
            #axs1[row, ind-5*row].imshow(abs(top_pop[ind][:n_chem, :n_react]- target_smat), cmap='magma')#(init_pop[:, :, top10_inds[ind]])
            axs1[row, ind-5*row].imshow(top_pop[ind][:n_chem, :n_react], cmap='magma')
            #axs1[row, ind-5].imshow(top_nets[ind])



        fig, axs = plt.subplots(2, 5)
        fig.set_size_inches(60, 30)

        for ind in range(10):
            if ind<5:
                row = 0
            else:
                row = 1
            #graph, nodenames = get_bs_hierarchy(top_nets[ind])
            # conn_comp_length = [len(c) for c in sorted(evo.nx.connected_components(graph.to_undirected()), key=len, reverse=True)]
            # longest_path = evo.nx.dag_longest_path_length(graph)
            
            # pos = evo.nx.planar_layout(graph)
            # evo.nx.draw(graph,node_size=1000, pos=pos, with_labels=False,
            #             ax=axs[row, ind-5*row])
            # evo.nx.draw_networkx_labels(graph, pos=pos, labels=nodenames,
            #                             ax=axs[row, ind-5*row])
            # fig.tight_layout()
            # print(top_nets[ind])
            graph, vertex_labels = graphtool.getGraph(top_nets[ind], True)
            
            pos = evo.nx.kamada_kawai_layout(graph, scale=100)
            evo.nx.draw(
                    graph, pos, edge_color='black', width=1, linewidths=1,
                    node_size=500, node_color='pink', alpha=0.9,
                    labels={node: node for node in graph.nodes()},
                    connectionstyle='arc3, rad = 0.2',
                    ax=axs[row, ind-5*row]
                )

            axs[row, ind-5*row].set_title('$F_{PRs}=$'+f'{round(run_fitness[ind][top_inds[ind], -1], 3)}\n'+
                                        '$F_{PRs, S_0}=$'+f'{round(ref_fit[ind*n_pop + top_inds[ind]], 3)}')

            evo.nx.draw_networkx_edge_labels(
                                        graph, pos,
                                        edge_labels=vertex_labels,
                                        font_color='blue',
                                        connectionstyle='arc3, rad = 0.2',
                                        ax=axs[row, ind-5*row]
                                        )
    


# plot buffering structure dist. over runs as 2d histo

if bsSizeDist==True:
    bs_fig, bs_ax = plt.subplots(1)
    bs_fig.set_size_inches(5, 5)
    bs_histo_runs = np.array(buff_hist_runs)
    bs_ax.imshow(bs_histo_runs)
    bs_ax.set_title('BS size distribution')
    bs_ax.axvline(x=3, c='red', label='init. target')

    bs_ax.set_yticks(np.linspace(0, len(run_names)-1, len(run_names)))
    # bs_ax.set_yticklabels(['$10^{-5}$', '$10^{-4}$',
    #                                '$10^{-3}$', '$10^{-2}$',
    #                                '$10^{-1}$'])
    # bs_ax.set_yticklabels(['0', '$10^{-5}$','$5\cdot10^{-5}$', 
    #                                '$7.5\cdot10^{-5}$','$10^{-4}$',
    #                                '$10^{-3}$', '$10^{-2}$',
    #                                '$10^{-1}$'])
    bs_ax.set_yticklabels(['0', '0.001', '0.01', '0.05', '0.1', '0.5'])
    bs_ax.set_ylabel('$p_{forward}$')
    bs_ax.set_xticks(np.linspace(0, n_react-1, n_react))

    bs_ax.set_xticklabels(np.linspace(0, n_react-1, n_react)+1)
    bs_ax.set_xlabel('# of indep. columns in $S^u$')
    bs_ax.legend()
    bs_fig.savefig('bsSizeDistr.pdf', format='pdf')

# plot BS hierarchy for different runs

if bsHierarchy==True:
    print(hier_longest_path_runs)
    print(cycle_hist_runs)

    bsHier_fig, bsHier_axs = plt.subplots(1, 2)
    bsHier_fig.set_size_inches(10, 5)

    bsHier_axs[0].plot(np.linspace(0, len(run_names)-1, len(run_names)),
                    cycle_hist_runs)
    bsHier_axs[1].plot(np.linspace(0, len(run_names)-1, len(run_names)),
                    hier_longest_path_runs)
    bsHier_axs[0].set_ylabel('# of involved chemicals')
    bsHier_axs[0].set_title('number of cycles in network')

    bsHier_axs[1].set_ylabel('# of buffering structures')
    bsHier_axs[1].set_title('longest path length along BS hierarchy')

    for bsHier_ax in bsHier_axs:
        bsHier_ax.set_xticks(np.linspace(0, len(run_names)-1, len(run_names)))
        # bsHier_ax.set_xticklabels(['$10^{-5}$', '$10^{-4}$',
        #                            '$10^{-3}$', '$10^{-2}$',
        #                            '$10^{-1}$'])
        # bsHier_ax.set_xticklabels(['0', '$10^{-5}$','$5\cdot10^{-5}$', 
        #                            '$7.5\cdot10^{-5}$','$10^{-4}$',
        #                            '$10^{-3}$', '$10^{-2}$',
        #                            '$10^{-1}$'])
        bsHier_ax.set_xticklabels(['0', '0.001', '0.01', '0.05', '0.1', '0.5'])
        bsHier_ax.tick_params(axis='x', labelrotation=45)
        bsHier_ax.set_xlabel('$p_{forward}$')
    bsHier_fig.savefig('bsHier.pdf', format='pdf')

if show_plots==True:
    plt.show()
    
