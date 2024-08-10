import numpy as np
import model as evo
from tqdm import tqdm
from load_evo_results import load_evo_results
from matplotlib import pyplot as plt
import getGraph_fromStochMatrix as graphtool
import matplotlib
import argparse

n_pop = 100
n_chem = 28
n_react = 50
n_epochs = 300
appendix = 'fullRand'
save_appendix = ''
fpath = ''#'c10_r20_n3_high0p/'
include_connectivity_contribution = False

dirichAlpha = evo.np.ones(n_react)
resp_p = 0.5

pert = evo.np.full(n_react, 1/n_react)
resp = evo.np.full(n_chem, 0.5)

target_smat = evo.np.load('ecoli_sMat.npy')[:n_chem, :n_react]
# target_smat = evo.np.zeros((n_chem, n_react))
# target_smat[:, 0] = evo.np.array([1, 1, 1])
# target_smat[0, 1] = 1
# target_smat[1, 2] = 1
# target_smat[:, 3] = evo.np.array([1, 1, 1])
# target_smat[0, 4] = 1
# target_smat[1, 4] = 1

target_smat_mutP = 0/(n_chem*n_react)

run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}'
init_pop_string = fpath + 'init_pop' + run_string

init_pop = np.load(init_pop_string+'.npy')
singular, sMatrix, aRank = evo.get_vector_sMatrix(init_pop)

# random target fitness, covariance fitness, overlap prob. fitness
fit_names = ['$S^U_{target} fitness', '$v_p$-$v_t$ fitness', 
             '$rand. v_p$-$v_t$ fitness']
n_fit = 3
cmap = matplotlib.colormaps.get_cmap('viridis')
cs = [cmap((i+1)/n_fit) for i in range(n_fit)]

all_fitness_hist = np.empty((n_pop, n_fit), dtype=float)
p1_hist = 0
for pop_ind in tqdm(range(n_pop)):
    
    if singular[pop_ind]:
    
        all_fitness_hist[pop_ind, :] = np.full(n_fit, -1000)
   
    else:
        # plt.imshow(sMatrix[pop_ind])
        # plt.show()
        p1_hist += (sMatrix[pop_ind][sMatrix[pop_ind]==1].size/
                    sMatrix[pop_ind].size)
        
        g = graphtool.getGraph(init_pop[:, :, pop_ind])[0]
        # p_fitness = evo.get_fitness_p(sMatrix[pop_ind][:n_chem, :n_react])
        # manualCov_fitness = evo.get_fitness_manualCov(sMatrix[pop_ind][:n_chem, :n_react])
        randSmat_fitness = evo.get_fitness_pertRespRandSmat(sMatrix[pop_ind][:n_chem, :n_react],
                                                        target_mat=target_smat,
                                                        target_mat_p=target_smat_mutP)[0]
        prDirich_fitness = evo.get_fitness_pertRespDirich(sMatrix[pop_ind][:n_chem, :n_react],
                                                          dirichAlpha, resp_p)
        pr_fitness = evo.get_fitness_pertResp(sMatrix[pop_ind][:n_chem, :n_react],
                                              pert=pert, resp=resp)
        conn_fit = evo.get_fitness_conn(g)

        if include_connectivity_contribution==True:
            p_fitness = p_fitness*conn_fit
            fixed_fitness = fixed_fitness*conn_fit
            manualCov_fitness = manualCov_fitness*conn_fit
        
        all_fitness_hist[pop_ind, :] = np.array([randSmat_fitness, pr_fitness,
                                                 prDirich_fitness])
p1_hist = p1_hist/n_pop
print(p1_hist)
initFit_fig, initFig_ax = plt.subplots(1, n_fit)
nbins = 100
binrange = [0, 1]
bin_width = (binrange[-1]-binrange[0])/nbins

for fit_ind, ax in enumerate(initFig_ax):
    if fit_ind == 0:
        ax.axvline(x=1-p1_hist**2, c='red', ls='--', label='1-$p_1^2$')
        
    
    histo = np.histogram(all_fitness_hist[:, fit_ind],
                         bins=nbins, range=binrange)
    bars = histo[1][:-1] + bin_width/2
    ax.bar(bars, histo[0]/sum(histo[0]), fc=cs[fit_ind], width=bin_width)
    ax.axvline(np.mean(all_fitness_hist[:, fit_ind][all_fitness_hist[:, fit_ind]>=0]), c=cs[fit_ind],
               label = f'<{fit_names[fit_ind]}>={np.mean(all_fitness_hist[:, fit_ind][all_fitness_hist[:, fit_ind]>=0])}')
    ax.set_title(fit_names[fit_ind]+f', N={n_pop} networks')
    ax.set_xlabel('fitness (a.u.)')
    ax.set_ylabel('absolute frequ. in initial pop.')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()

initFit_fig.set_size_inches(n_fit*5, 5)
initFit_fig.tight_layout()
initFit_fig.savefig(f'init_pop_fitDist'+run_string+'_'+save_appendix+'.pdf', 
                    format='pdf')
plt.show()


out_deg_chem_popAv = []
out_deg_react_popAv = []
in_deg_chem_popAv = []
in_deg_react_popAv = []
connected_comp_popAv = []
n_cycle_average = 0

for pop_ind in tqdm(range(n_pop)):
    graph = graphtool.getGraph(init_pop[:, :, pop_ind])[0]

    out_deg_chem = sorted(graph.out_degree, key=lambda x: x[0][-1], reverse=True)[n_react:]
    out_deg_react = sorted(graph.out_degree, key=lambda x: x[0][-1], reverse=True)[:n_react]

    in_deg_chem = sorted(graph.in_degree, key=lambda x: x[0][-1], reverse=True)[n_react:]
    in_deg_react = sorted(graph.in_degree, key=lambda x: x[0][-1], reverse=True)[:n_react]

    connected_comp = sorted(evo.nx.connected_components(graph.to_undirected()),
                            key=len, reverse=True)
    n_cycles = len(sorted(evo.nx.simple_cycles(graph)))

    out_deg_chem_popAv.append([tup[1] for tup in out_deg_chem])
    out_deg_react_popAv.append([tup[1] for tup in out_deg_react])
    in_deg_chem_popAv.append([tup[1] for tup in in_deg_chem])
    in_deg_react_popAv.append([tup[1] for tup in in_deg_react])
    connected_comp_popAv += [len(c) for c in connected_comp]
    n_cycle_average += n_cycles
n_cycle_average = n_cycle_average/n_pop
print(f'networks have on average {n_cycle_average} elementary cycles')

fig, axs = plt.subplots(1)
fig.set_size_inches(5, 5)
un_out_deg_chem = np.unique(out_deg_chem_popAv, return_counts=True)
un_in_deg_chem = np.unique(in_deg_chem_popAv, return_counts=True)

# axs[0].bar(un_out_deg_chem[0], un_out_deg_chem[1]/sum(un_out_deg_chem[1]),
#             label='out degre')
# axs[0].bar(un_in_deg_chem[0], un_in_deg_chem[1]/sum(un_in_deg_chem[1]),
#             alpha=0.5,label='in degree')
# axs[0].set_xticks(np.arange(n_react+1)[::])
# axs[0].set_xlabel('number of nodes')
# axs[0].set_ylabel('relative frequency')
# axs[0].set_xlim([-1, n_react+1])
# axs[0].legend()
# axs[0].set_title('chem. node degree distr.')

# un_out_deg_react = np.unique(out_deg_react_popAv, return_counts=True)
# un_in_deg_react = np.unique(in_deg_react_popAv, return_counts=True)
# axs[1].bar(un_out_deg_react[0], un_out_deg_react[1]/sum(un_out_deg_react[1]),
#             label='out degre')
# axs[1].bar(un_in_deg_react[0], un_in_deg_react[1]/sum(un_in_deg_react[1]),
#             alpha=0.5,label='in degree')
# axs[1].set_xticks(np.arange(n_chem+1)[::])
# axs[1].set_xlabel('number of nodes')
# axs[1].set_ylabel('relative frequency')
# axs[1].set_xlim([-1, n_chem+1])
# axs[1].set_title('reaction node degree distr.')
# axs[1].legend()

#print(np.unique(connected_comp_popAv, return_counts=True))
conn_comp = np.unique(connected_comp_popAv, return_counts=True)

#axs[2].bar(conn_comp[0], conn_comp[0]/(n_react+n_chem))
axs.bar(conn_comp[0], conn_comp[1]/sum(conn_comp[1]))
axs.set_title('dist. of connected comonents')
axs.set_xticks(np.arange(n_chem+n_react+1)[::])
axs.set_xlim([-1, n_chem+1])
axs.set_xlabel('number of connected nodes')
axs.set_ylabel('relative frequency')

example_pop_inds = [3, 10, 40, 80]
netFig, netAx = plt.subplots(1, 4)
netFig.tight_layout()
netFig.set_size_inches(15, 5)
for ax_ind, ax in enumerate(netAx):
    graph, vertex_labels = graphtool.getGraph(init_pop[:, :, ax_ind])
    
    connected_comp = sorted(evo.nx.connected_components(graph.to_undirected()),
                            key=len, reverse=True)
    #print(connected_comp)
    cycles = sorted(evo.nx.simple_cycles(graph))
    
    # top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    # evo.nx.draw(graph, pos=evo.nx.bipartite_layout(graph, nodes=top_nodes, scale=1000),
    #         width=1, with_labels=True, ax=ax)
    #nx.draw_circular(graph, ax=ax, with_labels=True)
    #ax.imshow(init_pop[:, :, ax_ind, ep])
    pos = evo.nx.spring_layout(graph, k = 5)
    evo.nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()},
            connectionstyle='arc3, rad = 0.1',
            ax=ax
           )
    evo.nx.draw_networkx_edge_labels(
                                 graph, pos,
                                 edge_labels=vertex_labels,
                                 font_color='blue',
                                 connectionstyle='arc3, rad = 0.1',
                                 ax=ax
                                )
    #plt.axis('off')
    
plt.show()

fig.savefig(f'networkProp'+run_string+'_'+save_appendix+'.pdf', format='pdf')
netFig.savefig(f'networkExample'+run_string+'_'+save_appendix+'.pdf', format='pdf')