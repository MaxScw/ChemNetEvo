import numpy as np
import model as evo
from tqdm import tqdm
from load_evo_results import load_evo_results
from matplotlib import pyplot as plt
import getGraph_fromStochMatrix as graphtool
import matplotlib
import argparse

n_pop = 10
n_chem = 3
n_react = 5
n_epochs = 10000
appendix = '_t0p001_mut1_run1_fixed'
save_appendix = ''
fpath = 'SmatDist_fullCorr/'#'c10_r20_n3_high0p/'
include_connectivity_contribution = False

dirichAlpha = evo.np.ones(n_react)
resp_p = 0.5

pert = evo.np.full(n_react, 1/n_react)
resp = evo.np.full(n_chem, 0.5)

target_smat = evo.np.zeros((n_chem, n_react))
target_smat[:, 0] = evo.np.array([1, 1, 1])
target_smat[0, 1] = 1
target_smat[1, 2] = 1
target_smat[:, 3] = evo.np.array([1, 1, 1])
target_smat[0, 4] = 1
target_smat[1, 4] = 1

run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}'
init_pop_string = fpath + 'init_pop' + run_string# + '_fixed'

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

init_pop = np.load(init_pop_string+'.npy')

singular, sMatrix, aRank = evo.get_vector_sMatrix(mut_pop[:, :, :,-1])#(init_pop)

n_fit = 3
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
        pertResp_fitness = evo.get_fitness_pertResp(sMatrix[pop_ind][:n_chem, :n_react],
                                             pert=pert, resp=resp)
        
        pertRespDirich_fitness = evo.get_fitness_pertRespDirich(sMatrix[pop_ind][:n_chem, :n_react],
                                                           alpha=dirichAlpha, resp_p=resp_p)
        pertRespSmat_fitness = evo.get_fitness_pertRespSmat(sMatrix[pop_ind][:n_chem, :n_react],
                                                            target_mat=target_smat)
        conn_fit = evo.get_fitness_conn(g)

        if include_connectivity_contribution==True:
            p_fitness = p_fitness*conn_fit
            manualCov_fitness = manualCov_fitness*conn_fit

        all_fitness_hist[pop_ind, :] = np.array([pertResp_fitness, 
                                                 pertRespDirich_fitness,
                                                 pertRespSmat_fitness])

top10_inds = np.argsort(all_fitness_hist[:, 2])[:]

top10_pop = [sMatrix[ind] for ind in top10_inds]

fig1, axs1 = plt.subplots(2, 5)
fig1.set_size_inches(25, 10)
for ind in range(10):
    if ind<5:
        row = 0
    else:
        row = 1
    # evo.nx.draw(graphtool.getGraph(mut_pop[:, :, 
    #     top10_inds[ind], -1], True)[0], connectionstyle='arc3, rad = 0.1',
    #     ax=axs[row, ind-5])
    axs1[row, ind-5*row].imshow(sMatrix[top10_inds[ind]])#(init_pop[:, :, top10_inds[ind]])
plt.show()



fig, axs = plt.subplots(2, 5)
fig.set_size_inches(25, 10)
for ind in range(10):
    if ind<5:
        row = 0
    else:
        row = 1
    graph, vertex_labels = graphtool.getGraph(mut_pop[:, :, 
        top10_inds[ind], -1], True)
    pos = evo.nx.planar_layout(graph)
    evo.nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()},
            connectionstyle='arc3, rad = 0.1',
            ax=axs[row, ind-5*row]
           )
    evo.nx.draw_networkx_edge_labels(
                                 graph, pos,
                                 edge_labels=vertex_labels,
                                 font_color='blue',
                                 connectionstyle='arc3, rad = 0.1',
                                 ax=axs[row, ind-5*row]
                                )
    #axs[row, ind-5].imshow(sMatrix[top10_inds[ind]])
plt.show()