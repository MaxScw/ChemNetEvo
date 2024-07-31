import numpy as np
from matplotlib import pyplot as plt
import model as evo
import getGraph_fromStochMatrix as graphtool

vmat = np.array([1, -1, 0, 0, 1,
                 0, 1, -1, 0, 0,
                 0, 0, +1, -1, -1]).reshape((3, 5))

# vmat_noLoop = np.array([1, -1, 0, 0, 0,
#                         0, 1, -1, 0, 1,
#                         0, 0, +1, -1, -1]).reshape((3, 5))

vmat_noLoop = np.array([1, -1, 0, 0, 0,
                 0, 1, -1, 0, 0,
                 0, 0, +1, -1, -1]).reshape((3, 5))

graph, vertex_labels = graphtool.getGraph(vmat, add_reservoir=True)
graph_noLoop, vertex_labels_noLoop = graphtool.getGraph(vmat_noLoop, add_reservoir=True)

#pos = evo.nx.spring_layout(graph, k = 1)
#pos_noLoop = evo.nx.spring_layout(graph_noLoop, k = 1)
pos = evo.nx.spectral_layout(graph)
pos_noLoop = evo.nx.spectral_layout(graph_noLoop)
vertex_labelss = [vertex_labels, vertex_labels_noLoop]
poss = [pos, pos_noLoop]
vmats = [vmat, vmat_noLoop]
graphs = [graph, graph_noLoop]

fig, axs = plt.subplots(1, 2)
fig.set_size_inches(14, 7)

for ind, ax in enumerate(axs):
    evo.nx.draw(
                graphs[ind], poss[ind], edge_color='black', width=1, linewidths=1,
                node_size=600, node_color='pink', alpha=0.9,
                labels={node: node for node in graphs[ind].nodes()},
                connectionstyle='arc3, rad = 0', ax=ax
            )
    evo.nx.draw_networkx_edge_labels(
                                    graphs[ind], poss[ind],
                                    edge_labels=vertex_labelss[ind],
                                    font_color='blue',
                                    connectionstyle='arc3, rad = 0',
                                    ax=ax
                                    )
    smat = evo.get_sMatrix(vmats[ind])
    print(smat[1])
    f_p = evo.get_fitness_p(smat[1])
    f_cov, contr_l1, contr_l2 = evo.get_fitness_manualCov(smat[1])
    p00 = smat[1][smat[1]==0].size/smat[1].size
    p11 = smat[1][smat[1]==1].size/smat[1].size
    pneg11 = smat[1][smat[1]==-1].size/smat[1].size
    ax.set_title('$F_{cov}$='+f'{round(f_cov, 3)}, '+'$F_p$='+f'{round(f_p, 3)}'+
                 '\n $p_{00}$='+f'{p00} '+'$p_{11}$='+f'{p11}'+' $p_{-1-1}$='+f'{pneg11}')