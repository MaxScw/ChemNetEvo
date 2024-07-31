import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from getGraph_fromStochMatrix import getGraph
import networkx as nx
import model as evo

ecoli_stoch = pd.read_csv('Ecoli_centralcarbon_stoichiometry.csv')
ecoli_stochArr = ecoli_stoch.to_numpy()
graph = getGraph(ecoli_stochArr)

fig, ax = plt.subplots(1)

graph, vertex_labels = getGraph(ecoli_stochArr, add_reservoir=True)
pos = nx.kamada_kawai_layout(graph)
nx.draw(
        graph, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in graph.nodes()},
        connectionstyle='arc3, rad = 0.1', ax=ax
        )
nx.draw_networkx_edge_labels(
                                graph, pos,
                                edge_labels=vertex_labels,
                                font_color='red',
                                connectionstyle='arc3, rad = 0.1', ax=ax
                            )
fig.set_size_inches(10, 20)
plt.axis('off')
plt.show()

in_hist = []
out_hist = []

for reaction in range(ecoli_stochArr.shape[1]):
    n_in = len(ecoli_stochArr[ecoli_stochArr[:, reaction]==-1, reaction])
    n_out = len(ecoli_stochArr[ecoli_stochArr[:, reaction]==1, reaction])
    in_hist.append(n_in)
    out_hist.append(n_out)


un_out = np.unique(out_hist, return_counts=True)
out_probs = un_out[1]/sum(un_out[1])
un_in = np.unique(in_hist, return_counts=True)
in_probs = un_in[1]/sum(un_in[1])

print(out_probs, in_probs)

ecoli_smat = evo.get_sMatrix(ecoli_stochArr)[1]
plt.imshow(ecoli_smat)
print(ecoli_smat.shape)
plt.show()

np.save('ecoli_sMat',ecoli_smat)
