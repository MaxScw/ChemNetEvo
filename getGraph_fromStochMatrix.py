import networkx as nx
import numpy as np
#import model as evo
from tqdm import tqdm
from load_evo_results import load_evo_results
from matplotlib import pyplot as plt
import matplotlib

def getGraph_bip(mat):
    graph = nx.DiGraph()
    chem_inds = np.array([str(n_chem+1)+'c' for n_chem in np.arange(mat.shape[0])])
    react_inds = np.array([str(n_react+1)+'r' for n_react in np.arange(mat.shape[1])])
    graph.add_nodes_from(chem_inds, bipartite=0)
    graph.add_nodes_from(react_inds, bipartite=1)

    rOutput_edge_list = np.argwhere(mat==1)[:, ::-1]
    rOutput_edge_str = np.empty(rOutput_edge_list.shape, dtype='U10')
    rOutput_edge_str[:, 0] = react_inds[rOutput_edge_list[:, 0]]
    rOutput_edge_str[:, 1] = chem_inds[rOutput_edge_list[:, 1]]
   
    rInput_edge_list = np.argwhere(mat==-1)
    rInput_edge_str = np.empty(rInput_edge_list.shape, dtype='U10')
    rInput_edge_str[:, 0] = chem_inds[rInput_edge_list[:, 0]]
    rInput_edge_str[:, 1] = react_inds[rInput_edge_list[:, 1]]
    
    graph.add_edges_from(rInput_edge_str)
    graph.add_edges_from(rOutput_edge_str)
    return graph

def getGraph(mat, add_reservoir=False):
    graph = nx.MultiDiGraph()
    chem_inds = np.array([str(n_chem+1)+'c' for n_chem in 
                          np.arange(mat.shape[0])])
    react_inds = np.array([str(n_react+1)+'r' for n_react in 
                           np.arange(mat.shape[1])])
    graph.add_nodes_from(chem_inds)
    vertex_labels = dict()

    rOutput_edge_list = np.argwhere(mat==1)
    rInput_edge_list = np.argwhere(mat==-1)

    for inp in rInput_edge_list:
        outp_found = 0
        for out in rOutput_edge_list:
            if out[-1]==inp[-1]:
                outp_found = 1
                graph.add_edge(chem_inds[inp[0]], chem_inds[out[0]])
                if (chem_inds[inp[0]], chem_inds[out[0]]) in vertex_labels.keys():
                    val = vertex_labels[(chem_inds[inp[0]], chem_inds[out[0]])]
                    vertex_labels[(chem_inds[inp[0]], chem_inds[out[0]])] = (
                        val + ', ' + react_inds[out[-1]])
                else:
                    vertex_labels[(chem_inds[inp[0]], chem_inds[out[0]])] = (
                        react_inds[out[-1]])
               
        if (outp_found == 0) and (add_reservoir==True):
            graph.add_node(f'out-{react_inds[inp[-1]]}')
            graph.add_edge(chem_inds[inp[0]], f'out-{react_inds[inp[-1]]}')
            vertex_labels[(chem_inds[inp[0]], f'out-{react_inds[inp[-1]]}')] = (
                react_inds[inp[-1]])
            
    if add_reservoir==True:
        for out in rOutput_edge_list:
            inp_found = 0
            for inp in rInput_edge_list:
                if out[-1]==inp[-1]:
                    inp_found = 1
            if inp_found==0:
                graph.add_node(f'in-{react_inds[out[-1]]}')
                graph.add_edge(f'in-{react_inds[out[-1]]}', chem_inds[out[0]])
                vertex_labels[(f'in-{react_inds[out[-1]]}', chem_inds[out[0]])] = (
                    react_inds[out[-1]])

      
    return graph, vertex_labels
    
if __name__=='__main__':
    n_pop = 1000
    n_chem = 10
    n_react = 20
    n_epochs = 2000
    save_ep = 200
    appendix = '_t0p001_mut1_run1'

    path = 'c10_r15_n3_low0p/'
    save_appendix = 'mut2000'

    run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}'
    evo_results = load_evo_results(n_pop=n_pop,
                                n_chem=n_chem,
                                n_react=n_react,
                                n_epochs=n_epochs,
                                appendix=appendix,
                                path=path)
    mut_pop = evo_results[0]
    mut_fit = evo_results[1]
    noMut_pop = evo_results[2]
    noMut_fit = evo_results[3]

    example = mut_pop[:, :, 10, -1]

    graph, vertex_labels = getGraph(example)
    #print(vertex_labels)
    pos = nx.spring_layout(graph, k = 10)
    nx.draw(
            graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9,
            labels={node: node for node in graph.nodes()},
            connectionstyle='arc3, rad = 0.1'
           )
    nx.draw_networkx_edge_labels(
                                 graph, pos,
                                 edge_labels=vertex_labels,
                                 font_color='red',
                                 connectionstyle='arc3, rad = 0.1'
                                )
    plt.axis('off')
    plt.show()
"""
    out_deg_chem_popAv = []
    out_deg_react_popAv = []
    in_deg_chem_popAv = []
    in_deg_react_popAv = []
    connected_comp_popAv = []
    ep = 5
    # ind = 2
    # plt.imshow(mut_pop[:, :, ind, ep])

    # graph = getGraph(mut_pop[:, :, ind, ep])
    # print(graph)
    # top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
    # bottom_nodes = set(graph) - top_nodes

    # nx.draw_networkx(
    # graph,
    # pos = nx.drawing.layout.bipartite_layout(graph, top_nodes))
    # plt.show()
    # nx.draw(graph.to_undirected(graph))
    # plt.show()

    # connected_comp = sorted(nx.connected_components(graph.to_undirected()),
    #                             key=len, reverse=True)
    # print(connected_comp)

    # out_deg_chem = sorted(graph.out_degree, key=lambda x: x[0][-1], reverse=True)[:n_react]
    # print(out_deg_chem)
    for pop_ind in tqdm(range(n_pop)):
        graph = getGraph(mut_pop[:, :, pop_ind, ep])

        out_deg_chem = sorted(graph.out_degree, key=lambda x: x[0][-1], reverse=True)[n_react:]
        out_deg_react = sorted(graph.out_degree, key=lambda x: x[0][-1], reverse=True)[:n_react]

        in_deg_chem = sorted(graph.in_degree, key=lambda x: x[0][-1], reverse=True)[n_react:]
        in_deg_react = sorted(graph.in_degree, key=lambda x: x[0][-1], reverse=True)[:n_react]

        connected_comp = sorted(nx.connected_components(graph.to_undirected()),
                                key=len, reverse=True)
        
        out_deg_chem_popAv.append([tup[1] for tup in out_deg_chem])
        out_deg_react_popAv.append([tup[1] for tup in out_deg_react])
        in_deg_chem_popAv.append([tup[1] for tup in in_deg_chem])
        in_deg_react_popAv.append([tup[1] for tup in in_deg_react])
        connected_comp_popAv += [len(c) for c in connected_comp]


    fig, axs = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)
    un_out_deg_chem = np.unique(out_deg_chem_popAv, return_counts=True)
    un_in_deg_chem = np.unique(in_deg_chem_popAv, return_counts=True)

    axs[0].bar(un_out_deg_chem[0], un_out_deg_chem[1]/sum(un_out_deg_chem[1]),
               label='out degre')
    axs[0].bar(un_in_deg_chem[0], un_in_deg_chem[1]/sum(un_in_deg_chem[1]),
               alpha=0.5,label='in degree')
    axs[0].set_xticks(np.arange(n_react+1)[::5])
    axs[0].set_xlim([-1, n_react+1])
    axs[0].legend()
    axs[0].set_title('chem. node degree distr.')

    un_out_deg_react = np.unique(out_deg_react_popAv, return_counts=True)
    un_in_deg_react = np.unique(in_deg_react_popAv, return_counts=True)
    axs[1].bar(un_out_deg_react[0], un_out_deg_react[1]/sum(un_out_deg_react[1]),
               label='out degre')
    axs[1].bar(un_in_deg_react[0], un_in_deg_react[1]/sum(un_in_deg_react[1]),
               alpha=0.5,label='in degree')
    axs[1].set_xticks(np.arange(n_chem+1)[::5])
    axs[1].set_xlim([-1, n_chem+1])
    axs[1].set_title('reaction node degree distr.')
    axs[1].legend()
    
    #print(np.unique(connected_comp_popAv, return_counts=True))
    conn_comp = np.unique(connected_comp_popAv, return_counts=True)
    
    #axs[2].bar(conn_comp[0], conn_comp[0]/(n_react+n_chem))
    axs[2].bar(conn_comp[0], conn_comp[1]/sum(conn_comp[1]))
    axs[2].set_title('dist. of connected comonents')
    axs[2].set_xticks(np.arange(n_chem+n_react+1)[::10])
    axs[2].set_xlim([-1, n_chem+n_react+1])

    example_pop_inds = [3, 10, 40, 80]
    netFig, netAx = plt.subplots(1, 4)
    netFig.tight_layout()
    netFig.set_size_inches(15, 5)


    un_pop = np.unique(mut_pop[:, :, :, ep], axis=2)

    for ax_ind, ax in enumerate(netAx):
        if ax_ind<un_pop.shape[-1]:
            graph = getGraph(un_pop[:, :, ax_ind])
        
            #print(evo.get_fitness(mut_pop[:, :, ax_ind, ep]))
            connected_comp = sorted(nx.connected_components(graph.to_undirected()),
                                    key=len, reverse=True)
            
            #print([len(connected_comp[i])for i in range(len(connected_comp))])

            #top_nodes = {n for n, d in graph.nodes(data=True) if d["bipartite"] == 0}
            #nx.draw(graph, pos=nx.bipartite_layout(graph, nodes=top_nodes, scale=1000),
            #        width=1, with_labels=True, ax=ax)
            #nx.draw(graph, ax=ax, with_labels=True)
            print(un_pop[:, :, ax_ind])
            ax.imshow(evo.get_sMatrix(un_pop[:, :, ax_ind])[1])
            print(evo.get_fitness_p(evo.get_sMatrix(un_pop[:, :, ax_ind])[1]))
            #ax.imshow(mut_pop[:, :, ax_ind, ep])

    fig.savefig(f'networkProp'+run_string+'_'+save_appendix+'.pdf', format='pdf')
    netFig.savefig(f'networkExample'+run_string+'_'+save_appendix+'.pdf', format='pdf')
    plt.show()
"""
