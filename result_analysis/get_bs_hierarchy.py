import networkx as nx
import numpy as np

def get_bs_hierarchy(smat, draw=False):
    bs_list = []
    n_chem = smat.shape[0]
    n_react = smat.shape[1]

    for reac in range(n_react):
        chem_resp = list(np.argwhere(smat[:, reac]==1).flatten())
        if len(chem_resp)==0:
            continue
        else:
            bs_tuple = [chem_resp, [reac]]
            for reac_next in range(n_react-(reac+1)):
                reac_next += reac+1
                
                chem_resp_next = list(np.argwhere(smat[:, reac_next]==1).flatten())
                if len(chem_resp)==len(chem_resp_next):
                    if (chem_resp_next == chem_resp):
                        bs_tuple[1].append(reac_next)

            bs_list.append(bs_tuple)

    bs_list=sorted(bs_list, key=lambda x: len(x[0])+len(x[1]))[::-1]

    bs_hir_graph = nx.DiGraph()
    for bs_ind in range(len(bs_list)):
        bs_hir_graph.add_node(bs_ind, content={'chemicals':bs_list[bs_ind][0], 'reactions':bs_list[bs_ind][1]})
        for sub_bs_ind in range(len(bs_list)-(bs_ind+1)):
            sub_bs_ind += bs_ind+1
            if set(bs_list[sub_bs_ind][0]).issubset(bs_list[bs_ind][0]):
                #if set(bs_list[sub_bs_ind][1]).issubset(bs_list[bs_ind][1]):
                bs_hir_graph.add_node(sub_bs_ind, content={'chemicals':bs_list[sub_bs_ind][0], 'reactions':bs_list[sub_bs_ind][1]})
                intermediate_found = False
                for subsub_bs_ind in range(len(bs_list)):
                    if (sub_bs_ind!=subsub_bs_ind)&(bs_ind!=subsub_bs_ind):
                        if (set(bs_list[subsub_bs_ind][0]).issubset(bs_list[bs_ind][0]) & set(bs_list[sub_bs_ind][0]).issubset(bs_list[subsub_bs_ind][0])):
                            if bs_hir_graph.has_edge(bs_ind, subsub_bs_ind):
                                intermediate_found = True
                if intermediate_found == False:
                    bs_hir_graph.add_edge(bs_ind, sub_bs_ind)


    labels = nx.get_node_attributes(bs_hir_graph, 'content')
    
    nodenames = {n:f'c:{labels[i]["chemicals"]} \n r:{labels[i]["reactions"]}' for i, n in enumerate(bs_hir_graph.nodes())}

    if draw==True:
        pos = nx.planar_layout(bs_hir_graph)
        nx.draw(bs_hir_graph,node_size=1000, pos=pos, with_labels=False)
        nx.draw_networkx_labels(bs_hir_graph, pos=pos, labels=nodenames)
    return bs_hir_graph, nodenames
    