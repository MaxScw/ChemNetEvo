import model as evo
import numpy as np

def get_effect_size_dist(network, target_smat):
    n_chem = network.shape[0]
    n_react = network.shape[1]
    
    smat = evo.get_sMatrix(network)[1]
    smat_fit = evo.get_fitness_pertRespSmat(smat[:n_chem, :n_react], 
                                            target_mat=target_smat)
    
    mutant_collection = []
    effect_sizes = []
    vals = [-1, 0, 1]
    mut_mat = []

    for i in range(n_chem):
        for j in range(n_react):
            curr_val = network[i, j]

            for val in vals:
                if val!=curr_val:
                    mut_net = network.copy()
                    mut_net[i, j] = val
                    mutant_collection.append(mut_net)
                    
                    mut_smat = evo.get_sMatrix(mut_net)

                    """
                    # check for diverging result
                    res_hist = []
                    count = 0
                    comp_results = False
                    
                    while comp_results==False:
                        count += 1
                        if count>=1000:
                            comp_results=True

                        mut_smat = evo.get_sMatrix(mut_net)
                        res_hist.append(mut_smat)
                        if len(res_hist)>1:
                            if res_hist[-1][0]!=res_hist[-2][0]:
                                print('difference in invertibility')
                                print(res_hist[-1][2])
                                print(np.linalg.cond(res_hist[-1][2]))
                                print(res_hist[-2][2])
                                print(np.linalg.cond(res_hist[-2][2]))
                            elif res_hist[-1][0]==False and res_hist[-2][0]==False: 
                                if res_hist[-1][1].shape!=res_hist[-2][1].shape:
                                    print('difference in s-matrix')
                    """
                    
                    if mut_smat[0]==True:
                        mut_smat_fit = -1
                    else:
                        mut_smat_fit = evo.get_fitness_pertRespSmat(mut_smat[1][:n_chem, :n_react], 
                                                target_mat=target_smat)
                    effect_sizes.append(mut_smat_fit-smat_fit)
                    mut_mat.append(mut_smat[1])

    return effect_sizes