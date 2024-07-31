import numpy as np
from time import time
from tqdm import tqdm
from getGraph_fromStochMatrix import getGraph
from scipy.linalg import null_space
import networkx as nx
import gc
import sys



""" allowed: 
mass-action conversion reactions between chemicals
conserved incoming and outgoing fluxes
"""
def get_reaction_derivatives(stoch_matrix):
    reaction_derivatives = stoch_matrix.copy().astype(float).T
    #reaction_derivatives[reaction_derivatives>=0] = 0
    #reaction_derivatives[reaction_derivatives<0] = 1+9*np.random.rand(
    #                        *reaction_derivatives[reaction_derivatives<0].shape)
    for i in range(stoch_matrix.shape[1]):
        for j in range(stoch_matrix.shape[0]):
            if reaction_derivatives[i, j]>=0:
                reaction_derivatives[i, j] = 0
            else:
                reaction_derivatives[i, j] = 1+9*np.random.rand()

    return reaction_derivatives

def get_vector_sMatrix(mat_pop, verbose=False):
    pop_size = mat_pop.shape[-1]
    singular = np.empty(pop_size, dtype=bool)

    #u, s, vh = np.linalg.svd(np.transpose(mat_pop, (2, 0, 1)))

    #r = [len(s[i, s[i, :]>1e-12]) for i in range(pop_size)]
    aRanks = []
    matrixrices = []

    for pop_ind in range(pop_size):
        #r = len(s[pop_ind, s[pop_ind, :]>1e-12])

        #ker = vh[pop_ind, :, r:]
        ker= null_space(mat_pop[:, :, pop_ind])
    
        nker = ker.shape[1]
        coker = null_space(mat_pop[:, :, pop_ind].T)
        #coker = u[pop_ind, :, r:]
        ncoker = coker.shape[1]
        
        eps = 1e-6

        rim = get_reaction_derivatives(mat_pop[:, :, pop_ind])
        
        amatrix = np.concatenate((rim, -1*ker),axis=1)
        
        if ncoker>0:
            Abottom = np.concatenate((-1*coker.T, np.zeros((ncoker, nker))),axis=1)
            amatrix = np.concatenate((amatrix,Abottom), axis=0)
        
        #ranks = np.linalg.matrix_rank(np.transpose(aMatrices, (2, 0, 1)))
        #amatrix[amatrix<1e-12] = 0

        if np.linalg.matrix_rank(amatrix) == amatrix.shape[0] and (amatrix.shape[0] == amatrix.shape[1]):

            
            singular[pop_ind] = False
            matrixrices.append(-np.linalg.inv(amatrix))
            matrixrices[-1][abs(matrixrices[-1])<eps] = 0
            matrixrices[-1][matrixrices[-1] < 0] = 1
            matrixrices[-1][matrixrices[-1] > 0] = 1
        
        else:
            singular[pop_ind] = True
            matrixrices.append(None)

    return singular, matrixrices, aRanks



def get_sMatrix(stoch_matrix):
    # obtain basis of kernel+cokernel from SVD of stoch. matrix
    #u, s, vh = np.linalg.svd(stoch_matrix)
    n_chem, n_react = stoch_matrix.shape
    #r = len(s[s>1e-12])

    #ker = vh[:, r:]
    ker= null_space(stoch_matrix)
   
    nker = ker.shape[1]
    coker = null_space(stoch_matrix.T)
    #coker = u[:, r:]
    ncoker = coker.shape[1]
    
    eps = 1e-6

    rim = get_reaction_derivatives(stoch_matrix)
    amatrix = np.concatenate((rim, -1*ker),axis=1)
    if ncoker>0:
        Abottom = np.concatenate((-1*coker.T, np.zeros((ncoker, nker))),axis=1)
        amatrix= np.concatenate((amatrix,Abottom), axis=0)

    singular = False

    #ranks = np.linalg.matrix_rank(np.transpose(aMatrices, (2, 0, 1)))

    #if np.linalg.cond(amatrix) > 1/sys.float_info.epsilon:
    if (np.linalg.matrix_rank(amatrix) == amatrix.shape[0]) and (amatrix.shape[0] == amatrix.shape[1]): 

     
        smatrix = -np.linalg.inv(amatrix)
        smatrix[abs(smatrix)<eps] = 0
        smatrix[smatrix < 0] = 1
        smatrix[smatrix > 0] = 1


    else:
        singular = True
        smatrix = None

    return singular, smatrix, amatrix

"""
calculate fitness as negative correlation of columns in S-matrix
"""
def get_fitness_conn(graph):
    conn_comp = sorted(nx.connected_components(
                            graph.to_undirected()),
                            key=len, reverse=True)
    av_conn_comp = np.array([len(conn_comp[i]) for i in range(len(conn_comp))]).mean()

    connectivity_fitness = av_conn_comp/len(graph)
    return connectivity_fitness

def get_fitness_cov(matrix):
    norm = matrix.shape[0]*matrix.shape[1]
    fitness = 1 - np.sum(np.cov(matrix.T))/norm
    return fitness

def get_fitness_manualCov(matrix):
    norm = matrix.shape[0]*matrix.shape[1]
    # cov = np.array([np.mean(np.multiply(np.tile(matrix[:, i], (matrix.shape[1], 1)).T, 
    #                                     matrix), axis=0) 
    #                 - matrix[:, i].mean()*matrix.mean(axis=0) 
    #        for i in range(matrix.shape[0])])
    prod_count = 0
    av_count = 0

    # contribution_list1 = []
    # contribution_list2 = []
    for col1 in range(matrix.shape[1]):
        for col2 in range(matrix.shape[1]-col1):
            
            prod_count += np.dot(matrix[:, col1], matrix[:, col2+col1])/matrix.shape[0]
            av_count += np.mean(matrix[:, col1])*np.mean(matrix[:, col2+col1])
            #print(np.mean(matrix[:, col1]), np.mean(matrix[:, col2]))
            # contribution_list1.append(np.dot(matrix[:, col1], matrix[:, col2])/matrix.shape[0])
            # contribution_list2.append(np.mean(matrix[:, col1])*np.mean(matrix[:, col2]))
    prod = prod_count/matrix.shape[1]**2
    av = av_count/matrix.shape[1]**2
    cov = prod-av

    fitness = 1 - cov
    return fitness#, contribution_list1, contribution_list2

def get_fitness_p(matrix):
    count = 0
    for row in range(matrix.shape[0]):
        for col1 in range(matrix.shape[1]):
            for col2 in range(matrix.shape[1]):
                if matrix[row, col1]==matrix[row, col2]:
                    if col1 != col2:
                        count += 1
    prob = count/(matrix.shape[0]*matrix.shape[1]**2)
    fitness = 1 - prob
    return fitness

def get_fitness_pertRespGauss(matrix, pert_loc, resp_loc, sigma):
    
    pert_scale_vec = np.full(matrix.shape[1], sigma)
    target_perturb = np.random.normal(loc=pert_loc, scale=pert_scale_vec, 
                                      size=matrix.shape[1])
    
    resp_scale_vec = np.full(matrix.shape[0], sigma)
    target_response = np.random.normal(loc=resp_loc, scale=resp_scale_vec, 
                                      size=matrix.shape[0])
    # target_perturb = np.zeros(matrix.shape[1])
    # target_perturb[0] = 1
    # target_perturb[0:3] = np.random.choice([1, 0], 3)

    # target_response = np.zeros(matrix.shape[0])
    # target_response[2] = 1
    # target_response[5:7] = np.random.choice([1, 0], 2)
    
    norm = 2*matrix.shape[0]
    
    fitness = 1-np.sqrt(np.sum((np.dot(matrix, target_perturb) - target_response)**2))/norm
    
    return fitness#*connectivity_fitness

def get_fitness_pertRespDirich(matrix, alpha, resp):
    pert = np.random.dirichlet(alpha)
    #resp = np.random.choice([0, 1], matrix.shape[0], p=[(1-resp_p), resp_p])

    norm = matrix.shape[0]
    fitness = 1 - np.sqrt(np.sum((np.dot(matrix, pert) - resp)**2))/norm
    return fitness

def get_fitness_pertResp(matrix, pert, resp):
    norm = matrix.shape[0]
    #fitness = np.sum((np.dot(matrix, pert) - resp)**2)/norm
    fitness = 1 - np.sqrt(np.sum((np.dot(matrix, pert) - resp)**2)/norm**2)
    return fitness


def get_fitness_pertRespSmat(matrix, target_mat):
    norm = target_mat.shape[0]*target_mat.shape[1]
    
    fitness = np.sqrt(np.sum((matrix - target_mat)**2)/norm**2)
   
    return 1 - fitness

def get_fitness_pertRespRandSmat(matrix, target_mat):
    norm = matrix.shape[0]*matrix.shape[1]
    target_mat = np.random.randint(0, 2, target_mat.shape)

    fitness = np.sqrt(np.sum((matrix - target_mat)**2)/norm**2)

    return 1 - fitness

def mut_targetSmat(orig_mat, curr_mat, forward_p, backward_p,
                   initialize_mc=False):
    stayf_p = 1-forward_p
    stayb_p = 1-backward_p
    n_chem = orig_mat.shape[0]
    n_react = orig_mat.shape[1]

    if initialize_mc==False:
        for i in range(n_chem):
            for j in range(n_react):
                if orig_mat[i, j]==curr_mat[i, j]:
                    curr_mat[i, j] = np.random.choice([curr_mat[i, j], 
                                                    1-curr_mat[i, j]], 
                                                    p=[stayf_p, forward_p])
                else:
                    curr_mat[i, j] = np.random.choice([curr_mat[i, j], 
                                                    1-curr_mat[i, j]], 
                                                    p=[stayb_p, backward_p])
    else:
        mc_converged = False
        count = 0
        while mc_converged==False:
            diff_entries = np.sum(abs(curr_mat-orig_mat))/(n_chem*n_react - np.sum(abs(curr_mat-orig_mat)))
            count += 1
            for i in range(n_chem):
                for j in range(n_react):
                    if orig_mat[i, j]==curr_mat[i, j]:
                        curr_mat[i, j] = np.random.choice([curr_mat[i, j], 
                                                        1-curr_mat[i, j]], 
                                                        p=[stayf_p, forward_p])
                    else:
                        curr_mat[i, j] = np.random.choice([curr_mat[i, j], 
                                                        1-curr_mat[i, j]], 
                                                        p=[stayb_p, backward_p])
            if backward_p>0:
                if abs(diff_entries - forward_p/backward_p)<1e-3:
                    mc_converged=True
                    print(f'{count} it. until mc conv.')
            else: 
                mc_converged = True

    return curr_mat



"""
select top class (asexual proliferation), middle class (stasis), low class (deletion)
"""
# def select(pop, fitness):
#     high_perc = 0.8
#     mid_perc = 0.4
#     high_ind = round(high_perc*len(fitness))
#     mid_ind = round(mid_perc*len(fitness))
   
#     sorted_fitness_inds = np.argsort(np.array(fitness))
#     sorted_pop = pop[:, :, sorted_fitness_inds]

#     prolif_pop = sorted_pop[:, :, high_ind:]
#     n_deletions = mid_ind
#     offspring_inds = np.random.choice(np.arange(prolif_pop.shape[-1]), n_deletions)
#     new_pop = sorted_pop[:]
#     new_pop[:, :, :mid_ind] = prolif_pop[:, :, offspring_inds]

#     return new_pop

"""
select stochastically based on relative fitness
"""
def select(pop, fitness, t_eff=1e-2):
    if type(fitness)==list:
        fitness = np.array(fitness)
    given_t_eff = t_eff
    # if np.mean(fitness)<0:
    #     t_eff = 10
    # else:
    #     t_eff = given_t_eff
    
    exp_fact = np.exp((fitness - np.mean(fitness))/t_eff)
    exp_fact[exp_fact>1e4] = 1e4
    # print(exp_fact)
    selection_prob = exp_fact/np.sum(exp_fact)
    # from matplotlib import pyplot as plt
    # plt.plot(selection_prob)
    # plt.show()
    new_indices = np.random.choice(pop.shape[-1], 
                                   size=pop.shape[-1], 
                                   p=selection_prob)
    
    new_pop = pop[:, :, new_indices]
    return new_pop

"""
general function mediating the mutation of
a whole population of stoch. matrices
"""
def mutate(pop, mutation_types, mutation_type_probs, mut_args):

    pop_size = pop.shape[-1]
    for pop_ind in range(pop_size):
        for mut_ind, mut in enumerate(mutation_types):
            if mutation_type_probs[mut_ind]>np.random.rand():
                pop[:, :, pop_ind] = mut(pop[:, :, pop_ind], mut_args[mut_ind])
    return pop

"""
simple mutation: for a random reaction, permute its input/output chemicals
"""
def shuffle_inputOutput(stoch_matrix, shuffle_oneColumn=False):
    if shuffle_oneColumn == True:
        rand_reaction_ind = np.random.choice(np.arange(stoch_matrix.shape[1]))

        rand_reaction_column = np.random.permutation(
                                    stoch_matrix[:, rand_reaction_ind])
        stoch_matrix[:, rand_reaction_ind] = rand_reaction_column
    else:
        rand_stoch_matrix = np.array([stoch_matrix[
                                     np.random.permutation(
                                     np.arange(stoch_matrix.shape[0])), i]
                                     for i in range(stoch_matrix.shape[1])]).T
        stoch_matrix = rand_stoch_matrix


    
    react_deriv = get_reaction_derivatives(stoch_matrix)    
    return stoch_matrix

"""
"enzyme duplication" mutation: adds random choice of {+1, -1} and {+1 -1 0} to reaction
"""

def enzyme_dup(stoch_matrix, allow_new_conversion=True):
    rand_react_ind = np.random.choice(stoch_matrix.shape[1])

    mut = False

    while mut==False:
        
        loss_or_gain = np.random.rand()
        if loss_or_gain>=0.5 and (stoch_matrix[:, rand_react_ind]==0).any():
            # function gain
            zero_chems = np.concatenate(np.argwhere(
                                stoch_matrix[:, rand_react_ind]==0))
            rand_new_int = np.random.choice(zero_chems)           
            stoch_matrix[rand_new_int, rand_react_ind] = np.random.choice([1, -1])
            mut = True
            # if allow_new_conversion==True and len(non_zero_chems)>1:
            #     rand_new_int = np.random.choice(non_zero_chems)
            #     stoch_matrix[rand_new_int, rand_react_ind] = np.random.choice([1, -1, 0])
        elif (stoch_matrix[:, rand_react_ind]!=0).any():
            if (stoch_matrix[:, rand_react_ind]!=0).size>1:
                # function loss
                non_zero_chems = np.concatenate(np.argwhere(
                                    stoch_matrix[:, rand_react_ind]!=0))
                rand_lost_int = np.random.choice(non_zero_chems) 
                stoch_matrix[rand_lost_int, rand_react_ind] = 0
                mut = True
            else:
                rand_react_ind = np.random.choice(stoch_matrix.shape[1])
    return stoch_matrix

def get_init_pop(n_pop, n_chem, n_react, maxIt=100):
    init_pop = np.zeros(shape=(n_chem, n_react, n_pop))
    #init_pop[0, :] = 1
    #init_pop[1, :] = -1
    #init_pop[:, :, :] = np.random.choice([-1, 1, 0], size=init_pop[:, :, :].shape)
    # init_pop[0, :, :] = np.random.choice([1, 0],
    #                                      size=init_pop[0, :, :].shape,
    #                                      p=[0.72/(0.72+0.12), 0.12/(0.72+0.12)])
    # init_pop[1, :, :] = np.random.choice([1, 0], 
    #                                      size=init_pop[1, :, :].shape,
    #                                      p=[0.16/(0.16+0.12), 0.12/(0.16+0.12)])
    # init_pop[2, :, :] = np.random.choice([-1, 0], 
    #                                      size=init_pop[2, :, :].shape,
    #                                      p=[0.58/(0.58+0.12), 0.12/(0.58+0.12)])
    # init_pop[3, :, :] = np.random.choice([-1, 0], 
    #                                      size=init_pop[1, :, :].shape,
    #                                      p=[0.3/(0.12+0.3), 0.12/(0.3+0.12)])

    init_pop[0, :, :] = np.random.choice([1, 0, -1],
                                         size=init_pop[0, :, :].shape
                                         )
    init_pop[1, :, :] = np.random.choice([1, 0, -1], 
                                         size=init_pop[1, :, :].shape
                                         )
    init_pop[2, :, :] = np.random.choice([1, 0, -1],
                                         size=init_pop[2, :, :].shape
                                         )

    # inp = [np.array([0, 0]),
    #        np.array([-1, 0]),
    #        np.array([-1, -1]),]
    # inp_indices = np.random.choice([0, 1, 2],
    #                                size=init_pop[0, :, :].shape,
    #                                p=[0.3, 0.58, 0.12])
    # print(inp_indices)
    # print(np.array([inp[ind] for ind in inp_indices]).T.shape)
    # init_pop[0:2, :, :] = np.array([inp[ind] for ind in inp_indices]).T 
    # print(init_pop[0:2, :, :])

    # outp = [np.array([0, 0]),
    #        np.array([1, 0]),
    #        np.array([1, 1]),]
    # outp_indices = np.random.choice([0, 1, 2], 
    #                                 size=init_pop[0, :, :].shape,
    #                                 p=[0.16, 0.72, 0.12])
    # print(outp_indices)
    # init_pop[2:4, :, :] = np.array([outp[ind] for ind in outp_indices]).T
    # print(init_pop[2:4, :, :])

   
    n_sing = 0
    for j in tqdm(range(n_pop)):
        init_pop[:, :, j] = shuffle_inputOutput(init_pop[:, :, j],
                                                shuffle_oneColumn=False)
    singular, smatrix, aRank = get_vector_sMatrix(init_pop, verbose=True)
    from matplotlib import pyplot as plt

    for j in tqdm(range(n_pop)):
        if singular[j]:
            it = 0
            while it < maxIt:
                it += 1
                init_pop[:, :, j] = shuffle_inputOutput(init_pop[:, :, j],
                                                        shuffle_oneColumn=False)
                sing, smatrix, aRank = get_sMatrix(init_pop[:, :, j])
                if not sing:
                    it = maxIt
            if sing:
                n_sing += 1
    print(f'{100*n_sing/n_pop} % of singular\
          networks in the starting generation') 
    return init_pop

def evolve_networks(mutation_types, mutation_type_probs, mut_args, init_pop, 
                    epochs=100, save_ep=10, n_chem=2, n_react=2, n_pop=100,
                    select_t=1e-2, alpha=None, resp_p=None, pert=None, resp=None,
                    target_smat=None, target_smat_fp=None, target_smat_bp=None,
                    verbose=0):
    population_hist = np.zeros((n_chem, n_react, n_pop, epochs//save_ep+1))
    #population_matrix_hist = []
    fitness_hist = np.zeros((n_pop, epochs+1))

    population_hist[:, :, :, 0] =  init_pop

    prev_pop = init_pop
    working_target_smat = target_smat.copy()

    for epoch in tqdm(range(epochs+1)):    
        gc.collect()

        pop = prev_pop

        if epoch>0:
            mut_pop = mutate(pop, mutation_types, mutation_type_probs, mut_args)
        else:
            mut_pop = pop

        n_sing = 0

        t_tot = time()

        ##### VECTORIZE

        singular, smatrix, aRank = get_vector_sMatrix(mut_pop)
        
        for pop_ind in range(n_pop):
            
            if singular[pop_ind]:
          
                fitness_hist[pop_ind, epoch] = -1
                n_sing += 1
            else:
                #fitness = (get_fitness_manualCov(smatrix[pop_ind][:n_chem, :n_react]))#*
                           #get_fitness_conn(getGraph(mut_pop[:, :, pop_ind])))
                
                # fitness = get_fitness_pertRespGauss(smatrix[pop_ind][:n_chem, :n_react],
                #                                    pert_loc=pert_loc, 
                #                                    resp_loc=resp_loc,
                #                                    sigma=sigma)*
                #            get_fitness_conn(getGraph(mut_pop[:, :, pop_ind])))

                # fitness = get_fitness_pertRespDirich(smatrix[pop_ind][:n_chem, :n_react],
                #                                      alpha, resp)
                # fitness = get_fitness_pertResp(smatrix[pop_ind][:n_chem, :n_react],
                #                                pert=pert, resp=resp)
                # fitness = get_fitness_pertRespSmat(smatrix[pop_ind][:n_chem, :n_react],
                #                                    target_mat=target_smat)
                
                fitness = get_fitness_pertRespSmat(smatrix[pop_ind][:n_chem, :n_react],
                                                   target_mat=working_target_smat)
                # fitness = get_fitness_pertRespRandSmat(smatrix[pop_ind][:n_chem, :n_react],
                #                                    target_mat=working_target_smat)
           
                fitness_hist[pop_ind, epoch] = fitness

        if epoch==0:
            init_mc = True
        else:
            init_mc = False

        working_target_smat = mut_targetSmat(orig_mat=target_smat,
                                             curr_mat=working_target_smat,
                                             forward_p=target_smat_fp,
                                             backward_p=target_smat_bp,
                                             initialize_mc=init_mc)
        

       
        #####

        t_tot = time() - t_tot
        if verbose==1 and epoch%10==0:
            print(f'{100*n_sing/n_pop} % singular networks in gen.{epoch}')
            print(max(fitness_hist[:, epoch]))
            print(np.mean(fitness_hist[:, epoch]))
            print(n_pop*np.std(fitness_hist[:, epoch])/select_t)
            #print(f'divergence of current target S-matrix={np.sum(abs(working_target_smat-target_smat))}')
            #print(working_target_smat)
        elif verbose==2:
            print(f'{n_sing} singular networks in gen.{epoch}')
            print(f'spread of fitness = {np.std(np.array(fitness_hist[:, epoch]))}')
            print(max(fitness_hist[:, epoch]))
            #print(f'divergence of current target S-matrix={np.mean(abs(working_target_smat-target_smat))}')

        new_pop = select(mut_pop, fitness_hist[:, epoch], select_t)
        
        prev_pop = new_pop

        if epoch%save_ep==0:
            
            population_hist[:, :, :, epoch//save_ep] = mut_pop

    return population_hist, fitness_hist