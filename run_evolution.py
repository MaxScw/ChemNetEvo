import model as evo
from matplotlib import pyplot as plt
import matplotlib as mpl
import gc

mutation_types = [evo.enzyme_dup]
mut_args = [True]
epochs = 1000
ep_frac = 10
n_chem = 3
n_react = 5
n_pop = 1000
t = 1e-1
nruns = 1

start_run_string = f'_N{n_pop}_ep{epochs}_c{n_chem}r{n_react}'

dirichAlpha = evo.np.ones(n_react)
dirichAlpha[0] = 1e6
dirichAlpha[4] = 1e6
resp = evo.np.zeros(n_chem)

#pert = evo.np.full(n_react, 1/n_react)
# pert = evo.np.zeros(n_react)
# pert[4] = 1/2
# pert[0] = 1/2
# #resp = evo.np.full(n_chem, 1)
# resp = evo.np.zeros(n_chem)
# resp[0] = 1

#target_smat = evo.np.load('ecoli_sMat.npy')[:n_chem, :n_react]
target_smat = evo.np.zeros((n_chem, n_react))
target_smat[:, 0] = evo.np.array([1, 1, 1])
target_smat[0, 1] = 1
target_smat[1, 2] = 1
target_smat[:, 3] = evo.np.array([1, 1, 1])
target_smat[0, 4] = 1
target_smat[1, 4] = 1
# target_smat = evo.np.random.randint(0 , 2, size=(n_chem, n_react))
# print(target_smat)
# evo.np.save(f'target_sMat'+start_run_string, target_smat)
#target_smat = evo.np.load(f'target_sMat'+start_run_string+'.npy')

target_smat_fp = 0
target_smat_bp = 0

for run in range(nruns):
    # define appendix string for plot names
    appendix = f'_fp0_bp0_dyn_run{run+1}'#f'enzymeDup_starter'
    run_string = start_run_string + appendix

    #evo.np.save('init_locVec'+run_string, pert_loc, resp_loc)

    init_pop = evo.get_init_pop(n_pop=n_pop, n_chem=n_chem,
                              n_react=n_react, maxIt=1000)

    #init_pop = evo.np.load('init_pop_N100_ep4000_c3r5_starter.npy')

    evo.np.save(f'init_pop'+run_string, init_pop)

    """ noMut_results = evo.evolve_networks(mutation_types, mutation_type_probs=[0],
                                    mut_args=mut_args,
                                    epochs=epochs, save_ep=epochs//ep_frac,
                                    n_chem=n_chem,
                                    n_react=n_react, n_pop=n_pop,
                                    init_pop=init_pop, select_t=t, 
                                    alpha=dirichAlpha,
                                    
                                    resp=resp, target_smat=target_smat,
                                    target_smat_fp=target_smat_fp,
                                    target_smat_bp=target_smat_bp, 
                                    verbose=1)
    noMut_pop = noMut_res
    noMut_fit = noMut_results[1]

    # save results

    evo.np.save(f'noMut_pop'+run_string, noMut_pop)
    evo.np.save(f'noMut_fitness'+run_string, noMut_fit)
    del noMut_results
    del noMut_pop
    del noMut_fit
    gc.collect() """

    mut_results = evo.evolve_networks(mutation_types,
                                    mutation_type_probs=[10/n_pop],                                                                           
                                    mut_args=mut_args,
                                    epochs=epochs, save_ep=epochs//ep_frac,
                                    n_chem=n_chem,
                                    n_react=n_react, n_pop=n_pop,
                                    init_pop=init_pop, select_t=t, 
                                    alpha=dirichAlpha,
                                    
                                    resp=resp, target_smat=target_smat,
                                    target_smat_fp=target_smat_fp,
                                    target_smat_bp=target_smat_bp, 
                                    verbose=1)
    mut_pop = mut_results[0]
    mut_fit = mut_results[1]

    # save results

    evo.np.save(f'mut_pop'+run_string, mut_pop)
    evo.np.save(f'mut_fitness'+run_string, mut_fit)

    del mut_results
    del mut_pop
    del mut_fit
    gc.collect()



