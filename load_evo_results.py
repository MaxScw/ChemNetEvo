import numpy as np

def load_evo_results(n_pop, n_chem, n_react, n_epochs, appendix, path='',
                     noMut=True):
    run_string = f'_N{n_pop}_ep{n_epochs}_c{n_chem}r{n_react}{appendix}.npy'

    mut_pop = np.load(path+'mut_pop'+run_string)
    mut_fitness = np.load(path+'mut_fitness'+run_string)
    
    if noMut==True:
        noMut_pop = np.load(path+'noMut_pop'+run_string)
        noMut_fitness = np.load(path+'noMut_fitness'+run_string)
        return mut_pop, mut_fitness, noMut_pop, noMut_fitness
    else:
        return mut_pop, mut_fitness