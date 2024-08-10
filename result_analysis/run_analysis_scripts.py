import model as evo
import os

n_pop = 1000
n_chem = 3
n_react = 5
n_epochs = 1000
save_ep = 100
appendix = '_fp0_bp0_run1'#'enzymeDup_starter'#enzymeDup_run1'
save_appendix = ''
fpath = ''

""" runstr_struct = f'python structural_analysis.py -n_pop {n_pop} -n_chem {n_chem}\
                  -n_react {n_react} -n_epochs {n_epochs} -save_ep {save_ep}\
                  --appendix {appendix} --save_appendix {save_appendix}\
                  --fpath {fpath}'

os.system(runstr_struct) """

runstr_an = f'python3 analysis.py -n_pop {n_pop} -n_chem {n_chem}\
                  -n_react {n_react} -n_epochs {n_epochs} -save_ep {save_ep}\
                  --appendix {appendix} --save_appendix {save_appendix}\
                  --fpath {fpath}'

os.system(runstr_an)
