import math
import numpy as np
from matplotlib import pyplot as plt

n_chem = 10
p_non = (1+2/3+2/3)/3
n_non = np.arange(1, 10)

def P_cycle(n_chem, n_non, p_conn):
    prefac = (math.factorial(n_chem)*p_conn**n_chem)/(n_chem**n_chem)

    a = np.arange(0, n_chem)
    a_fact = np.array([math.factorial(num) for num in a])
    summation = sum((n_chem**a)/(a_fact*p_conn**a))

    return prefac*summation

p_cycle_list = []
for non in n_non:
    p_conn = 0.5*non*(non-1)*(p_non/n_chem)**2
    p_cycle_list.append(P_cycle(n_chem=n_chem, n_non=non, p_conn=p_conn))

p_conn = p_conn = 0.5*n_non*(n_non-1)*(p_non/n_chem)**2

plt.scatter(n_non, p_cycle_list, label='$P_{cycle}$')
plt.scatter(n_non, p_conn, label='$p_{conn}$')

plt.ylabel('$probability$')
plt.xlabel('$n_{non}$')
plt.title('$n_{chem}$='+f'{n_chem}, '+'$p_{non}$='+f'{round(p_non,3)}')
plt.legend()
plt.show()