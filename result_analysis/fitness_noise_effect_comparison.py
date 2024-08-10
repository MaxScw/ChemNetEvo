import numpy as np
from matplotlib import pyplot as plt

fpath = ''#'SmatDist_noiseComp_fb=fp/'

maxModes = np.loadtxt(fpath+'maxMode.txt')
symmDevs = np.loadtxt(fpath+'symmDev.txt')
tick_labels = ['1/2', '2/3', '1', '17/13', '2']
#tick_labels = ['$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
ticks = np.linspace(0, len(tick_labels)-1, len(tick_labels))
symmDev_fig, symmDev_ax = plt.subplots(1)

symmDev_ax.plot(ticks, symmDevs)
symmDev_ax.set_xticks(ticks)
symmDev_ax.set_xticklabels(tick_labels)
symmDev_ax.set_xlabel('$c := p_{forward}/p_{backward}$')
symmDev_ax.set_ylabel('$\langle |M - M^T| \\rangle$')
symmDev_ax.set_title('deviation from symmetry in 2d-fitness histogram')

symmDev_fig.savefig('symmDev.pdf', format='pdf')

mode_fig, mode_ax = plt.subplots(1)

mode_ax.plot(ticks, maxModes[:, 0], color='green', label='original target')
mode_ax.plot(ticks, maxModes[:, 1], color='blue', label='final target')
mode_ax.legend()
mode_ax.set_xticks(ticks)
mode_ax.set_xticklabels(tick_labels)
mode_ax.set_xlabel('$c := p_{forward}/p_{backward}$')
mode_ax.set_ylabel('fitness (a.u.)')
mode_ax.set_title('highest mode in population fitness')

mode_fig.savefig('maxMode.pdf', format='pdf')

plt.show()