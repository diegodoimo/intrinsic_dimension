import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns

#sns.set_theme()
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 0.01})


#*******************************************************************************
id = 2
nrep = 1000
scirep= False
filename = 'no_center'

plots_folder = './plots'
results_folder = f'./results/ids_poisson_dataset'
#ndata = [128, 512, 2048, 8192]
ndata = [128, 512, 2048]
ndata = [11, 33, 101, 303]
fig = plt.figure(figsize = (9, 2.7))
gs = GridSpec(1, 4)
ks = [2**i for i in range(9)]
kticks = [4**i for i in range(5)]
for i in range(len(ndata)):
    if scirep:
        ids = np.load(f'{results_folder}/scirep/id2_nrep{nrep}_ndata{ndata[i]}_ntot{max(5*ndata[i], 2048)}.npy')
    else:
        ids = np.load(f'{results_folder}/id{id}_nrep{nrep}_ndata{ndata[i]}{filename}.npy')
    ax = fig.add_subplot(gs[i])

    mean = np.mean(ids, axis = 0)
    stderr = np.std(ids, axis = 0)/ids.shape[0]**0.5

    sns.lineplot(x = ks, y =  mean, marker = 'o', linewidth = 0.5, label = f'N = {ndata[i]}')
    ax.fill_between(ks, mean - 1.96*stderr, mean + 1.96*stderr, alpha = 0.2)
    ax.axhline(id, color = 'gray', linestyle = '--')

    ax.set_xscale('log')
    ax.set_xticks(kticks)
    ax.set_xticklabels(kticks, rotation = 45)
    ax.set_xlabel('$k_1$')

    if i ==0:
        ax.set_ylabel('ID', fontsize = 13)

gs.tight_layout(fig, rect = [0, 0.02, 1, 1.])

101/3


128*4*4*4

101*3*3*3


plt.plot(ids[:, 5], linewidth = 0, marker = '.')


#fig.text(0.48, 0.02, 'n (as in $r_{2n}/r_n$)', fontsize = 14)
#fig.suptitle(f'Average maximum likelihood over 1000 data samples')


plt.savefig('{plots_folder}/poisson_dataset.pdf')
