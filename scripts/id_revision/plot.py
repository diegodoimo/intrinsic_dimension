import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import scipy.stats
import pandas as pd
from matplotlib.ticker import NullFormatter

sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})




id_gride_ft = []
id_2nn_ft = []
id_geomle_ft = []
id_danco_ft = []
id_ess_ft = []


N = 1000
for p in [20, 50, 100, 200]:
    arr = np.genfromtxt(f'./results/mnist_feat{p}_n{N}_k10_40_bm_sample.txt')[:, 2]
    id_gride_ft.append(arr[0])
    id_2nn_ft.append(arr[1])
    id_geomle_ft.append(arr[2])
    id_danco_ft.append(arr[3])
    id_ess_ft.append(arr[4])

id_gride_n = []
id_2nn_n = []
id_geomle_n = []
id_danco_n = []
id_ess_n = []

p = 100
for N in [200, 500, 1000, 2000]:
    arr = np.genfromtxt(f'./results/mnist_feat{p}_n{N}_k10_40_bm_sample.txt')[:, 2]
    id_gride_n.append(arr[0])
    id_2nn_n.append(arr[1])
    id_geomle_n.append(arr[2])
    id_danco_n.append(arr[3])
    id_ess_n.append(arr[4])

id_ess_n


gride_mnist = np.load('./results/mnist_feat200_n1000_k10_40_bm_sample_gride.npy')
two_nn_mnist = np.load('./results/mnist_feat200_n1000_k10_40_bm_sample_2NN.npy')



fig = plt.figure(figsize = (7, 3.5))
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
p = [1000/2**i for i in range(len(gride_mnist[0][:-1]))]
sns.lineplot(x = p, y = gride_mnist[0][:-1], ax = ax, label = 'gride', marker = 'o')
sns.lineplot(x = p, y = two_nn_mnist[0], ax = ax, label = '2NN', marker = 'o')
ax.set_ylabel('ID', fontsize = 13)
ax.set_xlabel('# data', fontsize = 13)

ax = fig.add_subplot(gs[1])
sns.lineplot(x = gride_mnist[2], y = gride_mnist[0], ax = ax, label = 'gride_rs', marker = 'o')
ax.set_ylabel('ID', fontsize = 13)
ax.set_xlabel('distance range', fontsize = 13)

plt.savefig('girde_vs_2NN.png', dpi = 200)

fig = plt.figure(figsize = (7, 3.5))
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
p = [20, 50, 100, 200]
sns.lineplot(x = p, y = id_gride_ft, ax = ax, label = 'gride', marker = 'o')
sns.lineplot(x = p, y = id_2nn_ft, ax = ax, label = '2NN', marker = 'o')
sns.lineplot(x = p, y = id_geomle_ft, ax = ax, label = 'geomle', marker = 'o')
sns.lineplot(x = p, y = id_danco_ft, ax = ax, label = 'danco', marker = 'o')
sns.lineplot(x = p, y = id_ess_ft, ax = ax, label = 'ess', marker = 'o')
ax.set_xlabel('# features', fontsize = 13)
ax.set_ylabel('time (sec)', fontsize = 13)
ax.set_yscale('log')


ax = fig.add_subplot(gs[1])
p = [200, 500, 1000, 2000]
sns.lineplot(x = p, y = id_gride_n, ax = ax, label = 'gride', marker = 'o')
sns.lineplot(x = p, y = id_2nn_n, ax = ax, label = '2NN', marker = 'o')
sns.lineplot(x = p, y = id_geomle_n, ax = ax, label = 'geomle', marker = 'o')
sns.lineplot(x = p, y = id_danco_n, ax = ax, label = 'danco', marker = 'o')
sns.lineplot(x = p, y = id_ess_n, ax = ax, label = 'ess', marker = 'o')
ax.set_xlabel('# data', fontsize = 13)
ax.set_ylabel('time (sec)', fontsize = 13)
ax.set_yscale('log')

gs.tight_layout(fig)

plt.savefig('benchmarks.png', dpi = 200)
