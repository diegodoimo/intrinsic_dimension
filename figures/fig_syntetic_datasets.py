import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import scipy.stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 0.01})

#*******************************************************************************
"fig syntetic datastes"

results = './results/datasets/syntetic'
N = 32000
eps=0
names = {
    'normal':         {'N':16000,'D': 3,'d': 2,'eps': 0.01},
    'spiral1d':       {'N':16000,'D': 3,'d': 1,'eps': 0.01},
    'moebius':        {'N':16000,'D': 6,'d': 2,'eps': 0.01},
    'uniform2_0':     {'N':N,'D': 2,'d': 2,'eps': eps},
    'uniform5_0':     {'N':N,'D': 5,'d': 5,'eps': eps},
    'uniform10_0':    {'N':N,'D': 10,'d': 10,'eps': eps},
    'uniform20_0':    {'N':N,'D': 20,'d': 20,'eps': eps},
    'uniform50_0':    {'N':N,'D': 50,'d': 50,'eps': eps}
}


fig = plt.figure(figsize = (9, 2.7))
gs0 = GridSpec(1, 3)
gs1 = GridSpec(1, 1)
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
titles = ['Gaussian', 'Spiral', 'Moebius', 'Uniform']
algos = ['Gride', 'TwoNN', 'MLE']
for i, (key, kwargs) in enumerate(names.items()):
    #print(key)
    if i <3:
        N = 16000
        ax = fig.add_subplot(gs0[i])
    else:
        N = 32000
        if i==3:
            ax = fig.add_subplot(gs1[i-3])
        j=i-3


    X = np.load(f'{results}/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0]
    xticks = [N/2**i for i in range(len(X))]
    if i <3:
        for l, name in enumerate(['gride', 'twonn', 'mle']):
            X = np.load(f'{results}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0]
            if name == 'twonn':
                X = X[:-1]
            xticks = [N/2**k for k in range(len(X))]
            sns.lineplot(x=xticks, y=X, ax = ax, marker = 'o', label = f'{algos[l]}')


    else:
        #print(name, N, key, kwargs['D'], kwargs['d'], kwargs['eps'])
        X = np.load(f'{results}/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0]
        xticks = [N/2**i for i in range(len(X))]
        sns.lineplot(x=xticks, y=X, ax = ax, marker = 'o', color = colors[j])

    ax.set_xlabel('$N/k_2$')

    if i< 3:
        ax.set_xscale('log')
        ax.axhline(kwargs["d"], color = 'gray', linestyle = '--')
        ax.set_title(f'{titles[i]}', fontsize = 13)
    elif key =='uniform50_0':
        ax.set_title(f'{titles[-1]}', fontsize = 13)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.set_yticklabels([2, 5, 10, 20, 50])
        ax.set_ylim(1.6, 60)
        ax.set_ylabel('ID', fontsize = 13)
        ax.axhline(kwargs["d"], color = colors[j], linestyle = '--')
    else:
        ax.axhline(kwargs["d"], color = colors[j], linestyle = '--')
    if i in [0]:
        ax.set_ylabel('ID', fontsize = 13)



gs1.tight_layout(fig, rect = [0., 0, 0.27, 1])
gs0.tight_layout(fig, rect = [0.3, 0, 1, 1])
fig.text(0.01, 0.9, 'a', fontsize = 15, fontweight = 'bold')
fig.text(0.33, 0.9, 'b', fontsize = 15, fontweight = 'bold')



plt.savefig('./plots/syntetic_datasets.pdf')









#*******************************************************************************
"appendix syntetic datasets"


N = 16000
eps = 0.01

names = {
    'uniform20':        {'N':N,'D': 20,'d': 5,'eps': eps},
    'normal':           {'N':N,'D': 3,'d': 2,'eps': eps},
    'normal20':         {'N':N,'D': 20,'d': 2,'eps': eps},
    'sphere':           {'N':N,'D': 15,'d': 10,'eps': eps},
    'spiral1d':         {'N':N,'D': 3,'d': 1,'eps': eps},
    'swissroll':        {'N':N,'D': 3,'d': 2,'eps': eps},
    'moebius':          {'N':N,'D': 6,'d': 2,'eps': eps},
    'paraboloid':       {'N':N,'D': 30,'d': 9,'eps': eps},
    'nonlinear':        {'N':N,'D': 36,'d': 6,'eps': eps}
}

fig = plt.figure(figsize = (6, 6))
gs = GridSpec(3, 3)

for i, (key, kwargs) in enumerate(names.items()):
    #print(key)
    ax = fig.add_subplot(gs[i])

    for name in ['mle', 'twonn', 'gride']:
        X = np.load(f'{results}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0]
        if name == 'twonn':
            #print(X.shape)
            X = X[:-1]
            #print(X.shape)

        xticks = [N/2**i for i in range(len(X))]
        if i ==0:
            sns.lineplot(x=xticks, y=X, ax = ax, marker = 'o', label = f'{name}')
        else:
            sns.lineplot(x=xticks, y=X, ax = ax, marker = 'o')
    ax.axhline(kwargs["d"])
    ax.set_title(f'{key} ({kwargs["D"]}, {kwargs["d"]})')
    ax.set_xscale('log')
    if i in [0, 1, 2, 3, 4, 5]:
        ax.set_xticklabels([])
    if i in [0, 3, 6]:
        ax.set_ylabel('ID')
    if i in [6, 7, 8]:
        ax.set_xlabel('$N/k_2$')

gs.tight_layout(fig)



plt.savefig('./plots/ID_app1.pdf')
