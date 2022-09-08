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

#results = './results/datasets/syntetic/old'

results = '../scripts/results/syntetic_datasets'
N = 32000
eps=0
names = {
    'spiral1d':       {'N':16000,'D': 3,'d': 1,'eps': 0.01},
    'normal':         {'N':16000,'D': 3,'d': 2,'eps': 0.01},
    'moebius':        {'N':16000,'D': 6,'d': 2,'eps': 0.01},
    'uniform2_0':     {'N':N,'D': 2,'d': 2,'eps': eps},
    'uniform5_0':     {'N':N,'D': 5,'d': 5,'eps': eps},
    'uniform10_0':    {'N':N,'D': 10,'d': 10,'eps': eps},
    'uniform20_0':    {'N':N,'D': 20,'d': 20,'eps': eps},
    'uniform50_0':    {'N':N,'D': 50,'d': 50,'eps': eps}
}


#geenerate X data from danco and ess txt files
def gen_data(filename, key, ess_local = False):
    X = np.genfromtxt(filename)
    ids = []

    nrep_check = [2**i for i in range(10)]
    ndata_tot = [int(16000/2**i) for i in range(10)]

    for j, ndata in enumerate(ndata_tot):
        if ess_local:
            assert np.sum(X[:, 0]==ndata) == nrep_check[j] or np.sum(X[:, 0]==ndata) == 4
        else:
            assert np.sum(X[:, 0]==ndata) == nrep_check[j]  #check we do not lose data

        ids.append(  np.mean(X[:, 1][ X[:, 0]==ndata]) )

    return np.array(ids)




colors = ['C0', 'C1', 'C2', 'C3', 'C4']
titles = ['spiral', 'normal', 'moebius', 'uniform']
algos = ['Gride', 'TwoNN', 'DANCo', 'ESS', 'MLE', 'GeoMLE']


filename = '_k20_55'

fig = plt.figure(figsize = (10, 3))
gs0 = GridSpec(1, 3)
gs1 = GridSpec(1, 1)
for i, (key, kwargs) in enumerate(names.items()):
    if i <3:
        N = 16000
        ax = fig.add_subplot(gs0[i])
        for l, name in enumerate(['gride', 'twonn', 'danco', 'ess']):#, 'mle', 'geomle']):
            if name == 'ess':
                #X = gen_data(f'{results}/{name}/{key}_16k_eps0.01.txt', name)
                X = gen_data(f'{results}/{name}/ess_local/{key}_16k_eps0.01_k10local.txt', name, ess_local = True)[:-1]
            elif name == 'danco':
                X = gen_data(f'{results}/{name}/DANCo_16k_eps0.01_{key}.txt', name)[:-1]
            elif name == 'geomle':
                #print(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}{filename}.npy')
                X = np.load(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}{filename}.npy')[0][:-3]
            else:
                X = np.load(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0][:]
            if name == 'gride':
                pass
                #print(np.load(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[2])

            xticks = np.array([N/2**k for k in range(len(X))])
            if name in ['gride', 'twonn']: xticks/=1.5
            elif name in ['danco', 'mle', 'ess']: xticks/=5
            elif name in ['geomle']: xticks/=38
            marker = 'o'
            markersize = 5
            if name == 'gride':
                marker = 'X'
                markersize = 9


            if i==0:

                sns.lineplot(x=xticks, y=X, ax = ax, marker = marker, label = f'{algos[l]}', zorder = 20-5*i, markersize = markersize)
            else:
                sns.lineplot(x=xticks, y=X, ax = ax, marker = marker, zorder = 20-5*i, markersize = markersize) #do not plot legend

        ax.set_xscale('log')
        ax.set_xticks([10, 100, 1000, 10000])
        ax.set_xticklabels(['$10^1$','$10^2$','$10^3$', '$10^4$'])
        ax.axhline(kwargs["d"], color = 'gray', linestyle = '--')
        ax.set_title(f'{titles[i]} ({kwargs["d"]}, {kwargs["D"]})', fontsize =12)
        ax.set_xlabel('$N/\overline{k}$')

        if i in [0]:
            #ax.legend(loc='lower right', bbox_to_anchor=(0.5, 0.4), fontsize = 10)
            ax.legend(fontsize = 10)
            ax.set_ylabel('ID', fontsize = 14)
            #ax.set_xlim(3, 15000)


    else:

        N = 32000
        if i==3:
            ax = fig.add_subplot(gs1[i-3])

        X = np.load(f'{results}/gride/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0]
        xticks = np.array([N/2**k for k in range(len(X))])
        if name in ['gride', 'twonn']: xticks/=1.5
        elif name in ['danco', 'mle']: xticks/=5
        elif name in ['geomle']: xticks/=38


        sns.lineplot(x=xticks, y=X, ax = ax, marker = 'X', color = colors[i-3], markersize = 6)

        ax.set_xlabel('$N/\overline{k}$')
        if key =='uniform50_0':
            ax.set_title(f'{titles[-1]}', fontsize = 13)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_yticks([2, 5, 10, 20, 50])
            ax.set_yticklabels([2, 5, 10, 20, 50])
            ax.set_ylim(1.6, 60)
            ax.set_ylabel('ID', fontsize = 14)
            ax.axhline(kwargs["d"], color = colors[i-3], linestyle = '--')
        else:
            ax.axhline(kwargs["d"], color = colors[i-3], linestyle = '--')


gs0.tight_layout(fig, rect = [0.25, 0, 1, 1])
gs1.tight_layout(fig, rect = [0., 0, 0.23, 1])

fig.text(0.01, 0.9, 'a', fontsize = 14, fontweight = 'bold')
fig.text(0.3, 0.9, 'b', fontsize = 14, fontweight = 'bold')
fig.text(0.54, 0.9, 'c', fontsize = 14, fontweight = 'bold')
fig.text(0.77, 0.9, 'd', fontsize = 14, fontweight = 'bold')


plt.savefig('./plots/syntetic_datasets.pdf')


#*******************************************************************************
"appendix syntetic datasets"

N = 16000
eps = 0.001

names = {
    'spiral1d':         {'N':N,'D': 3,'d': 1,'eps': eps},
    'normal':           {'N':N,'D': 3,'d': 2,'eps': eps},
    'moebius':          {'N':N,'D': 6,'d': 2,'eps': eps},
    'normal20':         {'N':N,'D': 20,'d': 2,'eps': eps},
    'swissroll':        {'N':N,'D': 3,'d': 2,'eps': eps},

    'nonlinear':        {'N':N,'D': 36,'d': 6,'eps': eps},
    'uniform20':        {'N':N,'D': 20,'d': 5,'eps': eps},
    'sphere':           {'N':N,'D': 15,'d': 10,'eps': eps},
    'paraboloid':       {'N':N,'D': 30,'d': 9,'eps': eps}
}


filename = '_k20_55'

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
algos = ['Gride', 'TwoNN', 'DANCo', 'ESS', 'MLE', 'GeoMLE']


fig = plt.figure(figsize = (11, 5.5))


lim = [(0.8, 3.2), (1.7, 2.6), (1.1, 3.4), (1.8, 3.9), (1.4, 2.9), (4.8, 9.8), (3.6, 5.1), (6.7, 11.3), (0.8, 9.4)]


for i, (key, kwargs) in enumerate(names.items()):
    gs = GridSpec(1, 1)
    N = 16000
    ax = fig.add_subplot(gs[0])
    for l, name in enumerate(['gride', 'twonn', 'danco', 'ess', 'mle', 'geomle']):
        if name == 'ess':
                #X = gen_data(f'{results}/{name}/{key}_16k_eps{eps}.txt', name)
                X = gen_data(f'{results}/{name}/ess_local/{key}_16k_eps{eps}_k10local.txt', name, ess_local = True)[:-1]
        elif name == 'danco':
            X = gen_data(f'{results}/{name}/DANCo_16k_eps{eps}_{key}.txt', name)[:-1]
        elif name == 'geomle':
            X = np.load(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}{filename}.npy')[0][:-3]
        else:
            X = np.load(f'{results}/{name}/{name}_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy')[0][:]


        xticks = np.array([N/2**k for k in range(len(X))])
        if name in ['gride', 'twonn']: xticks/=1.5
        elif name in ['danco', 'mle', 'ess']: xticks/=5.5
        elif name in ['geomle']: xticks/=38
        marker = 'o'
        markersize = 5
        if name == 'gride':
            marker = 'X'
            markersize = 8
        if i==0:
            sns.lineplot(x=xticks, y=X, ax = ax, marker = marker, label = f'{algos[l]}', markersize = markersize)
        else:
            sns.lineplot(x=xticks, y=X, ax = ax, marker = marker, markersize = markersize) #do not plot legend

    ax.set_title(f'{key} ({kwargs["D"]}, {kwargs["d"]})', fontsize =12)
    ax.set_xscale('log')
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(['$10^1$','$10^2$','$10^3$', '$10^4$'])
    ax.set_ylim(lim[i][0], lim[i][1])
    if i+1 in [0, 1, 2, 3, 4]:
        ax.set_xticklabels([])
    if i+1 in [1, 5]:
        ax.set_ylabel('ID', fontsize = 15)
    if i+1 in [5, 6, 7, 8, 9]:
        ax.set_xlabel('$N/\overline{k}$')
    ax.axhline(kwargs["d"], linestyle = '--', color = 'black', label = 'True ID')

    if i > 3:
        gs.tight_layout(fig, rect = [(i-4)/5, 0.0, (i-3)/5, 0.51])

    else:
        if i == 0:
            gs.tight_layout(fig, rect = [0.2, 0.55, 0.4, 0.95])
            plt.legend(bbox_to_anchor =(-0.55, 1.), fontsize = 11)
        else:
            gs.tight_layout(fig, rect = [(i+1)/5, 0.55, (i+2)/5, 0.95])

fig.text(0.45, 0.95, f'$\sigma$  = {eps}', fontsize = 15, weight = 'bold')
plt.savefig(f'./plots/app_idsyntetic_eps{eps}.pdf')
