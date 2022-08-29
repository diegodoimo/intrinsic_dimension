import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
import scipy.stats
import pandas as pd
from sklearn.neighbors import NearestNeighbors


#sns.set_theme()
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 0.01})



np.random.seed(42)

N = 200
factor = 8

X = np.random.uniform(size = (N, 2))
nbrs = NearestNeighbors(n_neighbors=N-1).fit(X)
dist, nns = nbrs.kneighbors(X)
indx = np.concatenate((np.array([0, 1, 2]), np.random.choice(np.arange(1, N), int(N/factor), replace = False)))
X_dec = X[indx]
nbrs = NearestNeighbors(n_neighbors=3).fit(X_dec)
dist1, nns_dec= nbrs.kneighbors(X_dec)



fig = plt.figure(figsize = (10, 2.7))
gs = GridSpec(1, 1)
xticks = np.arange(4)/4
ax = fig.add_subplot(gs[0])
sns.scatterplot(x = X[:, 0], y = X[:, 1], ax = ax, color = 'silver')
for i in range(3):
    nb1 = nns[i, 1]
    nb2 =  nns[i, 2]
    ax.plot([X[i, 0], X[nb1, 0]] , [X[i, 1], X[nb1, 1]], linestyle = '--', color = 'black', linewidth = 1)
    ax.plot([X[i, 0], X[nb2, 0]] , [X[i, 1], X[nb2, 1]], linestyle = '--', color = 'black', linewidth  = 1)
    sns.scatterplot(x = X[i:i+1, 0], y = X[i:i+1, 1], ax = ax, s = 100, color = 'C3', zorder = 10)
    sns.scatterplot(x = X[nb1:nb1+1, 0], y = X[nb1:nb1+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
    sns.scatterplot(x = X[nb2:nb2+1, 0], y = X[nb2:nb2+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
gs.tight_layout(fig, rect = [0.01, 0., 0.31, 1])

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(x = X_dec[:, 0], y = X_dec[:, 1], ax = ax, color = 'silver')
for i in range(3):
    nb1 = nns_dec[i, 1]
    nb2 =  nns_dec[i, 2]
    ax.plot([X[i, 0], X_dec[nb1, 0]] , [X_dec[i, 1], X_dec[nb1, 1]], linestyle = '--', color = 'black', linewidth = 1)
    ax.plot([X[i, 0], X_dec[nb2, 0]] , [X_dec[i, 1], X_dec[nb2, 1]], linestyle = '--', color = 'black', linewidth  = 1)
    sns.scatterplot(x = X_dec[i:i+1, 0], y = X_dec[i:i+1, 1], ax = ax, s = 100, color = 'C3', zorder = 10)
    sns.scatterplot(x = X_dec[nb1:nb1+1, 0], y = X_dec[nb1:nb1+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
    sns.scatterplot(x = X_dec[nb2:nb2+1, 0], y = X_dec[nb2:nb2+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
gs.tight_layout(fig, rect = [0.39, 0., 0.69, 1])

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
sns.scatterplot(x = X[:, 0], y = X[:, 1], ax = ax, color = 'silver')
for i in range(3):
    nb1 = nns[i, 1*factor]
    nb2 =  nns[i, 2*factor]
    ax.plot([X[i, 0], X[nb1, 0]] , [X[i, 1], X[nb1, 1]], linestyle = '--', color = 'black', linewidth = 1)
    ax.plot([X[i, 0], X[nb2, 0]] , [X[i, 1], X[nb2, 1]], linestyle = '--', color = 'black', linewidth  = 1)
    sns.scatterplot(x = X[i:i+1, 0], y = X[i:i+1, 1], ax = ax, s = 100, color = 'C3', zorder = 10)
    sns.scatterplot(x = X[nb1:nb1+1, 0], y = X[nb1:nb1+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
    sns.scatterplot(x = X[nb2:nb2+1, 0], y = X[nb2:nb2+1, 1], ax = ax, s = 50, color = 'C0', zorder = 10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(xticks)
ax.set_yticks(xticks)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal')
gs.tight_layout(fig, rect = [0.7, 0., 1., 1])

fig.text(0.01, 0.9, 'a', fontsize = 18, fontweight = 'bold')
fig.text(0.39, 0.9, 'b', fontsize = 18, fontweight = 'bold')
fig.text(0.7, 0.9, 'c', fontsize = 18, fontweight = 'bold')
plt.savefig('./plots/cartoon.pdf')
