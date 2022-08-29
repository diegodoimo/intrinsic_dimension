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

#*******************************************************************************
from scipy.special import betaln

"likelihood TwoNN"
def compute_loglik_twoNN(mus, d):
    N = mus.shape[0]
    loglik = N * np.log(d) - (d + 1)* np.sum(np.log(mus), axis = 0)

    return loglik


"likelihood GRIDE"
def compute_loglik_gride(mus, d, n):
    num = np.log(d) + (n - 1) * np.log(mus**d - 1)

    den = ((2*n - 1) * d + 1) * np.log(mus)  + betaln(n,n)

    loglik = num - den
    return np.exp(loglik)

mus = np.linspace(1, 3, 1000)
mus.shape[0]
d = 1
n = 1

pdf1 = []
pdf4 = []
pdf16 = []
pdf = []

ns = [1, 16, 256]
for n in ns:
    pdf_tmp = []
    dims = [1, 2, 5, 10]
    for d in dims:
        pdf_tmp.append(compute_loglik_gride(mus, d, n))
    pdf.append(pdf_tmp)



n_plots = len(ns)
fig = plt.figure(figsize = (10, 2.7))
gs = GridSpec(1, 3)
for j in range(n_plots):
    ax = fig.add_subplot(gs[j])
    for i in range(4):
        sns.lineplot(x = mus, y = pdf[j][i], ax = ax, label= f'd = {dims[i]}', linewidth = 1.5)
    ax.set_xlabel('$\mu_n$', fontsize = 13)
    if ns[j]==1:
        ax.set_ylabel('$p(\mu_n|d)$', fontsize = 12)
    ax.legend(fontsize = 13)
    ax.set_title(f'$n_1 = {ns[j]}$', fontsize = 14)

plt.subplots_adjust(left=0.02,
                    bottom=0.01,
                    right=0.91,
                    top=0.98,
                    wspace=0.5,
                    hspace=0.)

plt.savefig('./plots/shape_pdf.pdf')
