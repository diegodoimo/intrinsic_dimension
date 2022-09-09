import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
import pandas as pd
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})


data_folder = './results/time_benchmarks'
k1 = 10
k2 = 40
nthr = '_16threads'

dancoN = np.genfromtxt(f'{data_folder}/DANCo_cifar_N.txt', skip_header = 1)
dancoP = np.genfromtxt(f'{data_folder}/DANCo_cifar_P.txt', skip_header = 1)

essN = np.genfromtxt(f'{data_folder}/ESS_cifar_N.csv', skip_header = 1, delimiter=',')[:, 1:]
essP = np.genfromtxt(f'{data_folder}/ESS_cifar_P.csv', skip_header = 1, delimiter = ',')[:, 1:]

data_folder = '../scripts/results/real_datasets/time_benchmark'

ncpu = 16     #this is the equivalent picasso speed on gride, twonn
grideN = np.load(f'{data_folder}/gride_cifarN_ncpu{ncpu}_thr16.npy')
grideP = np.load(f'{data_folder}/gride_cifarP_ncpu{ncpu}_thr16.npy')

twonnN = np.load(f'{data_folder}/twonn_cifarN_ncpu{ncpu}_thr16.npy')
twonnP = np.load(f'{data_folder}/twonn_cifarP_ncpu{ncpu}_thr16.npy')

mleN = np.load(f'{data_folder}/mle_cifarN_ncpu{ncpu}_thr16.npy')
mleP = np.load(f'{data_folder}/mle_cifarP_ncpu{ncpu}_thr16.npy')

geomleN = np.load(f'{data_folder}/geomle_cifarN_ncpu{ncpu}_thr16.npy')
geomleP = np.load(f'{data_folder}/geomle_cifarP_ncpu{ncpu}_thr16.npy')

data_folder = '../scripts/results/real_datasets/time_benchmark/ess_local_time'
essN = np.genfromtxt(f'{data_folder}/cifar_timeN_k10.txt')#, skip_header = 1, delimiter=',')[:, 1:]
essP = np.genfromtxt(f'{data_folder}/cifar_timeP_k10.txt')#, skip_header = 1, delimiter = ',')[:, 1:]
#*******************************************************************************

fig = plt.figure(figsize = (9.5, 2.8))

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
st=7

xticks = [1000, 3000, 10000, 30000]
xticklabels = ['$10^3$', '$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$']
x = 50000/grideN[:st, 0]
sns.lineplot(ax = ax, x=x, y = grideN[:st, 1], label = 'Gride', marker = 'X', markersize = 8, zorder = 10)
sns.lineplot(ax = ax, x=x, y = twonnN[:st, 1], label = 'TwoNN', marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoN[:st, 2], label = 'DANCo', marker = 'o')
sns.lineplot(ax = ax, x=x, y = essN[-st:, 1][::-1], label = 'ESS', marker = 'o')
sns.lineplot(ax = ax, x=x, y = mleN[:st, 1], label = 'MLE', marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleN[:st, 1], label = 'GeoMLE', marker = 'o')

ax.set_ylabel('time (sec.)', fontsize = 14)
ax.set_xlabel('N', fontsize = 14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.legend(fontsize = 10, frameon = False, bbox_to_anchor=(0.57,1.05), loc="upper right",)
gs.tight_layout(fig, rect = [0.24, 0.01, 0.58, 0.99])
plt.legend(fontsize = 10, bbox_to_anchor=(-1.1,.83), loc="upper left",)


st=5
gs = GridSpec(1, 1)
xticks = [3000, 10000, 30000, 100000]
xticklabels = ['$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$', '$10^5$']
yticks = [1, 10, 100, 1000]

ax = fig.add_subplot(gs[0])

x = 3*grideP[st:, 0]**2
sns.lineplot(ax = ax, x=x, y = grideP[st:, 1], marker = 'X', markersize = 8, zorder = 10)
sns.lineplot(ax = ax, x=x, y = twonnP[st:, 1], marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoP[st:, 2], marker = 'o')
sns.lineplot(ax = ax, x=x, y = essP[st:, 1], marker = 'o')
sns.lineplot(ax = ax, x=x, y = mleP[st:, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleP[st:, 1],  marker = 'o')

ax.set_ylabel('time (sec.)', fontsize = 14)
ax.set_xlabel('P', fontsize = 14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yticks(yticks)
#ax.set_yticklabels(yticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
gs.tight_layout(fig, rect = [0.64, 0.01, 0.97, 0.99])

plt.savefig('./plots/time_benchmarks_noID.pdf')






fig = plt.figure(figsize = (9.5, 2.8))

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
st=7

#yticks = [0.1, 1, 10, 100, 1000, 10000]
#yticklabels = ['0.02s', '0.13s', '1s', '8s', '1min', '8min', '1h',]
xticks = [1000, 3000, 10000, 30000]
#xticklabels = ['0.3k', '1k', '3k', '10k', '30k']
xticklabels = ['$10^3$', '$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$']
x = 50000/grideN[:st, 0]
sns.lineplot(ax = ax, x=x, y = grideN[:st, 2], label = 'Gride', marker = 'o')
sns.lineplot(ax = ax, x=x, y = twonnN[:st, 2], label = 'TwoNN', marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoN[:st, 1], label = 'DANCo', marker = 'o')
sns.lineplot(ax = ax, x=x, y = essN[:st, 1], label = 'ESS', marker = 'o')
sns.lineplot(ax = ax, x=x, y = mleN[:st, 2], label = 'MLE', marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleN[:st, 2], label = 'GeoMLE', marker = 'o')


ax.set_ylabel('ID', fontsize = 14)
ax.set_xlabel('N', fontsize = 14)
#ax.set_ylim(10**-2, 10**5)
#ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_yticks(yticks)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.legend(fontsize = 10, frameon = False, bbox_to_anchor=(0.57,1.05), loc="upper right",)

#gs.tight_layout(fig, rect = [0.01, 0.06, 0.3, 0.99])

gs.tight_layout(fig, rect = [0.27, 0.01, 0.6, 0.99])
plt.legend(fontsize = 10, bbox_to_anchor=(-1.1,.83), loc="upper left",)


st=5
gs = GridSpec(1, 1)
xticks = [3000, 10000, 30000, 100000]
xticklabels = ['$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$', '$10^5$']
yticks = [1, 10, 100, 1000]

ax = fig.add_subplot(gs[0])

x = 3*grideP[st:, 0]**2
sns.lineplot(ax = ax, x=x, y = grideP[st:, 2], marker = 'o')
sns.lineplot(ax = ax, x=x, y = twonnP[st:, 2], marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoP[st:, 1], marker = 'o')
sns.lineplot(ax = ax, x=x, y = essP[st:, 1], marker = 'o')
sns.lineplot(ax = ax, x=x, y = mleP[st:, 2],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleP[st:, 2],  marker = 'o')

ax.set_ylabel('ID', fontsize = 14)
ax.set_xlabel('P', fontsize = 14)
#ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_yticks(yticks)
#ax.set_yticklabels(yticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
gs.tight_layout(fig, rect = [0.66, 0.01, 0.99, 0.99])

plt.savefig('./plots/time_benchmarks_ID.pdf')


















"ID"
xticks = [1000, 3000, 10000, 30000]
#xticklabels = ['0.3k', '1k', '3k', '10k', '30k']
xticklabels = ['$10^3$', '$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$']
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
yticks = [10, 20, 50, 100, 200]
yticklabels = yticks
x = grideN[:st, 0]
sns.lineplot(ax = ax, x=x, y = grideN[:st, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = twonnN[:st, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoN[:st, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleN[:st, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = essN[:st, 3], marker = 'o')
ax.set_ylabel('ID', fontsize = 14)
ax.set_xlabel('n° data', fontsize = 14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
gs.tight_layout(fig, rect = [0.28, 0.06, 0.52, 0.99])



"intrinsic dimensions"
gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
x = grideP[st:, 0]
yticks = [10, 20, 50, 100, 200]
yticklabels = yticks
sns.lineplot(ax = ax, x=x, y = grideP[st:, 1], marker = 'o')
sns.lineplot(ax = ax, x=x, y = twonnP[st:, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = dancoP[st:, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = geomleP[st:, 1],  marker = 'o')
sns.lineplot(ax = ax, x=x, y = essP[st:, 3],  marker = 'o')
ax.set_xlabel('n° features', fontsize = 14)
ax.set_ylabel('ID', fontsize = 14)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
gs.tight_layout(fig, rect = [0.76, 0.06, 1., 1.])

fig.text(0.05, 0.94, 'a', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
fig.text(0.3, 0.94, 'b', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
fig.text(0.56, 0.94, 'c', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
fig.text(0.78, 0.94, 'd', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')

plt.savefig('./plots/time_benchmarks.pdf')










#
# data_folder = './results/time_benchmarks'
# k1 = 10
# k2 = 40
# nthr = '_16threads'
#
# "benchmark time"
# grideN = np.genfromtxt(f'{data_folder}/gride_cifarN{nthr}.txt', skip_header = 1)
# grideP = np.genfromtxt(f'{data_folder}/gride_cifarP{nthr}.txt', skip_header = 1)
#
# twonnN = np.genfromtxt(f'{data_folder}/2nn_simple_cifarN{nthr}.txt', skip_header = 1)
# twonnP = np.genfromtxt(f'{data_folder}/2nn_simple_cifarP{nthr}.txt', skip_header = 1)
#
# geomleN = np.genfromtxt(f'{data_folder}/geomle_cifarN_k{k1}_{k2}_nrep1_nboots20.txt', skip_header = 1)
# geomleP = np.genfromtxt(f'{data_folder}/geomle_cifarP_k{k1}_{k2}_nrep1_nboots20.txt', skip_header = 1)
#
# dancoN = np.genfromtxt(f'{data_folder}/DANCo_cifar_N.txt', skip_header = 1)
# dancoP = np.genfromtxt(f'{data_folder}/DANCo_cifar_P.txt', skip_header = 1)
#
# essN = np.genfromtxt(f'{data_folder}/ESS_cifar_N.csv', skip_header = 1, delimiter=',')[:, 1:]
# essP = np.genfromtxt(f'{data_folder}/ESS_cifar_P.csv', skip_header = 1, delimiter = ',')[:, 1:]
#
#
#
#
# data_folder = '../scripts/results/real_datasets/time_benchmark'
#
#
# grideN = np.load(f'{data_folder}/gride_cifarN_ncpu{ncpu}_thr16.npy')
# grideP = np.load(f'{data_folder}/gride_cifarP_ncpu{ncpu}_thr16.npy')
#
# twonnN = np.load(f'{data_folder}/twonn_cifarN_ncpu32_thr16.npy')
# twonnP = np.load(f'{data_folder}/twonn_cifarP_ncpu32_thr16.npy')
#
#
# mleN = np.load(f'{data_folder}/mle_cifarN_ncpu32_thr16.npy')
# mleP = np.load(f'{data_folder}/mle_cifarP_ncpu32_thr16.npy')
#
#
# geomleN = np.genfromtxt(f'{data_folder}/geomle_cifarN_k{k1}_{k2}_nrep1_nboots20.txt', skip_header = 1)
# geomleP = np.genfromtxt(f'{data_folder}/geomle_cifarP_k{k1}_{k2}_nrep1_nboots20.txt', skip_header = 1)
#
# dancoN = np.genfromtxt(f'{data_folder}/DANCo_cifar_N.txt', skip_header = 1)
# dancoP = np.genfromtxt(f'{data_folder}/DANCo_cifar_P.txt', skip_header = 1)
#
# essN = np.genfromtxt(f'{data_folder}/ESS_cifar_N.csv', skip_header = 1, delimiter=',')[:, 1:]
# essP = np.genfromtxt(f'{data_folder}/ESS_cifar_P.csv', skip_header = 1, delimiter = ',')[:, 1:]
#
#
#
#
# #*******************************************************************************
#
# fig = plt.figure(figsize = (10, 2.9))
#
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# st=7
#
# yticks = [0.1, 1, 10, 100, 1000, 10000]
# #yticklabels = ['0.02s', '0.13s', '1s', '8s', '1min', '8min', '1h',]
# xticks = [1000, 3000, 10000, 30000]
# #xticklabels = ['0.3k', '1k', '3k', '10k', '30k']
# xticklabels = ['$10^3$', '$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$']
# x = grideN[:st, 0]
# sns.lineplot(ax = ax, x=x, y = grideN[:st, 2], label = 'Gride', marker = 'o')
# sns.lineplot(ax = ax, x=x, y = twonnN[:st, 2], label = 'twoNN', marker = 'o')
# sns.lineplot(ax = ax, x=x, y = dancoN[:st, 2], label = 'DANCo', marker = 'o')
# sns.lineplot(ax = ax, x=x, y = geomleN[:st, 3], label = 'GeoMLE', marker = 'o')
# sns.lineplot(ax = ax, x=x, y = essN[:st, 2], label = 'ESS', marker = 'o')
# ax.set_ylabel('time', fontsize = 14)
# ax.set_xlabel('n°  data', fontsize = 14)
# ax.set_ylim(10**-2, 10**5)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_yticks(yticks)
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
# ax.legend(fontsize = 10, frameon = False, bbox_to_anchor=(0.57,1.05), loc="upper right",)
#
# gs.tight_layout(fig, rect = [0.01, 0.06, 0.3, 0.99])
#
#
# "ID"
# xticks = [1000, 3000, 10000, 30000]
# #xticklabels = ['0.3k', '1k', '3k', '10k', '30k']
# xticklabels = ['$10^3$', '$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$']
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# yticks = [10, 20, 50, 100, 200]
# yticklabels = yticks
# x = grideN[:st, 0]
# sns.lineplot(ax = ax, x=x, y = grideN[:st, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = twonnN[:st, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = dancoN[:st, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = geomleN[:st, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = essN[:st, 3], marker = 'o')
# ax.set_ylabel('ID', fontsize = 14)
# ax.set_xlabel('n° data', fontsize = 14)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticklabels)
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
# gs.tight_layout(fig, rect = [0.28, 0.06, 0.52, 0.99])
#
#
# st=5
# gs = GridSpec(1, 1)
# xticks = [3000, 10000, 30000, 100000]
# xticklabels = ['$3 \cdot 10^3$', '$10^4$', '$3 \cdot 10^4$', '$10^5$']
# yticks = [1, 10, 100, 1000]
#
# ax = fig.add_subplot(gs[0])
# x = grideP[st:, 0]
# sns.lineplot(ax = ax, x=x, y = grideP[st:, 2],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = twonnP[st:, 2],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = dancoP[st:, 2],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = geomleP[st:, 3],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = essP[st:, 2],  marker = 'o')
#
# ax.set_ylabel('time', fontsize = 14)
# ax.set_xlabel('n° features', fontsize = 14)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_yticks(yticks)
# #ax.set_yticklabels(yticklabels)
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
# gs.tight_layout(fig, rect = [0.54, 0.06, 0.78, 0.99])
#
#
#
# "intrinsic dimensions"
# gs = GridSpec(1, 1)
# ax = fig.add_subplot(gs[0])
# x = grideP[st:, 0]
# yticks = [10, 20, 50, 100, 200]
# yticklabels = yticks
# sns.lineplot(ax = ax, x=x, y = grideP[st:, 1], marker = 'o')
# sns.lineplot(ax = ax, x=x, y = twonnP[st:, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = dancoP[st:, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = geomleP[st:, 1],  marker = 'o')
# sns.lineplot(ax = ax, x=x, y = essP[st:, 3],  marker = 'o')
# ax.set_xlabel('n° features', fontsize = 14)
# ax.set_ylabel('ID', fontsize = 14)
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_yticks(yticks)
# ax.set_yticklabels(yticklabels)
# ax.set_xticks(xticks)
# ax.set_xticklabels(xticklabels)
# gs.tight_layout(fig, rect = [0.76, 0.06, 1., 1.])
#
# fig.text(0.05, 0.94, 'a', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
# fig.text(0.3, 0.94, 'b', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
# fig.text(0.56, 0.94, 'c', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
# fig.text(0.78, 0.94, 'd', fontsize = 14, fontweight = 'bold', va = 'center', ha = 'left')
#
# plt.savefig('./plots/time_benchmarks.pdf')
