import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
import pandas as pd
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})




#*******************************************************************************
data_folder = './results/datasets/real'

gride_mnist = np.load(f'{data_folder}/npy/gride_mnist_6742.npy')
gride_isomap = np.load(f'{data_folder}/npy/gride_isomap_698.npy')
gride_isolet = np.load(f'{data_folder}/npy/gride_isolet_7797.npy')


twonn_mnist = np.load(f'{data_folder}/npy/2nn_mnist_6742.npy')
twonn_isomap = np.load(f'{data_folder}/npy/2nn_isomap_698.npy')
twonn_isolet = np.load(f'{data_folder}/npy/2nn_isolet_7797.npy')


#*******************************************************************************
data_folder = './results/datasets/real'
k1 = 5
k2 = 15
def estract_data(arr, remove_val):
    arr = np.delete(arr, np.where(arr[:, 0]<=val), axis = 0)
    return arr
geo_mnist = np.genfromtxt(f'{data_folder}/geomle_mnist_k{k1}_{k2}_nrep10_nboots20.txt', skip_header = 1)
val = np.unique(geo_mnist[:, 0])[1]+1
geo_mnist = estract_data(geo_mnist, remove_val=val)

danco_mnist = np.genfromtxt(f'{data_folder}/DANCo_mnist_N.txt', skip_header = 1)
danco_mnist = estract_data(danco_mnist, remove_val=val)

ess_mnist = np.genfromtxt(f'{data_folder}/ESS_mnist_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
ess_mnist = estract_data(ess_mnist, remove_val=val)


geo_isomap = np.genfromtxt(f'{data_folder}/geomle_isomap_k{k1}_{k2}_nrep10_nboots20.txt', skip_header = 1)
val = np.unique(geo_isomap[:, 0])[0]+1
geo_isomap = estract_data(geo_isomap, remove_val=val)

danco_isomap = np.genfromtxt(f'{data_folder}/DANCo_isomap_N.txt', skip_header = 1)
danco_isomap = estract_data(danco_isomap, remove_val=val)

ess_isomap = np.genfromtxt(f'{data_folder}/ESS_faces_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
ess_isomap = estract_data(ess_isomap, remove_val=val)

geo_isolet = np.genfromtxt(f'{data_folder}/geomle_isolet_k{k1}_{k2}_nrep10_nboots20.txt', skip_header = 1)
val = np.unique(geo_isolet[:, 0])[1]+1
geo_isolet = estract_data(geo_isolet, remove_val=val)

danco_isolet = np.genfromtxt(f'{data_folder}/DANCo_isolet_N.txt', skip_header = 1)
danco_isolet = estract_data(danco_isolet, remove_val=val)

ess_isolet = np.genfromtxt(f'{data_folder}/ESS_isolet_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
ess_isolet = estract_data(ess_isolet, remove_val=val)


#*******************************************************************************
fig = plt.figure(figsize = (9.2, 2.8))

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
x = np.unique(danco_mnist[:, 0])

ax.errorbar(x = x, y = gride_mnist[0][:len(x)][::-1], yerr = gride_mnist[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax, x=x, y = gride_mnist[0][:len(x)][::-1], label = 'Gride', marker = 'o', color = 'C0')

ax.errorbar(x = x, y = twonn_mnist[0][:len(x)][::-1], yerr = twonn_mnist[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax, x=x, y = twonn_mnist[0][:len(x)][::-1], label = 'twoNN', marker = 'o', color = 'C1')

df_danco  = pd.DataFrame(np.array([danco_mnist[:, 1], danco_mnist[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax, data = df_danco, x = 'n', y = 'ID', marker = 'o',label = 'DANCo',
                                linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')
df_geo  = pd.DataFrame(np.array([geo_mnist[:, 1], geo_mnist[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax, data = df_geo, x = 'n', y = 'ID', marker = 'o',label = 'GeoMLE',
                                linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3')
df_ess  = pd.DataFrame(np.array([ess_mnist[:, 1], ess_mnist[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax, data = df_ess, x = 'n', y = 'ID', marker = 'o',label = 'ESS',
                                    linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C4')

ax.legend(fontsize = 10, frameon = False, bbox_to_anchor=(0.5,.83), loc="upper left",)
ax.set_ylabel('ID', fontsize = 15)
ax.set_xlabel('n° data', fontsize = 14)
gs.tight_layout(fig, rect = [0.01, 0.01, 0.35, 0.99])


gs = GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
x = np.unique(danco_isomap[:, 0])

ax1.errorbar(x = x, y = gride_isomap[0][:len(x)][::-1], yerr = gride_isomap[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax1, x=x, y = gride_isomap[0][:len(x)][::-1], marker = 'o', color = 'C0')

ax1.errorbar(x = x, y = twonn_isomap[0][:len(x)][::-1], yerr = twonn_isomap[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax1, x=x, y = twonn_isomap[0][:len(x)][::-1], marker = 'o', color = 'C1')

df_danco  = pd.DataFrame(np.array([danco_isomap[:, 1], danco_isomap[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax1, data = df_danco, x = 'n', y = 'ID', marker = 'o',
                        linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')
df_geo  = pd.DataFrame(np.array([geo_isomap[:, 1], geo_isomap[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax1, data = df_geo, x = 'n', y = 'ID', marker = 'o',
                            linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3' )
df_ess  = pd.DataFrame(np.array([ess_isomap[:, 1], ess_isomap[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax1, data = df_ess, x = 'n', y = 'ID', marker = 'o',
                linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C4')
ax1.set_xlabel('n° data', fontsize = 14)
ax1.set_ylabel('')
gs.tight_layout(fig, rect = [0.37, 0.01, 0.66, 0.99])


gs = GridSpec(1, 1)
ax2 = fig.add_subplot(gs[0])
x = np.unique(danco_isolet[:, 0])

ax2.errorbar(x = x, y = gride_isolet[0][:len(x)][::-1], yerr = gride_isolet[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax2, x=x, y = gride_isolet[0][:len(x)][::-1], marker = 'o', color = 'C0')

ax2.errorbar(x = x, y = twonn_isolet[0][:len(x)][::-1], yerr = twonn_isolet[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax2, x=x, y = twonn_isolet[0][:len(x)][::-1], marker = 'o', color = 'C1')


df_danco  = pd.DataFrame(np.array([danco_isolet[:, 1], danco_isolet[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax2, data = df_danco, x = 'n', y = 'ID', marker = 'o',
                    linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')
df_geo  = pd.DataFrame(np.array([geo_isolet[:, 1], geo_isolet[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax2, data = df_geo, x = 'n', y = 'ID', marker = 'o',
                linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3')
df_ess  = pd.DataFrame(np.array([ess_isolet[:, 1], ess_isolet[:, 0]]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax2, data = df_ess, x = 'n', y = 'ID', marker = 'o',
                    linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C4' )
#ax.axhspan(16, 22, alpha = 0.15, color = 'darkgoldenrod')
#ax2.set_title('Isolet', fontsize = 15)
ax2.set_ylabel('')
ax2.set_xlabel('n° data', fontsize = 14)
gs.tight_layout(fig, rect = [0.7, 0.01, 1., 0.99])

fig.text(0.05, 0.92, 'a', fontsize = 14, fontweight = 'bold')
fig.text(0.37, 0.92, 'b', fontsize = 14, fontweight = 'bold')
fig.text(0.69, 0.92, 'c', fontsize = 14, fontweight = 'bold')

plt.savefig('./plots/real_datasets.pdf')
