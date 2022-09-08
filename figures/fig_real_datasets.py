import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
import matplotlib
import pandas as pd
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 1})




#*******************************************************************************

ess_local = True
data_folder = '../scripts/results/real_datasets'

#data_folder = './results/datasets/real'

gride_mnist = np.load(f'{data_folder}/gride_mnist.npy')
gride_isomap = np.load(f'{data_folder}/gride_isomap.npy')
gride_isolet = np.load(f'{data_folder}/gride_isolet.npy')


twonn_mnist = np.load(f'{data_folder}/twonn_mnist.npy')
twonn_isomap = np.load(f'{data_folder}/twonn_isomap.npy')
twonn_isolet = np.load(f'{data_folder}/twonn_isolet.npy')


mle_mnist = np.load(f'{data_folder}/mle_mnist.npy')
mle_isomap = np.load(f'{data_folder}/mle_isomap.npy')
mle_isolet = np.load(f'{data_folder}/mle_isolet.npy')






def gen_ids(arr, range_max=4):
    ids, std = [], []
    for i in range(range_max):
        ndata = arr[0, 0]
        ndec = 2**i

        mask = arr[:, 0] == int(ndata/ndec)
        #print(mask)
        #print(  arr[mask][:, 1] )
        ids.append( np.mean( arr[mask][:, 1])   )
        std.append( np.std( arr[mask][:, 1])/np.sum(mask**0.5)   )
    return np.array([ids, std])



if ess_local:

    path = '/home/diego/ricerca/past/IDestimators/repo/scripts/results/real_datasets/ess_local'

    arr = np.genfromtxt(f'{path}/mnist_ones_k10.txt')
    ess_mnist = gen_ids(arr)

    arr = np.genfromtxt(f'{path}/isolet_k10.txt')
    ess_isolet = gen_ids(arr)

    arr = np.genfromtxt(f'{path}/isomap_k10.txt')
    ess_isomap = gen_ids(arr)
else:
    data_folder = './results/datasets/real'
    arr = np.genfromtxt(f'{data_folder}/ESS_mnist_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
    ess_mnist = gen_ids(arr)

    arr = np.genfromtxt(f'{data_folder}/ESS_faces_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
    ess_isomap = gen_ids(arr)

    arr = np.genfromtxt(f'{data_folder}/ESS_isolet_results.csv', skip_header = 1, delimiter = ',')[:, 1:]
    ess_isolet = gen_ids(arr)




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

geo_isomap = np.genfromtxt(f'{data_folder}/geomle_isomap_k{k1}_{k2}_nrep10_nboots20.txt', skip_header = 1)
val = np.unique(geo_isomap[:, 0])[0]+1
geo_isomap = estract_data(geo_isomap, remove_val=val)

danco_isomap = np.genfromtxt(f'{data_folder}/DANCo_isomap_N.txt', skip_header = 1)
danco_isomap = estract_data(danco_isomap, remove_val=val)

geo_isolet = np.genfromtxt(f'{data_folder}/geomle_isolet_k{k1}_{k2}_nrep10_nboots20.txt', skip_header = 1)
val = np.unique(geo_isolet[:, 0])[1]+1
geo_isolet = estract_data(geo_isolet, remove_val=val)

danco_isolet = np.genfromtxt(f'{data_folder}/DANCo_isolet_N.txt', skip_header = 1)
danco_isolet = estract_data(danco_isolet, remove_val=val)


data_folder = '../scripts/results/real_datasets'

geomle_mnist = np.load(f'{data_folder}/geomle_mnist_k20_55.npy')
geomle_isomap = np.load(f'{data_folder}/geomle_isomap_k20_55.npy')
geomle_isolet = np.load(f'{data_folder}/geomle_isolet_k20_55.npy')



#*******************************************************************************
fig = plt.figure(figsize = (10, 2.8))

gs = GridSpec(1, 1)
ax = fig.add_subplot(gs[0])
x = np.log( np.unique(danco_mnist[:, 0]) )


ax.errorbar(x = x-np.log(1.5), y = gride_mnist[0][:len(x)][::-1], yerr = gride_mnist[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax, x=x-np.log(1.5), y = gride_mnist[0][:len(x)][::-1], label = 'Gride', marker = 'o', color = 'C0')

ax.errorbar(x = x-np.log(1.5), y = twonn_mnist[0][:len(x)][::-1], yerr = twonn_mnist[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax, x=x-np.log(1.5), y = twonn_mnist[0][:len(x)][::-1], label = 'twoNN', marker = 'o', color = 'C1')

df_danco  = pd.DataFrame(np.array([danco_mnist[:, 1], np.log(danco_mnist[:, 0]/5.5) ]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax, data = df_danco, x = 'n', y = 'ID', marker = 'o',label = 'DANCo',
                                linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')


ax.errorbar(x = x-np.log(5.5), y = ess_mnist[0][:len(x)][::-1], yerr = ess_mnist[1][:len(x)], color = 'C3')
sns.lineplot(ax = ax, x=x-np.log(5.5), y = ess_mnist[0][:len(x)][::-1], label = 'ESS', marker = 'o', color = 'C3')

# df_ess  = pd.DataFrame(np.array([ess_mnist[:, 1], ess_mnist[:, 0]]).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax, data = df_ess, x = 'n', y = 'ID', marker = 'o',label = 'ESS',
#                                     linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3')

ax.errorbar(x = x-np.log(5.5), y = mle_mnist[0][:len(x)][::-1], yerr = mle_mnist[1][:len(x)], color = 'C4')
sns.lineplot(ax = ax, x=x-np.log(5.5), y = mle_mnist[0][:len(x)][::-1], marker = 'o', color = 'C4', label = 'MLE')

# df_geo  = pd.DataFrame(np.array([geo_mnist[:, 1], np.log(geo_mnist[:, 0]/10) ]).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax, data = df_geo, x = 'n', y = 'ID', marker = 'o',label = 'GeoMLE',
#                                 linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C5')



ax.errorbar(x = x-np.log(38), y = geomle_mnist[0][:len(x)][::-1], yerr = mle_mnist[1][:len(x)], color = 'C5')
sns.lineplot(ax = ax, x=x-np.log(38), y = geomle_mnist[0][:len(x)][::-1], marker = 'o', color = 'C5')



xticks = [30, 300, 3000]
xticklabels = [30, 300, 3000]
ax.set_xticks(np.log(xticks))
ax.set_xticklabels(xticklabels)

ax.set_ylabel('ID', fontsize = 15)
ax.set_xlabel('$N/\overline{k}$', fontsize = 14)

gs.tight_layout(fig, rect = [0.2, 0.01, 0.48, 0.99])
plt.legend(fontsize = 10, bbox_to_anchor=(-1.,.83), loc="upper left",)

gs = GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
x = np.log(np.unique(danco_isomap[:, 0]))

ax1.errorbar(x = x-np.log(1.5), y = gride_isomap[0][:len(x)][::-1], yerr = gride_isomap[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax1, x=x-np.log(1.5), y = gride_isomap[0][:len(x)][::-1], marker = 'o', color = 'C0')

ax1.errorbar(x = x-np.log(1.5), y = twonn_isomap[0][:len(x)][::-1], yerr = twonn_isomap[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax1, x=x-np.log(1.5), y = twonn_isomap[0][:len(x)][::-1], marker = 'o', color = 'C1')

df_danco  = pd.DataFrame(np.array([danco_isomap[:, 1], np.log(danco_isomap[:, 0]/5.5) ]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax1, data = df_danco, x = 'n', y = 'ID', marker = 'o',
                        linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')


ax1.errorbar(x = x-np.log(5.5), y = ess_isomap[0][:len(x)][::-1], yerr = ess_isomap[1][:len(x)], color = 'C3')
sns.lineplot(ax = ax1, x=x-np.log(5.5), y = ess_isomap[0][:len(x)][::-1], marker = 'o', color = 'C3')

# df_ess  = pd.DataFrame(np.array([ess_isomap[:, 1], ess_isomap[:, 0]]).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax1, data = df_ess, x = 'n', y = 'ID', marker = 'o',
#                 linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3')

ax1.errorbar(x = x-np.log(5.5), y = mle_isomap[0][:len(x)][::-1], yerr = mle_isomap[1][:len(x)], color = 'C4')
sns.lineplot(ax = ax1, x=x-np.log(5.5), y = mle_isomap[0][:len(x)][::-1], marker = 'o', color = 'C4')


# df_geo  = pd.DataFrame(np.array([geo_isomap[:, 1], np.log(geo_isomap[:, 0]/10)  ]).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax1, data = df_geo, x = 'n', y = 'ID', marker = 'o',
#                             linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C5' )

ax1.errorbar(x = x-np.log(38), y = geomle_isomap[0][:len(x)][::-1], yerr = mle_mnist[1][:len(x)], color = 'C5')
sns.lineplot(ax = ax1, x=x-np.log(38), y = geomle_isomap[0][:len(x)][::-1], marker = 'o', color = 'C5')


xticks = [3, 10, 30, 100, 300]
ax1.set_xticks(np.log(xticks))
ax1.set_xticklabels(xticks)
#ax1.set_xlim(70, 850)

ax1.set_xlabel('$N/\overline{k}$', fontsize = 14)
ax1.set_ylabel('')
gs.tight_layout(fig, rect = [0.49, 0.01, 0.73, 0.99])





gs = GridSpec(1, 1)
ax2 = fig.add_subplot(gs[0])
x = np.log(np.unique(danco_isolet[:, 0]))

ax2.errorbar(x = x-np.log(1.5), y = gride_isolet[0][:len(x)][::-1], yerr = gride_isolet[1][:len(x)], color = 'C0')
sns.lineplot(ax = ax2, x=x, y = gride_isolet[0][:len(x)][::-1], marker = 'o', color = 'C0')

ax2.errorbar(x = x-np.log(1.5), y = twonn_isolet[0][:len(x)][::-1], yerr = twonn_isolet[1][:len(x)], color = 'C1')
sns.lineplot(ax = ax2, x=x-np.log(1.5), y = twonn_isolet[0][:len(x)][::-1], marker = 'o', color = 'C1')

df_danco  = pd.DataFrame(np.array([danco_isolet[:, 1], np.log(danco_isolet[:, 0]/5.5) ]).T, columns = ['ID', 'n'])
sns.lineplot(ax = ax2, data = df_danco, x = 'n', y = 'ID', marker = 'o',
                    linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C2')

ax2.errorbar(x = x-np.log(5.5), y = ess_isolet[0][:len(x)][::-1], yerr = ess_isolet[1][:len(x)], color = 'C3')
sns.lineplot(ax = ax2, x=x-np.log(5.5), y = ess_isolet[0][:len(x)][::-1], marker = 'o', color = 'C3')
#
# df_ess  = pd.DataFrame(np.array([ess_isolet[:, 1], ess_isolet[:, 0]]).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax2, data = df_ess, x = 'n', y = 'ID', marker = 'o',
#                     linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C3' )

ax2.errorbar(x = x-np.log(5.5), y = mle_isolet[0][:len(x)][::-1], yerr = mle_isolet[1][:len(x)], color = 'C4')
sns.lineplot(ax = ax2, x=x-np.log(5.5), y = mle_isolet[0][:len(x)][::-1], marker = 'o', color = 'C4')

# df_geo  = pd.DataFrame(np.array([geo_isolet[:, 1], np.log(geo_isolet[:, 0]/38) ] ).T, columns = ['ID', 'n'])
# sns.lineplot(ax = ax2, data = df_geo, x = 'n', y = 'ID', marker = 'o',
#                 linewidth = '1.5', ci = 'sd', err_style = 'bars', color = 'C5')


ax2.errorbar(x = x-np.log(38), y = geomle_isolet[0][:len(x)][::-1], yerr = mle_mnist[1][:len(x)], color = 'C5')
sns.lineplot(ax = ax2, x=x-np.log(38), y = geomle_isolet[0][:len(x)][::-1], marker = 'o', color = 'C5')


ax2.set_ylabel('')
ax2.set_xlabel('$N/\overline{k}$', fontsize = 14)
ax2.set_xticklabels([])

xticks = [30, 300, 3000]
ax2.set_xticks(np.log(xticks))
ax2.set_xticklabels(xticks)

gs.tight_layout(fig, rect = [0.75, 0.01, 1., 0.99])

fig.text(0.23, 0.92, 'a', fontsize = 14, fontweight = 'bold')
fig.text(0.47, 0.92, 'b', fontsize = 14, fontweight = 'bold')
fig.text(0.74, 0.92, 'c', fontsize = 14, fontweight = 'bold')


np.log(38)

plt.savefig('./plots/real_datasets1.pdf')
