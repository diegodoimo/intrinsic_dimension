import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib
from sklearn.neighbors import NearestNeighbors


#sns.set_theme()
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 0.01})



#***********************************************************
# uniform data
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



#*********************************************************
#spiral data
N = 20000
std = 0.005
s = np.random.uniform(size = N)
t = s**0.5
x = np.array([t*np.cos(6*t), t*np.sin(6*t)]).T
x+=np.random.normal(scale = std, size = (N, 2))


loglik_twonn = np.load(f'./results/spiral/loglik_twonn_spiral.npy')
rs_twonn = np.log(np.load(f'./results/spiral/rs_twonn_spiral.npy'))
loglik_gride = np.load(f'./results/spiral/loglik_gride_spiral.npy')
rs_gride = np.log(np.load(f'./results/spiral/rs_gride_spiral.npy'))

id_twonn = np.load(f'./results/spiral/id_twonn_spiral20k.npy')
id_gride = np.load(f'./results/spiral/id_gride_spiral20k.npy')

factor = 20
X_spiral = x[np.random.choice(np.arange(1, N), int(N/factor), replace = False)]

#*******************************************************************************






fig = plt.figure(figsize = (9.5, 6.5))

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
ax.set_title('twoNN', fontsize = 15)
gs.tight_layout(fig, rect = [0.0, 0.68, 0.23, 1])

gs = GridSpec(1, 1)
xticks = np.arange(4)/4
ax = fig.add_subplot(gs[0])
sns.scatterplot(x = X_spiral[:, 0], y = X_spiral[:, 1], s = 10, color = 'black')
#ax.axis('off')
gs.tight_layout(fig, rect = [0.53, 0.65, 0.82, 0.95])

fig.text(0.0, 0.95, 'a', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')
fig.text(0.52, 0.95, 'b', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')





#*******************************************************************************


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
ax.set_title('Decimation', fontsize = 15)
gs.tight_layout(fig, rect = [0.0, 0.36, 0.23, 0.68])


gs = GridSpec(1,1)
ax = fig.add_subplot(gs[0])
sns.lineplot(x = np.log(id_twonn[2][:]), y = id_twonn[0][:], marker = 'o')
#sns.lineplot(x = rs_gride[:], y = loglik_gride[:], marker = 'o', label = 'GRIDE')
ax.set_ylabel('ID', fontsize = 14)
#ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 3, 10, 30, 100])))
ax.set_xticklabels(['$1\sigma$', '$3\sigma$', '$10\sigma$', '$30\sigma$', '$100\sigma$'])
ax.set_ylim(0.95, 2.1)
ax.axvspan(np.log(0.0015), np.log(0.01), alpha = 0.1, color = 'C0')
ax.axhline(1, label = 'true ID', color = 'black', linewidth = 0.5, linestyle = 'dashdot')
ax.legend(fontsize = 11)
gs.tight_layout(fig, rect = [0.32, 0.33, 0.64, 0.64])

gs = GridSpec(1,1)
ax = fig.add_subplot(gs[0])
sns.lineplot(x = rs_twonn[:], y = loglik_twonn[:], marker = 'o')
ax.set_xticks(np.log(0.006*np.array([1.5, 2, 3, 5])))
ax.set_xticklabels(['$1.5\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$'])
ax.set_ylim(-0.04, 1.06)
ax.set_ylabel('P(ID=2)', fontsize = 14)
#ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 2, 3, 5, 8])))
ax.set_xticklabels(['$1\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$', '$8\sigma$'])
ax.set_xlim(np.log(0.0015), np.log(0.01))

ax.axhline(0.5, color = 'black', linewidth = 0.5, linestyle = 'dashed', label = 'equal odds')
ax.legend(fontsize = 11)
gs.tight_layout(fig, rect = [0.68, 0.33, 1, 0.64])


fig.text(0.0, 0.64, 'c', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')
fig.text(0.32, 0.64, 'd', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')
fig.text(0.68, 0.64, 'e', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')


#*******************************************************************************
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
ax.set_title('Gride', fontsize = 15)
gs.tight_layout(fig, rect = [0.0, 0.06, 0.23, .38])


gs = GridSpec(1,1)
ax = fig.add_subplot(gs[0])
sns.lineplot(x = np.log(id_gride[2][:]), y = id_gride[0][:], marker = 'o', color = 'C1')
#sns.lineplot(x = rs_gride[:], y = loglik_gride[:], marker = 'o', label = 'GRIDE')
ax.set_ylabel('ID', fontsize = 14)
ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 3, 10, 30, 100])))
ax.set_xticklabels(['$1\sigma$', '$3\sigma$', '$10\sigma$', '$30\sigma$', '$100\sigma$'])
ax.set_ylim(0.95, 2.1)
ax.axvspan(np.log(0.0015), np.log(0.01), alpha = 0.1, color = 'C1')
ax.axhline(1, label = 'true ID', color = 'black', linewidth = 0.5, linestyle = 'dashdot')
ax.legend(fontsize = 11)
gs.tight_layout(fig, rect = [0.32, 0., .64, 0.34])

gs = GridSpec(1,1)
ax = fig.add_subplot(gs[0])
sns.lineplot(x = rs_gride[:], y = loglik_gride[:], marker = 'o', color = 'C1')
ax.set_xticks(np.log(0.006*np.array([1.5, 2, 3, 5])))
ax.set_xticklabels(['$1.5\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$'])
ax.set_ylim(-0.04, 1.06)
ax.set_ylabel('P(ID =2)', fontsize = 14)
ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 2, 3, 5, 8])))
ax.set_xticklabels(['$1\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$', '$8\sigma$'])
ax.set_xlim(np.log(0.0015), np.log(0.01))
ax.axhline(0.5, color = 'black', linewidth = 0.5, linestyle = 'dashed', label = 'equal odds')
ax.legend(fontsize = 11)
gs.tight_layout(fig, rect = [0.68, 0., 1., 0.34])

fig.text(0.0, 0.34, 'f', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')
fig.text(0.32, 0.34, 'g', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')
fig.text(0.68, 0.34, 'h', fontsize = 15, weight = 'bold', ha = 'left', va = 'top')


plt.savefig('./plots/cartoon2.pdf')






#*******************************************************************************
