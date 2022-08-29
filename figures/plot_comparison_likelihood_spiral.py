import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib

#sns.set_theme()
sns.set_context("paper",)
sns.set_style("ticks")
sns.set_style("whitegrid",rc={"grid.linewidth": 0.01})

N = 20000
std = 0.001
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

#*******************************************************************************

fig = plt.figure(figsize = (10.5, 2.8))


gs = GridSpec(1,2)


ax = fig.add_subplot(gs[0])
sns.lineplot(x = np.log(id_twonn[2][:]), y = id_twonn[0][:], marker = 'o')
#sns.lineplot(x = rs_gride[:], y = loglik_gride[:], marker = 'o', label = 'GRIDE')
ax.set_ylabel('ID', fontsize = 14)
ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 3, 10, 30, 100])))
ax.set_xticklabels(['$1\sigma$', '$3\sigma$', '$10\sigma$', '$30\sigma$', '$100\sigma$'])
ax.set_ylim(0.95, 2.1)
ax.axvspan(np.log(0.0015), np.log(0.01), alpha = 0.1, color = 'C0')
ax.axhline(1, label = 'true ID', color = 'black', linewidth = 0.5, linestyle = 'dashdot')
ax.legend(fontsize = 10)


ax = fig.add_subplot(gs[1])
sns.lineplot(x = rs_twonn[:], y = loglik_twonn[:], marker = 'o')
ax.set_xticks(np.log(0.006*np.array([1.5, 2, 3, 5])))
ax.set_xticklabels(['$1.5\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$'])
ax.set_ylim(-0.04, 1.06)
ax.set_ylabel('P(ID=2)', fontsize = 14)
ax.set_xlabel('$\overline{r}$', fontsize = 13)
ax.set_xticks(np.log(0.001*np.array([1, 2, 3, 5, 8])))
ax.set_xticklabels(['$1\sigma$', '$2\sigma$', '$3\sigma$', '$5\sigma$', '$8\sigma$'])
ax.set_xlim(np.log(0.0015), np.log(0.01))

ax.axhline(0.5, color = 'black', linewidth = 0.5, linestyle = 'dashed', label = 'equal odds')
ax.legend(fontsize = 10)
gs.tight_layout(fig, rect = [0., 0.0, 0.48, 0.93])

fig.text(0.225, 0.94, 'twoNN', fontsize = 15)


gs = GridSpec(1,2)
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
ax.legend(fontsize = 10)

ax = fig.add_subplot(gs[1])

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
ax.legend(fontsize = 9)

gs.tight_layout(fig, rect = [0.52, 0.0, 1, 0.93])

fig.text(0.75, 0.94, 'Gride', fontsize = 15)

plt.savefig('./plots/comparison_likelihood_spiral.pdf')
