from utils.estimators import return_id_scaling_gride
from utils.syntetic_datasets import *

from dadapy import IdEstimation
from skdim.id import DANCo, ESS
from skdim._commonfuncs import efficient_indnComb
from sklearn.neighbors import NearestNeighbors


from ess import ESS



# estimator = ESS()
#
# X = np.random.normal(size =(16000, 3))
# X.shape
#
# estimator = ESS().fit(X, n_neighbors= 30)
#
# estimator.dimension_

# estimator = ESS()
# estimator.set_params(n_neighbors=30)
# estimator.get_params().keys()


# estimator = ESS(neighbors=30)
# estimator._N_NEIGHBORS =
#
# groups = efficient_indnComb(1000, 2, np.random)
#
# groups
# groups[:,0]
# np.where(groups[:, 0]==5)
#
# len(groups)



N = 16000
eps = 0.01
names = {
        'uniform20':    [uniform,       {'N':N,'D': 20,'d': 5,'eps': eps}],
        'normal':       [normal,        {'N':N,'D': 3,'d': 2,'eps': eps}],
        'normal20':     [normal,        {'N':N,'D': 20,'d': 2,'eps': eps}],
        'sphere':       [sphere,        {'N':N,'D': 15,'d': 10,'eps': eps}],
        'spiral1d':     [spiral1d,      {'N':N,'D': 3,'d': 1,'eps': eps}],
        'swissroll':    [swissroll,     {'N':N,'D': 3,'d': 2,'eps': eps}],
        'moebius':      [moebius,       {'N':N,'D': 6,'d': 2,'eps': eps}],
        'paraboloid':   [paraboloid,    {'N':N,'D': 30,'d': 9,'eps': eps}],
        'nonlinear':    [nonlinear,     {'N':N,'D': 36,'d': 6,'eps': eps}]
}


for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)




results = './results'
#decimation to roughly reproduce what is done in TwoNN
def compute_id_decimation(estimator, X, N_min, k = None, nrep = None):

    N = X.shape[0]

    print(f'nrep = {nrep}')
    intrinsic_dim = np.zeros(Nsubsets.shape[0])
    intrinsic_dim_err = np.zeros(Nsubsets.shape[0])
    intrinsic_dim_scale = np.zeros((Nsubsets.shape[0]))

    for i, N_subset in enumerate(Nsubsets):
        print(N_subset)
        decimation = N_subset / N
        #nrep = int(np.rint(1.0 / decimation))
        ids_tmp = []
        rs_tmp = []
        for j in range(nrep[i]):
            idx = np.random.choice(N, size=N_subset, replace=False)
            X_decimated = X[idx]

            if k is not None:
                nbrs = NearestNeighbors(n_neighbors=k).fit(X_decimated)
                distances, dist_indices = nbrs.kneighbors(X_decimated)
                rs_tmp.append(np.mean(distances[:, -1]))

            d = estimator.fit(X, n_neighbors=k) #'K' parameter to be set
            ids_tmp.append(d.dimension_)


        intrinsic_dim[i] = np.mean(ids_tmp)
        intrinsic_dim_err[i] = np.std(ids_tmp) / len(ids_tmp) ** 0.5
        if k is not None:
            intrinsic_dim_scale[i] = np.mean(rs_tmp)

    return intrinsic_dim, intrinsic_dim_err, intrinsic_dim_scale


for i, (key, value) in enumerate(names.items()):
    if i ==0:
        func = value[0]
        kwargs = value[1]
        X = func(**kwargs)

        k =10
        N = X.shape[0]
        Nsubsets = np.round(N / np.array([2**i for i in range(15)])).astype(int)
        Nsubsets = Nsubsets[Nsubsets > N_min]

        nrep = [3 for i in range(len(Nsubsets))]
        nrep[0] = 1
        nrep[1] = 2
        print('danco')
        estimator = DANCo(k=k)
        danco_ids, danco_err, danco_rs = compute_id_decimation(estimator, X, N_min=64, k = k, nrep = nrep) #k is the default in Danco
        np.save(f'{results}/danco_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}_k{k}.npy', np.array([danco_ids, danco_err, danco_rs]))

        k = 30
        print('ess')
        nrep = np.ones(len(Nsubsets))
        estimator = ESS()
        danco_ids, danco_err, danco_rs = compute_id_decimation(estimator, X, N_min=64, k = k, nrep = nrep)
        np.save(f'{results}/ess_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}_k{k}.npy', np.array([mle_ids, mle_err, mle_rs]))






results = './results'
for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)

    print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')

    mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
    print(mle_ids)#, np.array([mle_ids, mle_err, mle_rs]))
    np.save(f'{results}/mle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([mle_ids, mle_err, mle_rs]))

    ie = IdEstimation(coordinates=X)
    twonn_ids, twonn_err, twonn_rs = ie.return_id_scaling_2NN(N_min = 8)
    print(twonn_ids)
    np.save(f'{results}/twonn_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([twonn_ids, twonn_err, twonn_rs]))

    ie = IdEstimation(coordinates=X)
    gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 2048)
    print(gride_ids)
    np.save(f'{results}/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))



N = 32000
eps=0
names = {
    'uniform2_0':    [uniform,     {'N':N,'D': 2,'d': 2,'eps': eps}],
    'uniform5_0':    [uniform,     {'N':N,'D': 5,'d': 5,'eps': eps}],
    'uniform10_0':    [uniform,     {'N':N,'D': 10,'d': 10,'eps': eps}],
    'uniform20_0':    [uniform,     {'N':N,'D': 20,'d': 20,'eps': eps}],
    'uniform50_0':    [uniform,     {'N':N,'D': 50,'d': 50,'eps': eps}],
}


N = 32000
eps=0
names = {
    'uniform2_0':    [uniform,     {'N':N,'D': 2,'d': 2,'eps': eps}],
    'normal':       [normal,        {'N':N,'D': 3,'d': 2,'eps': eps}],
}
for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)
    print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')
    mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True, unbiased = True)
    print(mle_ids)
