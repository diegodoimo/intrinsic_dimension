from utils.estimators import return_id_scaling_gride
from utils.syntetic_datasets import *
from dadapy import IdEstimation
from sklearn.neighbors import NearestNeighbors

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


results = './results/syntetic_datasets'
for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)

    print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')

    print('mle')
    mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
    np.save(f'{results}/mle/mle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([mle_ids, mle_err, mle_rs]))

    print('twonn')
    ie = IdEstimation(coordinates=X)
    twonn_ids, twonn_err, twonn_rs = ie.return_id_scaling_2NN(N_min = 16)
    np.save(f'{results}/twonn/twonn_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([twonn_ids, twonn_err, twonn_rs]))

    print('gride')
    ie = IdEstimation(coordinates=X)
    gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 2048)
    np.save(f'{results}/gride/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))

N = 32000
eps=0
names = {
    'uniform2_0':    [uniform,     {'N':N,'D': 2,'d': 2,'eps': eps}],
    'uniform5_0':    [uniform,     {'N':N,'D': 5,'d': 5,'eps': eps}],
    'uniform10_0':    [uniform,     {'N':N,'D': 10,'d': 10,'eps': eps}],
    'uniform20_0':    [uniform,     {'N':N,'D': 20,'d': 20,'eps': eps}],
    'uniform50_0':    [uniform,     {'N':N,'D': 50,'d': 50,'eps': eps}],
}

for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)
    print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')
    gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 4096)
    np.save(f'{results}/gride/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))
