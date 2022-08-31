from utils.estimators import return_id_scaling_gride, return_id_scaling_mle
from utils.syntetic_datasets import *
from dadapy import IdEstimation
from sklearn.neighbors import NearestNeighbors
from utils.geomle import geomle
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default = 'mnist', type = str)
parser.add_argument('--algo', default = None, type = str)
parser.add_argument('--nrep', default = 10, type = int)
parser.add_argument('--nbootstrap', default = 20, type = int)
parser.add_argument('--data_folder', default='/home/diego/ricerca/datasets', type=str)
parser.add_argument('--k1', default=5, type=int)
parser.add_argument('--k2', default=15, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--results_folder', default='./results/syntetic_datasets', type=str)
parser.add_argument('--uniform_gride', action = 'store_true')
args = parser.parse_args()

#*******************************************************************************
rng = np.random.default_rng(2022)
rng.random(args.seed)

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
#mle scaling decimating the dataset
for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)
    print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')
    print('mle')
    k1 = 10
    mle_ids, mle_err, mle_rs = return_id_scaling_mle(X, N_min = 16, k1 = k1, unbiased = False)
    print(mle_ids)
    np.save(f'{results}/mle/mle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}_nscaling_k{k1}.npy', np.array([mle_ids, mle_err, mle_rs]))

for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)
    for algo in ['mle', '2nn', 'gride', 'geomle']:
        if args.algo is not None:
            algo = args.algo

        print(f'computing ID for {key} dataset: {N} data, {X.shape[1]} features, true ID = {kwargs["d"]}')

        if algo =='mle':
            print('mle')
            mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
            np.save(f'{results}/mle/mle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([mle_ids, mle_err, mle_rs]))

        elif algo=='twonn':
            print('twonn')
            ie = IdEstimation(coordinates=X)
            twonn_ids, twonn_err, twonn_rs = ie.return_id_scaling_2NN(N_min = 16)
            np.save(f'{results}/twonn/twonn_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([twonn_ids, twonn_err, twonn_rs]))

        elif algo=='gride':
            print('gride')
            ie = IdEstimation(coordinates=X)
            gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 2048)
            np.save(f'{results}/gride/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))

        elif algo == 'geomle':
            print('geomle')
            #lets do explicit decimation here
            for fraction in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                nsubsample = int(nsample//fraction)
                print(fraction)
                if nsample > 2*args.k2:
                    for rep in range(int(fraction)):
                        X = X_full[np.random.choice(nsample, size = nsubsample, replace = False)]
                        id = geomle(X, k1 = 5, k2 = 15, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap , ver='GeoMLE')
                        with open(f'{args.results_folder}/geomle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}_k{args.k1}_{args.k2}_nrep{nrep}_nboots{nbootstrap}.txt', 'a') as f:
                            f.write(f'{X.shape[0]}  {np.mean(id): .3f} {np.std(id): .1f}\n')

    if args.algo is not None:
        break


#*******************************************************************************
if args.uniform_gride:
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
