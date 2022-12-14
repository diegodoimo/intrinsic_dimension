from utils.estimators import return_id_scaling_gride, return_id_scaling_mle
from utils.syntetic_datasets import *
from dadapy import IdEstimation
from sklearn.neighbors import NearestNeighbors
from utils.geomle import geomle, geomle_opt
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default = 'mnist', type = str)
parser.add_argument('--algo', default = None, type = str)
parser.add_argument('--N', default = 16000, type = int)
parser.add_argument('--eps', default = 0.01, type = float)
parser.add_argument('--nrep', default = 10, type = int)
parser.add_argument('--nbootstrap', default = 20, type = int)
parser.add_argument('--data_folder', default='/home/diego/ricerca/datasets', type=str)
parser.add_argument('--k1', default=5, type=int)
parser.add_argument('--k2', default=15, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--results_folder', default='./results/syntetic_datasets', type=str)
parser.add_argument('--uniform_gride', action = 'store_true')

args = parser.parse_args([])


args.algo = 'geomle'
args.k1 = 0
args.k2 = 10
args.results_folder = './tests'
#*******************************************************************************
rng = np.random.default_rng(2022)
rng.random(args.seed)

if not args.uniform_gride:
    N = args.N
    eps = args.eps

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

    for algo in ['mle', 'twonn', 'gride', 'geomle']:

        if args.algo is not None:
            algo = args.algo  #compue the id on all the datasets with a single selected algorithm (see end for loop)

        for i, (key, value) in enumerate(names.items()):

            func = value[0]
            kwargs = value[1]
            X = func(**kwargs)
            nsample = X.shape[0]

            print(f'computing ID for {key} dataset:\n{N} data\n{X.shape[1]} features\neps={eps}\ntrue ID = {kwargs["d"]}')
            if algo =='mle':
                print('mle')
                #mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
                k1 = 10
                mle_ids, mle_err, mle_rs = return_id_scaling_mle(X, N_min = 16, k1 = k1, unbiased = False)
                path = f'{args.results_folder}/mle'
                if not os.path.isdir(f'{path}'):
                    os.mkdir(f'{path}')
                np.save(f'{path}/mle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([mle_ids, mle_err, mle_rs]))

            elif algo=='twonn':
                print('twonn')
                ie = IdEstimation(coordinates=X)
                twonn_ids, twonn_err, twonn_rs = ie.return_id_scaling_2NN(N_min = 16)
                path = f'{args.results_folder}/twonn'
                if not os.path.isdir(f'{path}'):
                    os.mkdir(f'{path}')
                np.save(f'{path}/twonn_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([twonn_ids, twonn_err, twonn_rs]))

            elif algo=='gride':
                print('gride')
                ie = IdEstimation(coordinates=X)
                gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 2048)
                path = f'{args.results_folder}/gride'
                if not os.path.isdir(f'{path}'):
                    os.mkdir(f'{path}')
                np.save(f'{path}/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))

            elif algo == 'geomle':
                nsample= X.shape[0]
                print('geomle')
                #lets do explicit decimation here
                geomle_ids, geomle_err, geomle_rs  = [], [], []
                for i, fraction in enumerate([1, 2, 4, 8, 16,32, 64, 128, 256, 512]):
                    nsubsample = int(nsample//fraction)
                    print(fraction)
                    sys.stdout.flush()
                    if nsubsample > 2*args.k2:
                        nrep = fraction
                        X_bootstrap = X[np.random.choice(nsample, size = nsubsample, replace = False)]
                        ids, rs = geomle_opt(X_bootstrap, k1 = args.k1, k2 = args.k2, nb_iter1 = nrep, nb_iter2 = args.nbootstrap)

                    geomle_ids.append(np.mean(ids))
                    geomle_err.append( np.std(ids)/len(ids) )
                    geomle_rs.append( np.mean(rs) )

                path = f'{args.results_folder}/geomle'
                if not os.path.isdir(f'{path}'):
                    os.mkdir(f'{path}')
                np.save(f'{path}/geomle_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}_k{args.k1}_{args.k2}.npy', np.array([geomle_ids, geomle_err, geomle_rs]))

        if args.algo is not None:
            break



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
        ie = IdEstimation(coordinates=X)
        gride_ids, gride_err, gride_rs = ie.return_id_scaling_gride(range_max = 4096)
        print(gride_ids)

        path = f'{args.results_folder}/gride'
        if not os.path.isdir(f'{path}'):
            os.mkdir(f'{path}')
        np.save(f'{path}/gride_{key}_N{N/1000}k_D{kwargs["D"]}_d{kwargs["d"]}_eps{kwargs["eps"]}.npy', np.array([gride_ids, gride_err, gride_rs]))
