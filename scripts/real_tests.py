import numpy as np
import scipy.io
import argparse
from utils.estimators import return_id_scaling_gride, return_id_scaling_mle
from dadapy import IdEstimation
import torchvision.datasets as datasets
from utils.geomle import geomle_opt

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default = 'mnist', type = str)
parser.add_argument('--algo', default = None, type = str)
parser.add_argument('--nrep', default = 10, type = int)
parser.add_argument('--nbootstrap', default = 20, type = int)
parser.add_argument('--data_folder', default='../datasets/real', type=str)
parser.add_argument('--filename', default='', type=str)
parser.add_argument('--k1', default=20, type=int)
parser.add_argument('--k2', default=55, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--results_folder', default='./results/real_datasets', type=str)
args = parser.parse_args([])

rng = np.random.default_rng(2022)
rng.random(args.seed)
args.algo = 'geomle'


print('loading data...')
def load_isomap(save = False):
    "isomap faces"
    mat = scipy.io.loadmat(f'{args.data_folder}/face_data.mat')
    isomap_faces = mat['images'].T
    return isomap_faces

def load_isolet(save =False):
    "isolet dataset"
    #last entry is the class
    isolet_train = np.genfromtxt(f'{args.data_folder}/isolet1+2+3+4.data', delimiter = ',')[:, :-1]
    isolet_val = np.genfromtxt(f'{args.data_folder}/isolet5.data', delimiter = ',')[:, :-1]
    isolet = np.concatenate((isolet_train, isolet_val), axis = 0)
    return isolet

def load_mnist(save = False):
    "mnist dataset normalized from 0 to 1 only 1s are used as benchmark"
    mnist_mat = scipy.io.loadmat(f'{args.data_folder}/mnist.mat')
    MNIST = mnist_mat['training'][0, 0][3].transpose(2, 0,1)
    MNIST = MNIST.reshape(-1, 784)
    tg = mnist_mat['training'][0, 0][4].flatten()
    MNIST = MNIST[tg==1]
    return MNIST

data = {
    'isomap': load_isomap(),
    'isolet': load_isolet(),
    'mnist': load_mnist(),
}


for algo in ['gride', 'twonn', 'mle']:
    if args.algo is not None:
        algo = args.algo

    print(f'using algo {algo}')
    for key, X_full in data.items():  #values of the dictionary is a the dataset returned by the load function
        nsample = X_full.shape[0]

        if algo == 'gride':
            print(f'gride ndata= {nsample}')
            ie = IdEstimation(coordinates=X_full)
            ids, stds, rs = ie.return_id_scaling_gride(range_max = 128)
            np.save(f'{args.results_folder}/gride_{key}.npy', np.array([ids, stds, rs]) )

        elif algo=='twonn':
            print(f'2nn ndata= {nsample}')
            ie = IdEstimation(coordinates=X_full)
            ids, stds, rs = ie.return_id_scaling_2NN(N_min = 16)
            np.save(f'{args.results_folder}/twonn_{key}.npy', np.array([ids, stds, rs]) )

        elif algo=='mle':
            print(f'mle ndata= {nsample}')
            k1 = 10
            ids, stds, rs = return_id_scaling_mle(X_full, N_min = 16, k1 = 10, unbiased = False)
            np.save(f'{args.results_folder}/mle_{key}.npy', np.array([ids, stds, rs]) )

        elif algo == 'geomle':
            print(f'geomle ndata = {nsample}')
            #lets do explicit decimation here
            for fraction in [1, 2, 4, 8, 16, 32]:
                nsubsample = int(nsample//fraction)
                print(nsubsample)
                if nsample > 2*args.k2:
                    for rep in range(int(4*fraction)):
                        X = X_full[np.random.choice(nsample, size = nsubsample, replace = False)]
                        id = geomle_opt(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap , ver = 'GeoMLE')
                        with open(f'{args.results_folder}/geomle_{args.data_name}_k{args.k1}_{args.k2}_nrep{args.nrep}.txt', 'a') as f:
                            f.write(f'{X.shape[0]}  {np.mean(id): .3f} {np.std(id): .1f}\n')

    if args.algo is not None:
        break
