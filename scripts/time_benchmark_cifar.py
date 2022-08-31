import numpy as np
#used to generate cifar datasets
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import argparse

from utils.estimators import return_id_scaling_gride
from dadapy import IdEstimation

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default = 'mnist', type = str)
parser.add_argument('--algo', default = '2nn', type = str)
parser.add_argument('--nrep', default = 10, type = int)
parser.add_argument('--nbootstrap', default = 20, type = int)
parser.add_argument('--ver', default = 'GeoMLE', choices = ['GeoMLE', 'fastGeoMLE'], type = str)
parser.add_argument('--n_sample', default=-1, type=int)
parser.add_argument('--n_components', default=-1, type=int)
parser.add_argument('--data_folder', default='/home/diego/ricerca/datasets', type=str)
parser.add_argument('--filename', default='', type=str)
parser.add_argument('--k1', default=10, type=int)
parser.add_argument('--k2', default=40, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--evaluate_ids', action="store_true")
parser.add_argument('--benchmark_n', action="store_true")
parser.add_argument('--benchmark_p', action="store_true")
parser.add_argument('--results_folder', default='./results', type=str)
args = parser.parse_args([])

#*******************************************************************************
#time benchmark as function of N:
with open(f'{args.results_folder}/gride_cifarN.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/2nn_simple_cifarN.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/mle_cifarN.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/geomle_cifarN_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
X_full = CIFAR_train.data.transpose(0, 3, 1, 2).reshape(50000, -1)
for fraction in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
    nsample = int(ndata//fraction)
    print(f'nsample = {nsample}')

    X = X_full[np.random.choice(ndata, size = nsample, replace = False)]

    "gride"
    start = time.time()
    ids, stds, rs = return_id_scaling_gride(X, range_max = min(100, X.shape[0]//10))
    delay = time.time()-start
    with open(f'{args.results_folder}/gride_cifarN.txt', 'a') as f:
        f.write(f'{X.shape[0]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')

    "twoNN"
    start = time.time()
    #ids, stds, rs = return_id_scaling_2NN(X, N_min = 10)
    ids, stds, rs = compute_id_2NN(X, X.shape[0])
    delay = time.time()-start
    with open(f'{args.results_folder}/2nn_simple_cifarN.txt', 'a') as f:
        f.write(f'{X.shape[0]} {ids: .5f} {delay: .5f}\n')

    "mle"
    start = time.time()
    mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
    delay = time.time()-start
    with open(f'{args.results_folder}/mle_simple_cifarN.txt', 'a') as f:
        f.write(f'{X.shape[0]} {ids: .5f} {delay: .5f}\n')

    "geomle"
    start = time.time()
    id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap, ver = args.ver)
    delay = time.time()-start
    with open(f'{args.results_folder}/geomle_cifarN_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'a') as f:
        f.write(f'{X.shape[0]}  {np.mean(id): .3f} {np.std(id): .1f}   {delay}\n')


#*******************************************************************************
with open(f'{args.results_folder}/gride_cifarP.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/2nn_simple_cifarP.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/mle_cifarP.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')

with open(f'{args.results_folder}/geomle_cifarP_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'w') as f:
    f.write(f'{"N":6} {"id":12} {"time":12}\n')


#benchmak P
def build_dataset(images, targets, category, size):
    images = images[targets==category]
    X = torch.from_numpy(images.transpose((0, 3, 1, 2))).contiguous()
    X = X.to(dtype=torch.get_default_dtype()).div(255)
    X = transforms.functional.resize(X, size, interpolation = InterpolationMode.BICUBIC, antialias= True)
    "transform back to byte tensor"
    X = X.mul(255).byte()
    print(f'shape = {X.shape}')
    X_np = X.numpy().reshape(-1, np.prod(X[0].shape))
    return X_np

CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
for p in [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]:

    X = build_dataset(
            images =CIFAR_train.data,
            targets = np.array(CIFAR_train.targets),
            category=3,
            size = p)

    "gride"
    start = time.time()
    ids, stds, rs = return_id_scaling_gride(X)
    delay = time.time()-start
    with open(f'{args.results_folder}/gride_cifarP.txt', 'a') as f:
        f.write(f'{X.shape[1]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')

    "twoNN"
    start = time.time()
    ids, stds, rs = compute_id_2NN(X, X.shape[0])
    delay = time.time()-start
    with open(f'{args.results_folder}/2nn_simple_cifarP.txt', 'a') as f:
        f.write(f'{X.shape[1]} {ids: .5f} {delay: .5f}\n')

    "mle"
    start = time.time()
    mle_ids, mle_err, mle_rs = return_id_scaling_gride(X, range_max = 2048, mg_estimator = True)
    delay = time.time()-start
    with open(f'{args.results_folder}/mle_simple_cifarP.txt', 'a') as f:
        f.write(f'{X.shape[0]} {ids: .5f} {delay: .5f}\n')

    "geomle"
    start = time.time()
    id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap, ver = args.ver)
    delay = time.time()-start
    with open(f'{args.results_folder}/geomle_cifarP_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'a') as f:
            f.write(f'{X.shape[1]}  {np.mean(id): .3f} {np.std(id): .1f}   {delay}\n')
