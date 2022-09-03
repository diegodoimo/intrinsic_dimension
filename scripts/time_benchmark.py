import numpy as np
#used to generate cifar datasets
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import argparse
import os
import time
import torch
import sys

from utils.estimators import return_id_scaling_gride, return_id_scaling_mle
from dadapy import IdEstimation
from utils.geomle import geomle, geomle_opt


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default = 'mnist', type = str)
parser.add_argument('--cifar_folder', default = '/home/diego/Documents/dottorato/ricerca/datasets/cifar10', type = str)
parser.add_argument('--algo', default = None, type = str)
parser.add_argument('--nrep', default = 1, type = int)
parser.add_argument('--nbootstrap', default = 20, type = int)
parser.add_argument('--data_folder', default='../datasets/real', type=str)
parser.add_argument('--filename', default='', type=str)
parser.add_argument('--k1', default=5, type=int)
parser.add_argument('--k2', default=15, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--results_folder', default='./results/real_datasets/time_benchmark', type=str)

args = parser.parse_args()

#*******************************************************************************
#benchmak P
def build_dataset(images, targets=None, category=None, size=None, transform = False):
    if category is not None:
        images = images[targets==category]
    X = torch.from_numpy(images.transpose((0, 3, 1, 2))).contiguous()
    X = X.to(dtype=torch.get_default_dtype()).div(255)
    if transform:
        X = transforms.functional.resize(X, size, interpolation = InterpolationMode.BICUBIC, antialias= True)
    X_np = X.numpy().reshape(-1, np.prod(X[0].shape))
    return X_np

if not os.path.isdir(f'{args.results_folder}'):
    os.makedirs(f'{args.results_folder}')


filename = f'{args.filename}'

CIFAR_train = datasets.CIFAR10(root=args.cifar_folder, train=True, download=True, transform=None)
X_full = build_dataset(images =CIFAR_train.data)
for algo in ['gride', 'twonn', 'mle', 'geomle']:
    if args.algo is not None:
        algo = args.algo

    print('benchmark N:', algo)
    print(f"n = {ndata}, type = {X_full.dtype}, shape = {X_full.shape}")

    ndata = X_full.shape[0]
    nscaling = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times = np.zeros((len(nscaling), 3))

    for i, p in enumerate(nscaling):

        nsample = int(ndata//p)
        print(f'nsample = {nsample}')
        sys.stdout.flush()
        X = X_full[np.random.choice(ndata, size = nsample, replace = False)]

        "gride"
        if algo == 'gride':

            ie = IdEstimation(coordinates=X)
            start = time.time()
            ids, stds, rs = ie.return_id_scaling_gride(range_max=min(100, int(ndata/10) ) )
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])

        "twoNN"
        if algo == 'twonn':
            ie = IdEstimation(coordinates=X)
            start = time.time()
            ids, stds, rs = ie.compute_id_2NN()
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])

        "mle"
        if algo == 'mle':
            k1 = 10
            start = time.time()
            ids, err, rs = return_id_scaling_mle(X, N_min = 16, k1 = k1, unbiased = False)
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])

        "geomle"
        if algo == 'geomle':
            filename = f'_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}'
            start = time.time()
            ids, rs = geomle_opt(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap)
            delay = time.time()-start

    np.save(f'{args.results_folder}/{args.algo}_cifarN{filename}.npy', times)
    if args.algo is not None:
        break

#*******************************************************************************
filename = f'{args.filename}'

for algo in ['gride', 'twonn', 'mle', 'geomle']:
    if args.algo is not None:
        algo = args.algo

    print('benchmark P:', algo)

    CIFAR_train = datasets.CIFAR10(root=args.cifar_folder, train=True, download=True, transform=None)
    features = [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]
    times = np.zeros((len(features), 3))

    for i, p in enumerate(features):
        print(p)
        X = build_dataset(
                images =CIFAR_train.data,
                targets = np.array(CIFAR_train.targets),
                category=3,
                size = p)

        "gride"
        if algo == 'gride':

            ie = IdEstimation(coordinates=X)
            start = time.time()
            ids, stds, rs = ie.return_id_scaling_gride(X)
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids[:3])])

        "twoNN"
        if algo == 'twonn':

            ie = IdEstimation(coordinates=X)
            start = time.time()
            ids, stds, rs = ie.compute_id_2NN()
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])

        "mle"
        if algo == 'mle':

            start = time.time()
            ids, err, rs = return_id_scaling_mle(X, N_min = 16, k1 = k1, unbiased = False)
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])


        "geomle"
        if algo == 'geomle':

            filename = f'_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}'
            start = time.time()
            ids, rs = geomle_opt(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap)
            delay = time.time()-start
            times[i] = np.array([p, delay, np.mean(ids)])

    np.save(f'{args.results_folder}/{args.algo}_cifarP{filename}.npy', times)
    if args.algo is not None:
        break
