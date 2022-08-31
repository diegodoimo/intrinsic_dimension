#import os
#os.environ["OMP_NUM_THREADS"] = "12"
import scipy.io
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from id_estimation import return_id_scaling_gride, return_id_scaling_2NN, compute_id_2NN
#from skdim.id import DANCo
#from skdim.id import ESS
from sklearn.decomposition import PCA
from geomle import geomle
import sys
import argparse
import time
import torch


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
parser.add_argument('--ver_ess', default = 'a')
parser.add_argument('--results_folder', default='./results', type=str)
args = parser.parse_args()


# def build_dataset(pic, targets, category, sizes, name):
#     pic = pic[targets==category]
#     for size in sizes:
#         "transform to tensor to do the operations on higher precision crucial!"
#         X = torch.from_numpy(pic.transpose((0, 3, 1, 2))).contiguous()
#         X = X.to(dtype=torch.get_default_dtype()).div(255)
#         X = transforms.functional.resize(X, size, interpolation = InterpolationMode.BICUBIC, antialias= True)
#         "transform back to byte tensor"
#         X = X.mul(255).byte()
#         print(f'shape = {X.shape}')
#         X_np = X.numpy().reshape(-1, np.prod(X[0].shape))
#         print(f'reshape = {X_np.shape}')
#         mdict = {'images': X_np}
#         scipy.io.savemat(f'datasets/{name}_{size}x{size}.mat', mdict)
#         np.save(f'datasets/{name}_{size}x{size}.npy', X_np)
#
#
#"test N, p scaling (build cifar dataset)"
#MNIST_train = datasets.MNIST(root='/home/diego/ricerca/datasets/mnist', train=True, download=False, transform=None)
#CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
#full_cifar = CIFAR_train.data.transpose(0, 3, 1, 2).reshape(50000, -1)
#np.save('datasets/cifar_training.npy', full_cifar)
#scipy.io.savemat('datasets/cifar_training.mat', {'images': full_cifar})
#
#     filename+='.txt'
#     return filename


"test N, p scaling (build cifar dataset)"
#MNIST_train = datasets.MNIST(root='/home/diego/ricerca/datasets/mnist', train=True, download=False, transform=None)
# CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
# full_cifar = CIFAR_train.data.transpose(0, 3, 1, 2).reshape(50000, -1)
# np.save('datasets/cifar_training.npy', full_cifar)
# scipy.io.savemat('datasets/cifar_training.mat', {'images': full_cifar})
#
# sizes = [int(4*(2**0.5)**i) for i in range(12)]
# build_dataset(
#         pic = CIFAR_train.data,
#         targets = np.array(CIFAR_train.targets),
#         category =3,
#         sizes = sizes,
#         name='cifar_cat',
# )

#*******************************************************************************
rng = np.random.default_rng(2022)
rng.random(args.seed)


print('loading data...')
def load_isomap(save = False):
    "isomap faces"
    mat = scipy.io.loadmat('./datasets/face_data.mat')
    isomap_faces = mat['images'].T
    if save: np.save('./datasets/isomap_faces.npy', isomap_faces)
    return isomap_faces
#np.save('isomap_faces.npy', isomap_faces)

def load_isolet(save =False):
    "isolet dataset"
    #last entry is the class
    isolet_train = np.genfromtxt('./datasets/isolet1+2+3+4.data', delimiter = ',')[:, :-1]
    isolet_val = np.genfromtxt('./datasets/isolet5.data', delimiter = ',')[:, :-1]
    isolet = np.concatenate((isolet_train, isolet_val), axis = 0)
    if save: np.save('./datasets/isolet.npy', isolet)
    return isolet
#

def load_mnist(save = False):
    "mnist dataset (digit in geo are preprocessed MNIST)"
    MNIST_train = datasets.MNIST(root=f'{args.data_folder}/mnist', train=True, download=False, transform=None)
    MNIST = MNIST_train.data.numpy().reshape(-1, 784)
    MNIST = MNIST[MNIST_train.targets==1]
    if save: np.save('./datasets/mnist.npy', MNIST)
    # MNIST = MNIST.astype('float')
    "mnist dataset normalized from 0 to 1 only 1s are used as benchmark"
    mnist_mat = scipy.io.loadmat(f'./datasets/mnist.mat')
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


# load_mnist(save = True)
#
#
# isolet_train = np.genfromtxt('./datasets/isolet1+2+3+4.data', delimiter = ',', dtype = np.single)[:, :-1]
#
# isolet_train = np.genfromtxt('./datasets/isolet1+2+3+4.data', delimiter = ',', dtype = np.half)[:, :-1]
# isolet_val = np.genfromtxt('./datasets/isolet5.data', delimiter = ',', dtype = np.half)[:, :-1]
# isolet = np.concatenate((isolet_train, isolet_val), axis = 0)
# np.save('./datasets/isolet_float16.npy', isolet)
#
#
# ids, stds, rs = return_id_scaling_gride(isolet, range_max = 512)
#
# ids


if args.data_name!='cifar':
    #for args.data_name, val in data.items():
    X_full = data[args.data_name]
    ndata = X_full.shape[0]
    print(f'{args.data_name}')
    if args.algo == 'gride':
        with open(f'{args.results_folder}/gride_{args.data_name}.txt', 'w') as f:
            f.write(f'{"N":6} {"id":12} {"time":12}\n')
        nsample = X_full.shape[0]
        "gride"
        X = X_full
        print(f'gride ndata= {nsample}')
        start = time.time()
        ids, stds, rs = return_id_scaling_gride(X, range_max = max(512, X.shape[0]//4))
        delay = time.time()-start
        with open(f'{args.results_folder}/gride_{args.data_name}.txt', 'a') as f:
                f.write(f'{X.shape[0]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')
        np.save(f'{args.results_folder}/gride_{args.data_name}_{nsample}.npy', np.array([ids, stds, rs]) )

    elif args.algo=='2nn':
        with open(f'{args.results_folder}/2nn_{args.data_name}.txt', 'w') as f:
            f.write(f'{"N":6} {"id":12} {"time":12}\n')
        nsample = X_full.shape[0]
        X = X_full
        print(f'2nn ndata= {nsample}')
        start = time.time()
        ids, stds, rs = return_id_scaling_2NN(X, N_min = 10)
        delay = time.time()-start
        with open(f'{args.results_folder}/2nn_{args.data_name}.txt', 'a') as f:
                f.write(f'{X.shape[0]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')
        np.save(f'{args.results_folder}/2nn_{args.data_name}_{nsample}.npy', np.array([ids, stds, rs]) )

    elif args.algo == 'geomle':
        with open(f'{args.results_folder}/geomle_{args.data_name}_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'w') as f:
            f.write(f'{"N":6} {"id":12} {"time":12}\n')

        for fraction in [1, 2, 4, 8, 16, 32]:
            nsample = int(ndata//fraction)
            if nsample > 2*args.k2:
                for rep in range(int(4*fraction)):
                    X = X_full[np.random.choice(ndata, size = nsample, replace = False)]
                    start = time.time()
                    id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap , ver = args.ver)
                    delay = time.time()-start

                    with open(f'{args.results_folder}/geomle_{args.data_name}_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'a') as f:
                        f.write(f'{X.shape[0]}  {np.mean(id): .3f} {np.std(id): .1f}   {delay}\n')
                    print(f'geomle finished in {delay}sec')
                    sys.stdout.flush()


elif args.data_name == 'cifar':
    if args.benchmark_n:

        X_full = np.load(f'./datasets/cifar_training.npy')
        ndata = X_full.shape[0]

        if args.algo == 'gride':
            with open(f'{args.results_folder}/gride_cifarN.txt', 'w') as f:
                f.write(f'{"N":6} {"id":12} {"time":12}\n')

            for fraction in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                nsample = int(ndata//fraction)
                print(f'nsample = {nsample}')
                "gride"
                X = X_full[np.random.choice(ndata, size = nsample, replace = False)]
                start = time.time()
                ids, stds, rs = return_id_scaling_gride(X, range_max = min(512, X.shape[0]//4))
                delay = time.time()-start
                with open(f'{args.results_folder}/gride_cifarN.txt', 'a') as f:
                    f.write(f'{X.shape[0]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')
                np.save(f'{args.results_folder}/gride_cifarN_{nsample}.npy', np.array([ids, stds, rs]) )

        if args.algo == '2nn':
            with open(f'{args.results_folder}/2nn_simple_cifarN.txt', 'w') as f:
                f.write(f'{"N":6} {"id":12} {"time":12}\n')

            for fraction in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                nsample = int(ndata//fraction)
                print(f'nsample = {nsample}')
                "2NN"
                X = X_full[np.random.choice(ndata, size = nsample, replace = False)]
                start = time.time()
                #ids, stds, rs = return_id_scaling_2NN(X, N_min = 10)
                ids, stds, rs = compute_id_2NN(X, X.shape[0])
                delay = time.time()-start
                with open(f'{args.results_folder}/2nn_simple_cifarN.txt', 'a') as f:
                    f.write(f'{X.shape[0]} {ids: .5f} {delay: .5f}\n')
                np.save(f'{args.results_folder}/2nn_simple_cifarN_{nsample}.npy', np.array([ids, stds, rs]) )

        elif args.algo == 'geomle':
            with open(f'{args.results_folder}/geomle_cifarN_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'w') as f:
                f.write(f'{"N":6} {"id":12} {"time":12}\n')

            for fraction in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                nsample = int(ndata//fraction)
                print(f'nsample = {nsample}')
                X = X_full[np.random.choice(ndata, size = nsample, replace = False)]
                start = time.time()
                id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap, ver = args.ver)
                delay = time.time()-start
                with open(f'{args.results_folder}/geomle_cifarN_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'a') as f:
                    f.write(f'{X.shape[0]}  {np.mean(id): .3f} {np.std(id): .1f}   {delay}\n')

    if args.benchmark_p:
        if args.algo == 'gride':

            with open(f'{args.results_folder}/gride_cifarP.txt', 'w') as f:
                f.write(f'{"P":6} {"id":12} {"time":12}\n')

            for p in [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]:
                print(f'p={p}')
                X = np.load(f'./datasets/cifar_cat_{p}x{p}.npy')
                start = time.time()
                ids, stds, rs = return_id_scaling_gride(X, range_max = min(512, X.shape[0]//4))
                delay = time.time()-start
                with open(f'{args.results_folder}/gride_cifarP.txt', 'a') as f:
                    f.write(f'{X.shape[1]} {np.mean(ids[:3]): .5f} {delay: .5f}\n')
                np.save(f'{args.results_folder}/gride_cifarP_{p}.npy', np.array([ids, stds, rs]) )

        elif args.algo == '2nn':
            with open(f'{args.results_folder}/2nn_simple_cifarP.txt', 'w') as f:
                f.write(f'{"P":6} {"id":12} {"time":12}\n')

            for p in [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]:
                print(f'p={p}')
                X = np.load(f'./datasets/cifar_cat_{p}x{p}.npy')
                start = time.time()
                #ids, stds, rs = return_id_scaling_2NN(X, N_min = 10)
                ids, stds, rs = compute_id_2NN(X, X.shape[0])
                delay = time.time()-start
                with open(f'{args.results_folder}/2nn_simple_cifarP.txt', 'a') as f:
                    f.write(f'{X.shape[1]} {ids: .5f} {delay: .5f}\n')
                np.save(f'{args.results_folder}/2nn_simple_cifarP_{p}.npy', np.array([ids, stds, rs]) )

        elif args.algo == 'geomle':
            with open(f'{args.results_folder}/geomle_cifarP_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'w') as f:
                f.write(f'{"P":6} {"id":12} {"time":12}\n')
            for p in [4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181]:
                print(f'p={p}')
                X = np.load(f'./datasets/cifar_cat_{p}x{p}.npy')

                start = time.time()
                id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = args.nrep, nb_iter2 = args.nbootstrap, ver = args.ver)
                delay = time.time()-start
                with open(f'{args.results_folder}/geomle_cifarP_k{args.k1}_{args.k2}_nrep{args.nrep}_nboots{args.nbootstrap}.txt', 'a') as f:
                        f.write(f'{X.shape[1]}  {np.mean(id): .3f} {np.std(id): .1f}   {delay}\n')



# def get_data(args):
#     if args.n_components >0:
#         assert args.n_components < X.shape[1]
#         pca = PCA(n_components=args.n_components)
#         X = pca.fit_transform(data[args.data_name])
#     else:
#         X = data[args.data_name]
#
#     if args.n_sample >0:
#         assert args.n_sample < X.shape[0]
#         indx = np.random.choice(X.shape[0], size=args.n_sample, replace=False)
#         X = X[indx]
#     return X


# def compute_ids(X, args, filename):
#     print('id computation started')
#     "Gride"
#     start = time.time()
#     ids, stds, rs = return_id_scaling_gride(X, range_max = max(512, X.shape[0]//4))
#     delay = time.time()-start
#
#     with open(f'{args.results_folder}/{filename}', 'w') as f:
#         f.write(f'gride   {np.mean(ids[:5])}   {delay}\n')
#     filename_ = filename.split('.')[0]
#     np.save(f'{args.results_folder}/{filename_}_gride', np.array([ids, stds, rs]) )
#     print(f'gride finished in {delay}sec')
#     sys.stdout.flush()
#
#     "2NN"
#     start = time.time()
#     ids, stds, rs = return_id_scaling_2NN(X, N_min = 10)
#     delay = time.time()-start
#
#     with open(f'{args.results_folder}/{filename}', 'a') as f:
#         f.write(f'2NN   {np.mean(ids[:5])}   {delay}\n')
#     np.save(f'{args.results_folder}/{filename_}_2NN', np.array([ids, stds, rs]) )
#     print(f'twoNN finished in {delay}sec')
#     sys.stdout.flush()

    # start = time.time()
    # id = geomle(X, k1 =args.k1, k2 = args.k2, nb_iter1 = 10)
    # delay = time.time()-start
    #
    # with open(f'{args.results_folder}/{filename}', 'a') as f:
    #     f.write(f'geomle   {np.mean(id)}({np.std(id)})   {delay}\n')
    # print(f'geomle finished in {delay}sec')
    # sys.stdout.flush()


    # "DANCo"
    # danco = DANCo()
    # start = time.time()
    # id = danco.fit(X)
    # delay = time.time()-start
    #
    # with open(f'{args.results_folder}/{filename}', 'a') as f:
    #     f.write(f'danco   {id.dimension_}   {delay}\n')
    # print(f'DANCo finished in {delay/60}min')
    # sys.stdout.flush()
    #
    #
    # "ESS"
    # ess = ESS()
    # start = time.time()
    # id = ess.fit(X)
    # delay = time.time()-start
    #
    # with open(f'{args.results_folder}/{filename}', 'a') as f:
    #     f.write(f'ess   {id.dimension_}   {delay}\n')
    # print(f'ESS finished in {delay/60}min')
    # sys.stdout.flush()











# if args.evaluate_ids:
#     print('evaluate ids')
#     X = get_data(args)
#     filename = get_filename(args)
#     compute_ids(X, args, filename)
#
# if args.benchmark_features:
#     args.n_sample = 1000
#     args.reduce_sample = True
#     for ncomponents in [20, 50, 100, 200]:
#         print(f'benchmark features: {ncomponents}')
#         args.n_components = ncomponents
#         args.filename='bm_features'
#
#
#         X = get_data(args)
#         filename  = get_filename(filename)
#         compute_ids(X, args, filename)
#
# if args.benchmark_sample:
#     args.n_components = 100
#     for nsample in [200, 500, 1000, 2000]:
#         print(f'benchmark sample: {nsample}')
#         args.n_sample = nsample
#         args.reduce_sample = True
#         args.filename='bm_sample'
#         filename = get_filename(args)
#
#         X = get_data(args)
#         compute_ids(X, args, filename)
