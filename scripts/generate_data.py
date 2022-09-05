from utils.syntetic_datasets import *
from scipy.io import savemat
import numpy as np
import scipy

#used to generate cifar datasets
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--N', default = 16000, type = int)
parser.add_argument('--eps', default = 0.0, type = float)

parser.add_argument('--syntetic', action = 'store_true')
parser.add_argument('--real', action = 'store_true')

parser.add_argument('--csv', action = 'store_true')
parser.add_argument('--mat', action = 'store_true')
parser.add_argument('--npy', action = 'store_true')


parser.add_argument('--cifar', action = 'store_true')
parser.add_argument('--data_folder', default='../datasets', type=str)
args = parser.parse_args()

if args.syntetic:
    N = args.N
    eps = args.eps
    folder = args.data_folder+'/syntetic'
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

    mdict = {}
    for i, (key, value) in enumerate(names.items()):
        func = value[0]
        kwargs = value[1]
        X = func(**kwargs)

        if args.npy:
            path = f'{folder}/npy'
            if not os.path.isdir(f'{path}'):
                os.mkdir(f'{path}')
            np.save(f'{path}/{key}_{int(N/1000)}k_eps{eps}.npy', X)

        #tests on ESS
        if args.csv:
            path = f'{folder}/csv'
            print(path)
            np.savetxt(f'{path}/{key}_{int(N/1000)}k_eps{eps}.csv', X, delimiter=",")
        mdict[key] =  X

    if args.mat:
        #tests on DANCo matlab
        path = f'{folder}/mat'
        if not os.path.isdir(f'{path}'):
            os.mkdir(f'{path}')
        savemat(f"{path}/datasets_{int(N/1000)}k_eps{eps}.mat", mdict)

if args.real:
    folder = args.data_folder+'/real'
    #real datasets these verision of cifar occupy overall a lot of memory almost 1GB:
    #better to generate them on the fly during the analysis (gride twonn mle geomle)
    def build_dataset(pic, targets, category, sizes, name, folder):

        pic = pic[targets==category]
        for size in sizes:
            "transform to tensor to do the operations on higher precision crucial!"
            X = torch.from_numpy(pic.transpose((0, 3, 1, 2))).contiguous()
            X = X.to(dtype=torch.get_default_dtype()).div(255)
            X = transforms.functional.resize(X, size, interpolation = InterpolationMode.BICUBIC, antialias= True)
            "transform back to byte tensor"
            X = X.mul(255).byte()
            print(f'shape = {X.shape}')
            X_np = X.numpy().reshape(-1, np.prod(X[0].shape))
            print(f'reshape = {X_np.shape}')
            mdict = {'images': X_np}

            scipy.io.savemat(f'{folder}/{name}_{size}x{size}.mat', mdict)
            np.save(f'{folder}/{name}_{size}x{size}.npy', X_np)

    if args.cifar:
        path = folder+'/cifar'
        if not os.path.isdir(f'{path}'):
            os.mkdir(f'{path}')

        "test P scaling (build cifar dataset)"
        CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
        sizes = [int(4*(2**0.5)**i) for i in range(12)]

        build_dataset(
                pic = CIFAR_train.data,
                targets = np.array(CIFAR_train.targets),
                category =3,
                sizes = sizes,
                name='cifar_cat',
                folder = path
        )

        "N scaling save the full cifar dataset at 32x32 size"
        CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
        full_cifar = CIFAR_train.data.transpose(0, 3, 1, 2).reshape(50000, -1)
        np.save(f'{path}/cifar_training.npy', full_cifar)

    else:
        mat = scipy.io.loadmat(f'{folder}/face_data.mat')
        isomap_faces = mat['images'].T
        np.save(f'{folder}/isomap.npy', isomap_faces)

        isolet_train = np.genfromtxt(f'{folder}/isolet1+2+3+4.data', delimiter = ',')[:, :-1]
        isolet_val = np.genfromtxt(f'{folder}/isolet5.data', delimiter = ',')[:, :-1]
        isolet = np.concatenate((isolet_train, isolet_val), axis = 0)
        np.save(f'{folder}/isolet.npy', isolet)

        "mnist dataset normalized from 0 to 1 only 1s are used as benchmark"
        mnist_mat = scipy.io.loadmat(f'{folder}/mnist.mat')
        MNIST = mnist_mat['training'][0, 0][3].transpose(2, 0,1)
        MNIST = MNIST.reshape(-1, 784)
        tg = mnist_mat['training'][0, 0][4].flatten()
        MNIST = MNIST[tg==1]
        np.save(f'{folder}/mnist_ones.npy', MNIST)
