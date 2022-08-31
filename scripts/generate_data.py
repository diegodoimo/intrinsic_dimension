from utils.syntetic_datasets import *
from scipy.io import savemat
import numpy as np


#used to generate cifar datasets
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


#syntetic datasets
data_folder = './datasets/syntetic'
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

mdict = {}
for i, (key, value) in enumerate(names.items()):
    func = value[0]
    kwargs = value[1]
    X = func(**kwargs)

    np.save(f'{data_folder}/npy/{key}_{int(N/1000)}k_eps{eps}.npy', X)
    #tests on ESS
    np.savetxt(f'{data_folder}/csv/{key}_{int(N/1000)}k_eps{eps}.csv', X, delimiter=",")
    mdict[key] =  X
#tests on DANCo matlab
savemat(f"{data_folder}/matlab_datasets.mat", mdict)




#real datasets these verision of cifar occupy overall a lot of memory almost 1GB:
#better to generate them on the fly during the analysis (gride twonn mle geomle)
def build_dataset(pic, targets, category, sizes, name):
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
        scipy.io.savemat(f'datasets/{name}_{size}x{size}.mat', mdict)
        np.save(f'datasets/{name}_{size}x{size}.npy', X_np)

"test P scaling (build cifar dataset)"
CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
sizes = [int(4*(2**0.5)**i) for i in range(12)]
build_dataset(
        pic = CIFAR_train.data,
        targets = np.array(CIFAR_train.targets),
        category =3,
        sizes = sizes,
        name='cifar_cat',
)


"N scaling save the full cifar dataset at 32x32 size"
CIFAR_train = datasets.CIFAR10(root='/home/diego/ricerca/datasets/cifar10', train=True, download=False, transform=None)
full_cifar = CIFAR_train.data.transpose(0, 3, 1, 2).reshape(50000, -1)
np.save('datasets/cifar_training.npy', full_cifar)
