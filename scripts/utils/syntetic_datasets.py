import numpy as np
from sklearn.datasets import make_swiss_roll

"taken from "
def uniform(N, D=20, d=5, eps=0, seed=42):
    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    dataset[:, :d] += np.random.uniform(low=0.0, high= 1.0, size = (N, d))
    return dataset

def normal(N, D=3, d=2, eps=0, seed=42):
    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    dataset[:, :d] += np.random.normal(scale = 1, size = (N, d))
    return dataset

"id = D-1, d =10, D=11"
def sphere(N, D=15, d=10, eps=0, seed=42):
    assert D>d

    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    tmp = np.random.normal(loc = 0.0, scale = 1, size =(N, d+1))
    normalization = np.sqrt(np.sum(tmp**2, axis = 1, keepdims = True))
    tmp = tmp/normalization

    dataset[:, :d+1] += tmp

    return dataset

def spiral1d(N, D=3, d = 1, eps=0.1, seed =42):
    assert d==1

    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    t = np.random.uniform((4*np.pi)**-1, 2*np.pi, N) #eps = 0.1

    #t = np.random.uniform(0,1, N)
    dataset[:, :3] += np.array([t*np.cos(t), t*np.sin(t), t]).T
    return dataset

def swissroll(N, D=3, d=2, eps=0.3, seed =42):
    assert d == 2
    #eps = 0.3 N = 10k
    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(0, 1, N))
    y = 20* np.random.uniform(size=N)
    x = t * np.cos(t)
    z = t * np.sin(t)
    X = np.vstack((x, y, z))
    X = X.T

    #tmp, t_noisy = make_swiss_roll(N, noise=eps, random_state = 42)

    dataset[:, :3] += 0.05*X

    return dataset

def moebius(N, D=6, d=3, ntwists = 10, eps=0, seed =42):
    assert d==2
    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    phi = 2*np.random.uniform(0, 1, N)*np.pi
    rad = 2*np.random.uniform(0, 1, N)-1

    x = (1+0.5*rad*np.cos(0.5*ntwists*phi))*np.cos(phi)
    y = (1+0.5*rad*np.cos(0.5*ntwists*phi))*np.sin(phi)
    z = 0.5*rad*np.sin(0.5*ntwists*phi)

    dataset[:, :3] = np.vstack([x, y, z]).T
    return dataset


def paraboloid(N, D=30, d=9, eps = 0, seed = 42):
    assert D == 3 * (d + 1)
    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    E = np.random.exponential(1, (d+1,N))
    X = ((1 + E[1:]/E[0])**-1).T
    X = np.hstack([X, (X ** 2).sum(axis=1)[:,np.newaxis]])
    dataset += np.hstack([X, np.sin(X), X**2])

    assert dataset.shape == (N, D)

    return dataset


def nonlinear(N, D=30, d=9, eps = 0, seed = 42):
    assert D >= d
    m = int(D / (2 * d))
    assert D == 2 * m * d

    np.random.seed(seed)
    if eps>0:
        dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5, size =(N, D))
    else:
        dataset = np.zeros(shape =(N, D))

    p = np.random.uniform(0, 1, size=(d, N))
    F = np.zeros((2*d, N))

    F[0::2, :] = np.cos(2*np.pi*p)
    F[1::2, :] = np.sin(2*np.pi*p)

    R = np.zeros((2*d, N))
    R[0::2, :] = np.vstack([p[1:], p[0]])
    R[1::2, :] = np.vstack([p[1:], p[0]])
    D = (R * F).T
    dataset = np.hstack([D] * m)
    #print(dataset.shape)
    #assert a.any(dataset.shape[0] == N and dataset.shape[1] == D)
    return dataset

# def gen_nonlinear_data(n, dim, d, sampler):
#     assert dim >= d
#     m = int(dim / (2 * d))
#     assert dim == 2 * m * d
#
#     p = sampler(d, n)
#     F = np.zeros((2*d, n))
#
#     F[0::2, :] = np.cos(2*np.pi*p)
#     F[1::2, :] = np.sin(2*np.pi*p)
#
#     R = np.zeros((2*d, n))
#     R[0::2, :] = np.vstack([p[1:], p[0]])
#     R[1::2, :] = np.vstack([p[1:], p[0]])
#     D = (R * F).T
#     data = pd.DataFrame(np.hstack([D] * m))
#
#     assert data.shape == (n, dim)
#     return data






# "from https://link.springer.com/article/10.1007/s10994-012-5294-7"
# def curved_data(N, D=24, d=6, eps=0, seed =42):
#     D = 4*d
#     np.random.seed(seed)
#     if eps>0:
#         dataset = np.random.normal(loc = 0.0, scale = eps/D**0.5**0.5, size =(N, D))
#     else:
#         dataset = np.zeros( shape=(N, D))
#
#     X = np.random.uniform(0, 1, size = (N, d))
#
#     #X_prime = np.cos(np.sin(np.pi*X))
#     #X_second = np.sin(np.cos(np.pi*X))
#     X_prime = np.zeros_like(X)
#     X_second = np.zeros_like(X)
#     for i in range(d):
#         X_prime[:, i] = np.tan( X[:, i]**i*np.cos(X[:, i]**(d-i+1) ) )
#         X_second[:, i] = np.arctan( X[:, i]**(d-i+1)*np.sin(X[:, i]**i) )
#
#     X_cat = np.concatenate((X_prime, X_second), axis = 1)
#
#     #print(X_cat.shape)
#     X_cat = np.concatenate((X_cat, X_cat), axis = 1)
#     #print(X_cat.shape)
#     dataset+= X_cat
#
#     return dataset



# __all__ = ('DataGenerator')
#
# import pandas as pd
# import numpy as np
# from .utils import bound_nonuniform_sampler, uniform_sampler
#
#
# class DataGenerator():
#
#     def __init__(self,
#                  random_state: int = None,
#                  type_noise:   str = 'norm'):
#
#         self.set_rng(random_state)
#         self.set_gen_noise(type_noise)
#         self.dict_gen = {
#             #syntetic data
#             'Helix1d':        gen_helix1_data,
#             'Helix2d':        gen_helix2_data,
#             'Helicoid':       gen_helicoid_data,
#             'Spiral':         gen_spiral_data,
#             'Roll':           gen_roll_data,
#             'Scurve':         gen_scurve_data,
#             'Star':           gen_star_data,
#             'Moebius':        gen_moebius_data,
#
#             'Sphere':         gen_sphere_data,
#             'Norm':           gen_norm_data,
#             'Uniform':        gen_uniform_data,
#             'Cubic':          gen_cubic_data,
#
#             'Affine_3to5':    gen_affine3_5_data,
#             'Affine':         gen_affine_data,
#
#             'Nonlinear_4to6': gen_nonlinear4_6_data,
#             'Nonlinear':      gen_nonlinear_data,
#             'Paraboloid':     gen_porabaloid_data,
#         }
#
#
#     def set_rng(self, random_state:int=None):
#         if random_state is not None:
#             np.random.seed(random_state)
#
#     def set_gen_noise(self, type_noise:str):
#         if not hasattr(self, 'rng'):
#             self.set_rng()
#         if type_noise == 'norm':
#             self.gen_noise = np.random.randn
#         if type_noise == 'uniform':
#             self.gen_noise = lambda n, dim: np.random.rand(n, dim) - 0.5
#
#     def gen_data(self, name:str, n:int, dim:int, d:int, type_sample:str='uniform', noise:float=0.0):
#     # Parameters:
#     # --------------------
#     # name: string
#     #     Type of generetic data
#     # n: int
#     #     The number of sample points
#     # dim: int
#     #     The dimension of point
#     # d: int
#     #     The hyperplane dimension
#     # noise: float, optional(default=0.0)
#     #     The value of noise in data
#
#     # Returns:
#     # data: pd.Dataframe of shape (n, dim)
#     #     The points
#         assert name in self.dict_gen.keys(),\
#                'Name of data is unknown'
#         if (type_sample == 'uniform'):
#             if name == 'Sphere':
#                 sampler = np.random.randn
#             else:
#                 sampler = np.random.rand
#         elif (type_sample == 'nonuniform'):
#             if name == 'Sphere':
#                 sampler = uniform_sampler
#             else:
#                 sampler = bound_nonuniform_sampler
#         else:
#             assert False, 'Check type_sample'
#
#         data = self.dict_gen[name](n=n, dim=dim, d=d, sampler=sampler)
#         noise = self.gen_noise(n, dim) * noise
#
#         return  data + noise







# #surphace of a cube
# def gen_cubic_data(n, dim, d, sampler):
#     assert d < dim
#     cubic_data = np.array([[]]*(d + 1))
#     for i in range(d + 1):
#         n_once = int(n / (2 * (d + 1)) + 1)
#         #1st side
#         data_once = sampler(d + 1, n_once)
#         data_once[i] = 0
#         cubic_data = np.hstack([cubic_data, data_once])
#         #2nd side
#         data_once = sampler(d + 1, n_once)
#         data_once[i] = 1
#         cubic_data = np.hstack([cubic_data, data_once])
#     cubic_data = cubic_data.T[:n]
#     data = pd.DataFrame(np.hstack([cubic_data, np.zeros((n, dim - d - 1))]))
#     assert data.shape == (n, dim)
#     return data
#
#

#
# def gen_moebius_data(n, dim, d, sampler):
#     assert dim == 3
#     assert d == 2
#
#     phi = sampler(n) * 2 * np.pi
#     rad = sampler(n) * 2 - 1
#     data = pd.DataFrame(np.vstack([(1+0.5*rad*np.cos(5.0*phi))*np.cos(phi),
#                                    (1+0.5*rad*np.cos(5.0*phi))*np.sin(phi),
#                                    0.5*rad*np.sin(5.0*phi)])).T
#
#     assert data.shape == (n, dim)
#     return data
#
#
#
#
# #*******************************************************************************
#
#
#
#
#
#
#
# def gen_helix1_data(n, dim, d, sampler):
#     assert d < dim
#     assert d == 1
#     assert dim >= 3
#     t = 2 * np.pi / n + sampler(n) * 2 * np.pi
#     data = pd.DataFrame(np.vstack([(2 + np.cos(8*t))*np.cos(t),
#                                    (2 + np.cos(8*t))*np.sin(t),
#                                    np.sin(8*t), np.zeros((dim - 3, n))])).T
#     assert data.shape == (n, dim)
#     return data
#
# def gen_helix2_data(n, dim, d, sampler):
#     assert d < dim
#     assert d == 2
#     assert dim >= 3
#     r = 10 * np.pi * sampler(n)
#     p = 10 * np.pi * sampler(n)
#     data = pd.DataFrame(np.vstack([r*np.cos(p), r*np.sin(p),
#                                    0.5*p, np.zeros((dim - 3, n))])).T
#     assert data.shape == (n, dim)
#     return data
#
# def gen_helicoid_data(n, dim, d, sampler):
#     assert d <= dim
#     assert d == 2
#     assert dim >= 3
#     u = 2 * np.pi / n + sampler(n) * 2 * np.pi
#     v = 5 * np.pi * sampler(n)
#     data = pd.DataFrame(np.vstack([np.cos(v),
#                                    np.sin(v) * np.cos(v),
#                                    u,
#                                    np.zeros((dim - 3, n))])).T
#     assert data.shape == (n, dim)
#     return data
#
# def gen_scurve_data(n, dim, d, sampler):
#     assert d < dim
#     assert dim >= 3
#     assert d == 2
#     t = 3 * np.pi * (sampler(n) - 0.5)
#     p = 2.0 * sampler(n)
#
#     data = pd.DataFrame(np.vstack([np.sin(t),
#                                    p,
#                                    np.sign(t) * (np.cos(t) - 1),
#                                    np.zeros((dim - d - 1, n))])).T
#     assert data.shape == (n, dim)
#     return data
#
# def gen_affine_data(n, dim, d, sampler):
#     assert dim >= d
#
#     p = sampler(d, n) * 5 - 2.5
#     v = np.eye(dim, d)
# #     v = np.random.randint(0, 10, (dim, d))
#     data = pd.DataFrame(v.dot(p).T)
#
#     assert data.shape == (n, dim)
#     return data
#
# def gen_affine3_5_data(n, dim, d, sampler):
#     assert dim == 5
#     assert d == 3
#
#     p = 4 * sampler(d, n)
#     A = np.array([[ 1.2, -0.5, 0],
#                   [ 0.5,  0.9, 0],
#                   [-0.5, -0.2, 1],
#                   [ 0.4, -0.9, -0.1],
#                   [ 1.1, -0.3, 0]])
#     b = np.array([[3, -1, 0, 0, 8]]).T
#     data = A.dot(p) + b
#     data = pd.DataFrame(data.T)
#
#     assert data.shape == (n, dim)
#     return data
#
# def gen_nonlinear4_6_data(n, dim, d, sampler):
#     assert dim == 6
#     assert d == 4
#
#     p0, p1, p2, p3 = sampler(d, n)
#     data = pd.DataFrame(np.vstack([p1**2 * np.cos(2*np.pi*p0),
#                                    p2**2 * np.sin(2*np.pi*p0),
#                                    p1 + p2 + (p1-p3)**2,
#                                    p1 - 2*p2 + (p0-p3)**2,
#                                   -p1 - 2*p2 + (p2-p3)**2,
#                                    p0**2 - p1**2 + p2**2 - p3**2])).T
#
#     assert data.shape == (n, dim)
#     return data
#
