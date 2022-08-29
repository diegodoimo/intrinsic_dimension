import numpy as np
import argparse
import os
import pathlib
import time

from dadapy import data
import utils.functions as ut
from utils.pairwise_distance import compute_mus

def poisson_dataset_two_step(ntot, d):

    ndata = np.random.poisson(ntot, size = 1)[0]
    radii = np.random.uniform(size = ndata)**(1/d)
    radii = np.sort(radii)


    #sample from d-sphere
    u = np.random.normal(loc = 0,scale = 1, size = (ndata, d))  # an array of d normally distributed random variables
    theta = u/np.linalg.norm(u, axis = 1, keepdims = True)
    #poisson process dataset
    data = theta*radii[:, None]

    return data

#*******************************************************************************
def poisson_dataset(ndata, d):

    #V = A*(r^d_i-r^d_{i-1}) ; l'area della superficie sferica A,  è una costante consideriamola 1 nel calcolo di raggi
    shell = np.random.exponential(scale = 1, size = ndata-1)
    volumes = [0]
    for i in range(len(shell)):
        volumes.append(volumes[i]+shell[i])
    volumes = np.array(volumes)
    radii=volumes**(1/d)

    #sample from d-sphere
    u = np.random.normal(loc = 0,scale = 1, size = (ndata, d))  # an array of d normally distributed random variables
    theta = u/np.linalg.norm(u, axis = 1, keepdims = True)
    #poisson process dataset
    data = theta*radii[:, None]

    return data

#*******************************************************************************
parser = argparse.ArgumentParser()
parser.add_argument('--nsamples', default=1, type=int)
parser.add_argument('--ndata', default=1000, type=int)
parser.add_argument('--ntot', default = None, type=int)
parser.add_argument('--id', default=2, type=int)
parser.add_argument('--n', default=8, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--no_sklearn', action = 'store_true')
parser.add_argument('--filename', default = '')
parser.add_argument('--results_path', metavar = 'DIR', default = './test')

args = parser.parse_args([])

#*******************************************************************************
# args.nsamples = 10
# args.n = 8
# args.ndata = 128
# args.id = 3
# np.random.seed(args.seed)


#code to estimate how many datapoints are safely enough to avoid boundary effects
test_sample = poisson_dataset(max(args.ndata, 2*2**args.n), args.id)
r_ndata = np.linalg.norm(test_sample[-1])
r_n = np.linalg.norm(test_sample[2*2**args.n-1])
if r_ndata> r_n: r_tot = r_ndata + 2*r_n
else: r_tot = 2*r_n

ntot = int(test_sample.shape[0]*(r_tot/(max(r_ndata, r_n)))**args.id)

ntot = max(ntot, 5000)
if args.ntot is not None:
    ntot = args.ntot
    args.filename += f'_{args.ntot}'

#*******************************************************************************


ids = np.empty((args.nsamples, args.n))
ids_std = np.empty((args.nsamples, args.n))
print(f'id = {args.id}\nn_samples = {args.nsamples}\nntot = {ntot}\nndata = {args.ndata}')

#s = 0
#dataset  = poisson_dataset_two_step(ntot_vec[s], args.id)

for s in range(args.nsamples):
    if s%20==0:
        print(f'ndata {args.ndata}: {s} samples computed')

    # dataset  = poisson_dataset(ntot, args.id)
    # the number of realizations inside the unit ball are poisson distributed. The data are then uniformly sampled.
    dataset  = poisson_dataset_two_step(ntot, args.id)

    #compute mus
    if not args.no_sklearn:
        mus = compute_mus(X = dataset[:args.ndata], Y = dataset, range_scaling = int(2*2**args.n), working_memory = 4096)[0]

    else:
        #explicit mu computation, slower
        step_size = np.array([2**i for i in range(args.n+1)])
        mus = np.empty((args.ndata, args.n))
        count = 0
        for j in range(args.ndata):
            dist = np.linalg.norm(dataset-dataset[j], axis = 1)

            indx = np.nonzero(dist < np.finfo(dataset.dtype).eps)[0]
            if len(indx)>1:
                count+=len(indx)
                dist[indx] = 5*np.finfo(dataset.dtype).eps
            dist.sort()
            mus[j] = np.divide(dist[step_size[1:]], dist[step_size[:len(step_size)-1]])

        if len(indx)>1:
            print(f'there are {count} couples at 0 distance')

    #estimate ID
    N = mus.shape[0]
    for i in range(mus.shape[1]):
        n1 = 2**i
        id = ut._argmax_loglik(
                    dtype =mus.dtype,
                    d0=10**-3,
                    d1=10**3,
                    mus = mus[:, i],
                    n1 = n1,
                    n2= 2 * n1,
                    N = N,
                    eps=1e-8
                    )
        ids[s, i] = id

        ids_std[s, i] = (
            1/ ut._fisher_info_scaling(
            id, mus[:, i], n1, 2 * n1, eps=10*np.finfo(mus.dtype).eps)
                    ) ** 0.5

if not os.path.isdir(f'{args.results_path}'):
    pathlib.Path(f'{args.results_path}').mkdir(parents=True, exist_ok=True)
np.save(f'{args.results_path}/id{args.id}_nrep{args.nsamples}_ndata{args.ndata}{args.filename}.npy', ids)
np.save(f'{args.results_path}/id{args.id}_fisher_info_nrep{args.nsamples}_ndata{args.ndata}{args.filename}.npy', ids_std)

#*******************************************************************************
# import matplotlib.pyplot as plt
# plt.fill_between(np.arange(ids.shape[1]), np.mean(ids, axis = 0)+2*np.std(ids, axis = 0)/ids.shape[0]**0.5,
# np.mean(ids, axis = 0)-2*np.std(ids, axis = 0)/ids.shape[0]**0.5, alpha = 0.1)
#
#
# plt.plot(np.arange(ids.shape[1]), np.mean(ids, axis = 0))
# plt.hlines(2., 0, 8)
# plt.fill_between(np.arange(ids.shape[1]), ids[0]+2*ids_std[0],ids[0]-2*ids_std[0], alpha = 0.2)
# plt.plot(np.arange(ids.shape[1]), ids[0])
#
#
#
# #-------------------------------------------------------------------------------
# data_tot = uni_from_exponential(200, 2)
# plt.scatter(data_tot[:, 0], data_tot[:,1], s= 5)
# plt.savefig('Uniform_2d.png', dpi = 200)
