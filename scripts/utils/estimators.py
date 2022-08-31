# Copyright 2021-2022 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This module implements contains the implementation of the IdEstimation class."""
import copy
import math
import multiprocessing
from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors

cores = multiprocessing.cpu_count()
rng = np.random.default_rng()


# def compute_cross_nn_distances(X_new, X, maxk, metric="euclidean", period=None):
#
#     # nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric, p=p).fit(X)
#     nbrs = NearestNeighbors(n_neighbors=maxk, metric=metric).fit(X)
#
#     distances, dist_indices = nbrs.kneighbors(X_new)
#
#     return distances, dist_indices
def compute_nn_distances(X, maxk, metric="euclidean", period=None):

    nbrs = NearestNeighbors(n_neighbors=maxk+1, metric=metric).fit(X)
    distances, dist_indices = nbrs.kneighbors(X)
    #
    # distances, dist_indices = compute_cross_nn_distances(
    #     X, X, maxk + 1, metric=metric, period=period
    # )
    return distances, dist_indices

# ----------------------------------------------------------------------------------------------

def _compute_id_2NN(mus, fraction, algorithm="base"):

    N = mus.shape[0]
    N_eff = int(N * fraction)
    mus_reduced = np.sort(mus)[:N_eff]

    if algorithm == "ml":
        intrinsic_dim = (N - 1) / np.sum(mus)

    elif algorithm == "base":
        y = -np.log(1 - np.arange(1, N_eff + 1) / N)

        def func(x, m):
            return m * x

        intrinsic_dim, _ = curve_fit(func, mus_reduced, y)

    else:
        raise ValueError("Please select a valid algorithm type")

    return intrinsic_dim

    # ----------------------------------------------------------------------------------------------
def compute_id_2NN(
    X, N, algorithm="base", fraction=0.9, decimation=1, set_attr=True
):

    nrep = int(np.rint(1.0 / decimation))
    ids = np.zeros(nrep)
    rs = np.zeros(nrep)

    for j in range(nrep):

        N_subset = int(np.rint(N * decimation))
        idx = np.random.choice(N, size=N_subset, replace=False)
        X_decimated = X[idx]

        distances, dist_indices = compute_nn_distances(
            X_decimated,
            maxk=3,  # only compute first 2 nn
            metric='euclidean',
        )

        mus = np.log(distances[:, 2] / distances[:, 1])
        ids[j] = _compute_id_2NN(mus, fraction, algorithm)
        rs[j] = np.mean(distances[:, np.array([1, 2])])

    intrinsic_dim = np.mean(ids)
    intrinsic_dim_err = np.std(ids) / len(ids) ** 0.5
    intrinsic_dim_scale = np.mean(rs)

    return intrinsic_dim, intrinsic_dim_err, intrinsic_dim_scale

    # ----------------------------------------------------------------------------------------------

def return_id_scaling_2NN(
    X,
    N_min=10,
    algorithm="base",
    fraction=0.9,
):
    N = X.shape[0]
    max_ndec = int(math.log(N, 2)) - 1
    Nsubsets = np.round(N / np.array([2**i for i in range(max_ndec)]))
    Nsubsets = Nsubsets.astype(int)

    if N_min is not None:
        Nsubsets = Nsubsets[Nsubsets > N_min]

    ids_scaling = np.zeros(Nsubsets.shape[0])
    ids_scaling_err = np.zeros(Nsubsets.shape[0])
    rs_scaling = np.zeros((Nsubsets.shape[0]))

    for i, N_subset in enumerate(Nsubsets):

        ids_scaling[i], ids_scaling_err[i], rs_scaling[i] = compute_id_2NN(X, N,
            algorithm=algorithm,
            fraction=fraction,
            decimation=N_subset / N,
            set_attr=False,
        )

    return ids_scaling, ids_scaling_err, rs_scaling

# ----------------------------------------------------------------------------------------------
def return_id_scaling_gride(X, range_max=64, d0=0.001, d1=1000, eps=1e-7, mg_estimator = False, unbiased = False):

    N = X.shape[0]
    max_rank = min(N, range_max)

    distances, dist_indices, mus, rs = _return_mus_scaling(X,
            range_scaling=max_rank, maxk = 30, mg_estimator=mg_estimator, unbiased = unbiased
        )

    if mg_estimator:
        rs_scaling = np.mean(rs, axis = 0)
        ids_scaling = 1/np.mean(mus, axis = 0)
        "error is not implemented"
        ids_scaling_err = np.zeros_like(ids_scaling)

    else:
        # array of ids (as a function of the average distance to a point)
        ids_scaling = np.zeros(mus.shape[1])
        # array of error estimates (via fisher information)
        ids_scaling_err = np.zeros(mus.shape[1])
        "average of the kth and 2*kth neighbor distances taken over all datapoints for each id estimate"
        rs_scaling = np.mean(rs, axis=(0, 1))
        for i in range(mus.shape[1]):
            n1 = 2**i
            id = _argmax_loglik(
                mus.dtype, d0, d1, mus[:, i], n1, 2 * n1, N, eps=eps
            )  # eps=precision id calculation
            ids_scaling[i] = id

            ids_scaling_err[i] = (
                1
                / _fisher_info_scaling(
                    id, mus[:, i], n1, 2 * n1, eps=5 * 1e-10
                )  # eps=regularization small numbers
            ) ** 0.5

    return ids_scaling, ids_scaling_err, rs_scaling

def _remove_zero_dists(distances):
    dtype = distances.dtype

    # find all points with any zero distance
    indx_ = np.nonzero(distances[:, 1] < np.finfo(dtype).eps)[0]
    # set nearest distance to eps:
    distances[indx_, 1] = np.finfo(dtype).eps

    return distances

    # ----------------------------------------------------------------------------------------------
def _mus_scaling_reduce_func(dist, start, range_scaling, maxk, mg_estimator = False, unbiased = False):

    max_step = int(math.log(range_scaling, 2))
    steps = np.array([2**i for i in range(max_step)])

    sample_range = np.arange(dist.shape[0])[:, None]
    neigh_ind = np.argpartition(dist, range_scaling - 1, axis=1)
    neigh_ind = neigh_ind[:, :range_scaling]

    # argpartition doesn't guarantee sorted order, so we sort again
    neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

    dist = np.sqrt(dist[sample_range, neigh_ind])
    dist = _remove_zero_dists(dist)

    if mg_estimator:
        mus = np.zeros((dist.shape[0], len(steps)-1))
        rs = np.zeros((dist.shape[0], len(steps)-1))
        for i in range(len(steps)-1):
            k = steps[i+1]
            tmp = np.log(dist[:, k:k+1]/dist[:, 1:k])
            if unbiased:
                print(k)
                mus[:, i] = tmp.sum(axis=1)/(k- 2)
            else:
                mus[:, i] = tmp.sum(axis=1)/(k- 1)
            rs[:, i] = np.mean(dist[:, :k+1], axis = 1)
            #print(rs.shape, mus.shape)
    else:
        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
        rs = dist[:, np.array([steps[:-1], steps[1:]])]

    dist = copy.deepcopy(dist[:, : maxk + 1])
    neigh_ind = copy.deepcopy(neigh_ind[:, : maxk + 1])

    return dist, neigh_ind, mus, rs

def _return_mus_scaling(X, range_scaling, maxk=30, mg_estimator = False, unbiased = False):
    """Return the "mus" needed to compute the id.

    Adapted from kneighbors function of sklearn
    https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
    It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
    instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
    up to neighbors' rank 'range scaling'.
    For big datasets it avoids out of memory errors

    Args:
        range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

    Returns:
        dist (np.ndarray(float)): the FULL distance matrix sorted in increasing order of distances up to maxk
        neighb_ind np.ndarray(int)): the FULL matrix of the indices of the nearest neighbors up to maxk
        mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
        rs np.ndarray(float)): the FULL matrix of the distances of the neighbors involved in the mu estimates
    """
    reduce_func = partial(
        _mus_scaling_reduce_func, range_scaling=range_scaling, maxk = maxk, mg_estimator = mg_estimator, unbiased = unbiased
    )

    kwds = {"squared": True}
    chunked_results = list(
        pairwise_distances_chunked(
            X,
            X,
            reduce_func=reduce_func,
            metric='euclidean',
            n_jobs=1,
            working_memory=1024,
            **kwds,
        )
    )

    neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)

    return (
        np.vstack(neigh_dist),
        np.vstack(neigh_ind),
        np.vstack(mus),
        np.vstack(rs),
    )


def _loglik(d, mus, n1, n2, N, eps):
    one_m_mus_d = 1.0 - mus ** (-d)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps
    sum = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return sum - (N - 1) / d


def _argmax_loglik(dtype, d0, d1, mus, n1, n2, N, eps=1.0e-7):
    # mu can't be == 1 add some noise
    indx = np.nonzero(mus == 1)
    mus[indx] += 1e-10  # np.finfo(dtype).eps

    l1 = _loglik(d1, mus, n1, n2, N, eps)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = _loglik(d2, mus, n1, n2, N, eps)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    d = (d0 + d1) / 2.0
    return d


def _fisher_info_scaling(id_ml, mus, n1, n2, eps):
    N = len(mus)
    one_m_mus_d = 1.0 - mus ** (-id_ml)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < eps] = eps
    log_mu = np.log(mus)

    j0 = N / id_ml**2

    factor1 = np.divide(log_mu, one_m_mus_d)
    factor2 = mus ** (-id_ml)
    tmp = np.multiply(factor1**2, factor2)
    j1 = (n2 - n1 - 1) * np.sum(tmp)

    return j0 + j1
