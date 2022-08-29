from functools import partial
import numpy as np
#from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import _precompute_metric_params
from sklearn.utils.validation import _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import sys
from joblib import parallel_backend, Parallel, delayed
import time
import math
import psutil
import gc
import copy

def _kneighbors_reduce_func(dist, start, n_neighbors = 3, range_scaling=None, argsort = False):
	"""Reduce a chunk of distances to the nearest neighbors.
	Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`
	Parameters
	----------
	dist : ndarray of shape (n_samples_chunk, n_samples)
		The distance matrix.
	start : int
		The index in X which the first row of dist corresponds to.
	n_neighbors : int
		Number of neighbors required for each sample.
	return_distance : bool
		Whether or not to return the distances.
	Returns
	-------
	dist : array of shape (n_samples_chunk, n_neighbors)
		Returned only if `return_distance=True`.
	neigh : array of shape (n_samples_chunk, n_neighbors)
		The neighbors indices.
	"""


	"maybe faster without argpartition"
	max_step = int(math.log(range_scaling, 2))
	steps = np.array([2**i for i in range(max_step)])
	sample_range = np.arange(dist.shape[0])[:, None]
	if argsort:
		neigh_ind = np.argsort(dist, axis = 1)[:, : range_scaling]
		#print(neigh_ind.shape)
	else:
		neigh_ind = np.argpartition(dist, steps[-1], axis=1)
		neigh_ind = neigh_ind[:, :steps[-1]+1]
		# argpartition doesn't guarantee sorted order, so we sort again
		neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
		#print(neigh_ind.shape)

	#print(neigh_ind.shape)
	"compute mus and rs"
	dist = np.sqrt(dist[sample_range, neigh_ind])

	# find all points with any zero distance
	indx_ = np.nonzero(dist[:, 1] < np.finfo(dist.dtype).eps)[0]
	# set nearest distance to eps:
	dist[indx_, 1] = np.finfo(dist.dtype).eps


	mus = dist[:, steps[1:]] / dist[:, steps[:-1]]
	rs = dist[:, np.array([steps[:-1], steps[1:]])]

	dist = copy.deepcopy(dist[:, : n_neighbors])
	neigh_ind = copy.deepcopy(neigh_ind[:, : n_neighbors])
	#
	return dist, neigh_ind, mus, rs


def compute_mus(X, range_scaling = None, Y = None, working_memory=1024, n_jobs=1, argsort = False):
    """
    Description:
        adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
        It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
        instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
        up to neighbors' rank 'range scaling'.
        For big datasets it avoids out of memory errors

    Args:
        range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

    Returns:
        dist (np.ndarray(float)): the FULL distance matrix sorted in increasing order of neighbor distances up to maxk
        neighb_ind np.ndarray(int)): the FULL matrix of the indices of the nearest neighbors up to maxk
        mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
        rs np.ndarray(float)): the FULL matrix of the distances of the neighbors involved in the mu estimates
    """

    reduce_func = partial(
        _kneighbors_reduce_func, range_scaling = range_scaling, argsort = argsort
    )

    kwds = {"squared": True}
    if Y is None:
        Y = X
    chunked_results = list(
        pairwise_distances_chunked(
            X,
            Y,
            reduce_func=reduce_func,
            metric="euclidean",
            n_jobs=n_jobs,
            working_memory=working_memory,
            **kwds,
        )
    )

    neigh_dist, neigh_ind, mus, rs = zip(*chunked_results)
    return mus



"below is not necessary, we can just use the pairwise distance chunked from the library"
def pairwise_distances_chunked(
    X,
    Y=None,
    *,
    reduce_func=None,
    metric="euclidean",
    n_jobs=None,
    working_memory=None,
    **kwds,
):

    n_samples_X = _num_samples(X)
    if metric == "precomputed":
        slices = (slice(0, n_samples_X),)
    else:
        if Y is None:
            Y = X
        # We get as many rows as possible within our working_memory budget to
        # store len(Y) distances in each row of output.
        #
        # Note:
        #  - this will get at least 1 row, even if 1 row of distances will
        #    exceed working_memory.
        #  - this does not account for any temporary memory usage while
        #    calculating distances (e.g. difference of vectors in manhattan
        #    distance.
        chunk_n_rows = get_chunk_n_rows(
            row_bytes=8 * _num_samples(Y),
            max_n_rows=n_samples_X,
            working_memory=working_memory,
        )
        slices = gen_batches(n_samples_X, chunk_n_rows)

    # precompute data-derived metric params
    params = _precompute_metric_params(X, Y, metric=metric, **kwds)
    kwds.update(**params)

    # return Parallel(n_jobs=n_jobs, require='sharedmem')(
	#      delayed(compute_distance_slice)(sl, n_samples_X, X, Y, metric, n_jobs, _num_samples, reduce_func, _check_chunk_size, kwds) for sl in slices)

    end = time.time()
    for sl in slices:
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        #print('RAM memory % used before distance:', psutil.virtual_memory()[2])
        D_chunk = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=n_jobs, **kwds)

        #print('RAM memory % used after distance:', psutil.virtual_memory()[2])
        #print('distance_shape:', D_chunk.shape)
        if (X is Y or Y is None):# and PAIRWISE_DISTANCE_FUNCTIONS.get(metric, None) is euclidean_distances:
            # zeroing diagonal, taking care of aliases of "euclidean",
            # i.e. "l2"
            D_chunk.flat[sl.start :: _num_samples(X) + 1] = 0
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            end0 = time.time()
            #print('RAM memory % used before reduce:', psutil.virtual_memory()[2])
            D_chunk = reduce_func(D_chunk, sl.start)
            #print('RAM memory % used after reduce:', psutil.virtual_memory()[2])
            #print('distance_shape:', D_chunk[0].shape, D_chunk[1].shape)
            _check_chunk_size(D_chunk, chunk_size)
            #print(f'{D_chunk[0].shape[0]} distances computed in {(end0-end)/60}min, sorted in {(time.time()-end0)/60}min')
            sys.stdout.flush()
            end = time.time()
        yield D_chunk


def compute_distance_slice(sl, n_samples_X, X, Y, metric, n_jobs, _num_samples, reduce_func, _check_chunk_size, kwds):
        if sl.start == 0 and sl.stop == n_samples_X:
            X_chunk = X  # enable optimised paths for X is Y
        else:
            X_chunk = X[sl]
        D_chunk = pairwise_distances(X_chunk, Y, metric=metric, n_jobs=1, **kwds)
        if (X is Y or Y is None):#and PAIRWISE_DISTANCE_FUNCTIONS.get(metric, None) is euclidean_distances:
            # zeroing diagonal, taking care of aliases of "euclidean",
            # i.e. "l2"
            D_chunk.flat[sl.start :: _num_samples(X) + 1] = 0
        if reduce_func is not None:
            chunk_size = D_chunk.shape[0]
            D_chunk = reduce_func(D_chunk, sl.start)
            _check_chunk_size(D_chunk, chunk_size)
        return D_chunk

def _check_chunk_size(reduced, chunk_size):
    """Checks chunk is a sequence of expected size or a tuple of same."""
    if reduced is None:
        return
    is_tuple = isinstance(reduced, tuple)
    if not is_tuple:
        reduced = (reduced,)
    if any(isinstance(r, tuple) or not hasattr(r, "__iter__") for r in reduced):
        raise TypeError(
            "reduce_func returned %r. Expected sequence(s) of length %d."
            % (reduced if is_tuple else reduced[0], chunk_size)
        )
    if any(_num_samples(r) != chunk_size for r in reduced):
        actual_size = tuple(_num_samples(r) for r in reduced)
        raise ValueError(
            "reduce_func returned object of length %s. "
            "Expected same length as input: %d."
            % (actual_size if is_tuple else actual_size[0], chunk_size)
        )
