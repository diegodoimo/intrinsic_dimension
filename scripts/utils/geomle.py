__all__ = ('geomle')

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression, Ridge
from functools import partial
import time
import sys

def tolist(x):
    if type(x) in {int, float}:
        return [x]
    if type(x) in {list, tuple}:
        return list(x)
    if type(x) == np.ndarray:
        return x.tolist()

def drop_zero_values(dist):
    mask = dist[:,0] == 0
    dist[mask] = np.hstack([dist[mask][:, 1:], dist[mask][:,0:1]])
    dist = dist[:, :-1]
    assert np.all(dist > 0)
    return dist

def mle_center(X, X_center, k=5, dist=None):
    """
    Returns Levina-Bickel dimensionality estimation

    Input parameters:
    X        - data points
    X_center - data centers
    k        - number of nearest neighbours (Default = 5)
    dist     - matrix of distances to the k nearest neighbors of each point (Optional)

    Returns:
    dimensionality estimation for the k
    """
    assert k>3
    if len(X_center.shape) != 2:
        X_center = X_center.values.reshape(1, -1)
    if dist is None:
        neighb = NearestNeighbors(n_neighbors=k+1, n_jobs=1,
                                  algorithm='ball_tree').fit(X)
        dist, ind = neighb.kneighbors(X_center)
        dist = drop_zero_values(dist)
    dist = dist[:, 0:k]
    assert dist.shape == (X_center.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k - 1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    Rs = dist[:, -1]
    return intdim_sample, Rs

def fit_poly_reg(x, y, w=None, degree=(1, 2), alpha=5e-3):
    """
    Fit regression and return f(0)

    Input parameters:
    x - features (1d-array)
    y - targets (1d-array)
    w - weights for each points (Optional)
    degree - degrees of polinoms (Default tuple(1, 2))
    alpha - parameter of regularization (Default 5e-3)

    Returns:
    zero coefficiend of regression
    """
    X = np.array([x ** i for i in tolist(degree)]).T
    lm = Ridge(alpha=alpha)
    lm.fit(X, y, w)
    return lm.intercept_

def _func(df, degree, alpha):
    gr_df = df.groupby('k')
    R = gr_df['R'].mean().values
    d = gr_df['dim'].mean().values
    std = gr_df['dim'].std().values
    if np.isnan(std).any(): std = np.ones_like(std)
    return fit_poly_reg(R, d, std**-1, degree=degree, alpha=alpha)


def geomle_opt(X, k1=10, k2=40, nb_iter1=10, nb_iter2=20, degree=(1, 2),
           alpha=5e-3, ver='GeoMLE', random_state=None, debug=False):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples

    Input parameters:
    X            - data
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 40)
    nb_iter1     - number of bootstrap iterations (Default = 10)
    nb_iter1     - number of bootstrap iterations for each regresion (Default = 20)
    degree       - (Default = (1, 2))
    alpha        - (Default = 5e-3)
    random_state - random state (Optional)

    Returns:
    array of shape (nb_iter1,) of regression dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    dim_space = X.shape[1]

    ids = []
    rs = []
    iter = 0
    repetition = 0
    while (nb_iter1-iter > 0 and repetition < 3):

    #for i in range(nb_iter1):
        dim_all_, R_all_, k_all_, idx_all_ = [], [], [], []
        start = time.time()
        for j in range(nb_iter2):
            idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
            flag =1
            "make sure that bootstrap sample has more data than k2+2 (non existent in the original implementation...)"
            if len(idx)<k2+2:
                flag =0
                while flag==0:
                    idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
                    if len(idx)>k2+2:
                        flag =1

            X_bootstrap = X[idx]
            neighb = NearestNeighbors(n_neighbors=k2+1, n_jobs=1,
                                      algorithm='brute').fit(X_bootstrap)

            dist, ind = neighb.kneighbors(X)
            dist = drop_zero_values(dist)
            dist = dist[:, 0:k2]
            assert np.all(dist > 0)

            for k in range(k1, k2+1):
                dim, R = mle_center(X_bootstrap, X, k, dist)    #dim = id mle, R = max neighbor distnace
                dim_all_ += list(dim)                           #mle estimates
                R_all_ += list(R)                               #max radii relative to mle estimates
                k_all_ += [k] * dim.shape[0]


        kall_ = np.array(k_all_)
        R_all_ = np.array(R_all_)
        R_ = np.empty(k2+1-k1)          #average radii at different ks
        d_ = np.empty(k2+1-k1)          #average distances at different ks
        dim_all_ = np.array(dim_all_)
        for j, k in enumerate(range(k1, k2+1)):
            mask = kall_ == k
            R_[j] = np.mean(R_all_[mask])
            d_[j] = np.mean(dim_all_[mask])

        #ridge regression
        X_ = np.array([R_ ** l for l in list(degree)]).T
        lm = Ridge(alpha=alpha)
        lm.fit(X_, d_)
        ID = lm.intercept_

        if ID<0:
            if repetition == 2:
                ID = -1
                print('algorithm not converged')
                ids.append(ID)
                rs.append(np.mean(R_))
            else:
                repetition+=1

        elif ID > dim_space:
            if repetition == 2:
                ID = np.inft
                print('algorithm not converged')
                ids.append(ID)
                rs.append(np.mean(R_))
            else:
                repetition+=1
        else:
            iter+=1
            ids.append(ID)
            rs.append(np.mean(R_))


    return np.array(ids), np.array(rs)
































def geomle(X, k1=10, k2=40, nb_iter1=10, nb_iter2=20, degree=(1, 2),
           alpha=5e-3, ver='GeoMLE', random_state=None, debug=False):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples

    Input parameters:
    X            - data
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 40)
    nb_iter1     - number of bootstrap iterations (Default = 10)
    nb_iter1     - number of bootstrap iterations for each regresion (Default = 20)
    degree       - (Default = (1, 2))
    alpha        - (Default = 5e-3)
    random_state - random state (Optional)

    Returns:
    array of shape (nb_iter1,) of regression dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    dim_space = X.shape[1]

    result = []
    result_ = []
    data_reg = []
    for i in range(nb_iter1):

        dim_all, R_all, k_all, idx_all = [], [], [], []
        dim_all_, R_all_, k_all_, idx_all_ = [], [], [], []
        start = time.time()
        for j in range(nb_iter2):
            idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
            flag =1
            "make sure that bootstrap sample has more data than k2+2 (non existent in the original implementation...)"
            if len(idx)<k2+2:
                flag =0
                while flag==0:
                    idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
                    if len(idx)>k2+2:
                        flag =1
            #X_bootstrap = X.iloc[idx]
            X_bootstrap = X[idx]

            neighb = NearestNeighbors(n_neighbors=k2+1, n_jobs=1,
                                      algorithm='brute').fit(X_bootstrap)

            dist, ind = neighb.kneighbors(X)
            dist = drop_zero_values(dist)
            dist = dist[:, 0:k2]
            assert np.all(dist > 0)

            for k in range(k1, k2+1):

                dim, R = mle_center(X_bootstrap, X, k, dist)

                dim_all += dim.tolist()
                R_all += R.tolist()
                idx_all += list(range(nb_examples))
                k_all += [k] * dim.shape[0]
            ##print(dim)

            for k in range(k1, k2+1):

                dim, R = mle_center(X_bootstrap, X, k, dist)

                dim_all_ += list(dim)                    #mle estimates
                R_all_ += list(R)                        #max radii relative to mle estimates
                k_all_ += [k] * dim.shape[0]
            ##print(dim)
        ##print(time.time()-start)

        R_ = np.empty(k2+1-k1)
        d_ = np.empty(k2+1-k1)
        kall_ = np.array(k_all)
        R_all_ = np.array(R_all_)
        dim_all_ = np.array(dim_all_)
        ##print(len(R_))
        for i, k in enumerate(range(k1, k2+1)):
            mask = kall_ ==k
            ##print(i)
            ##print(i, )
            R_[i] = np.mean(R_all_[mask])
            d_[i] = np.mean(dim_all_[mask])

        #print(R_)
        #print(d_)
        #print('here1')
        X_ = np.array([R_ ** i for i in list(degree)]).T
        #for i in range(len(X_)):
            #print(X_[i, 0])
        #print(np.mean(X), len(X))
        lm = Ridge(alpha=alpha)
        lm.fit(X_, d_)
        ID = lm.intercept_
        #return lm.intercept_
        if ID<0: ID =0
        if ID > dim_space:
            ID = dim_space

        result_.append(ID)
        print(ID)

        ##print(time.time()-start)
        ##print("dim_all", dim_all)
        ##print("rall", R_all)
        ##print("idx_all", idx_all)
        ##print("kall", k_all)
        sys.stdout.flush()


        start = time.time()
        data={'dim': dim_all,
              'R': R_all,
              'idx': idx_all,
              'k': k_all}

        df = pd.DataFrame(data, dtype = np.float64)
        if ver == 'GeoMLE':
            func = partial(_func, degree=degree, alpha=alpha)
            reg = df.groupby('idx').apply(func).values.mean()
            data_reg.append(df)

        elif ver == 'fastGeoMLE':
            df_gr = df.groupby(['idx', 'k']).mean()[['R', 'dim']]
            #print(df_gr)
            R = df_gr.groupby('k').R.mean()
            ##print(R)
            d = df_gr.groupby('k').dim.mean()
            #print(d)
            std = df_gr.groupby('k').dim.std()
            #print('here12')
            reg = fit_poly_reg(R, d, std**-1, degree=degree, alpha=alpha)
            data_reg.append((R, d, std))
        else:
            assert False, 'Unknown mode {}'.format(ver)
        reg = 0 if reg < 0 else reg
        reg = dim_space if reg > dim_space else reg

        print(reg)
        print('\n')
        result.append(reg)
        ##print(time.time()-start)

    if debug:
        return np.array(result), data_reg
    else:
        return np.array(result), np.array(result_)




def geomle_(X, k1=10, k2=40, nb_iter1=10, nb_iter2=20, degree=(1, 2),
           alpha=5e-3, ver='GeoMLE', random_state=None, debug=False):
    """
    Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples

    Input parameters:
    X            - data
    k1           - minimal number of nearest neighbours (Default = 10)
    k2           - maximal number of nearest neighbours (Default = 40)
    nb_iter1     - number of bootstrap iterations (Default = 10)
    nb_iter1     - number of bootstrap iterations for each regresion (Default = 20)
    degree       - (Default = (1, 2))
    alpha        - (Default = 5e-3)
    random_state - random state (Optional)

    Returns:
    array of shape (nb_iter1,) of regression dimensionality estimation for k = k1..k2 averaged over bootstrap samples
    """
    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    dim_space = X.shape[1]

    result = []
    for i in range(nb_iter1):

        dim_all, R_all, k_all = [], [], [], []
        start = time.time()
        for j in range(nb_iter2):
            idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
            flag =1
            "make sure that bootstrap sample has more data than k2+1"
            if len(idx)<k2+1:
                flag =0
                while flag==0:
                    idx = np.unique(rng.randint(0, nb_examples - 1, size=nb_examples))
                    if len(idx)>k2+1:
                        flag =1
            #X_bootstrap = X.iloc[idx]
            X_bootstrap = X[idx]

            neighb = NearestNeighbors(n_neighbors=k2+1, n_jobs=1,
                                      algorithm='brute').fit(X_bootstrap)
            dist, ind = neighb.kneighbors(X)
            dist = drop_zero_values(dist)
            dist = dist[:, 0:k2]
            assert np.all(dist > 0)

            for k in range(k1, k2+1):
                dim, R = mle_center(X_bootstrap, X, k, dist)
                dim_all += list(dim)                    #mle estimates
                R_all += list(R)                        #max radii relative to mle estimates
                k_all += [k] * dim.shape[0]
        #print(time.time()-start)

        R = np.empty(k2+1-k1)
        d = np.empty(k2+1-k1)
        kall = np.array(k_all)
        for i, k in enumerate(range(k1, k2+1)):
            mask = kall==k
            R[i] = np.mean(R_all[mask])
            d[i] = np.mean(dim_all[mask])


        X = np.array([R ** i for i in list(degree)]).T
        lm = Ridge(alpha=alpha)
        lm.fit(X, y)
        ID = lm.intercept_
        #return lm.intercept_
        if ID<0: ID =0
        if ID > dim_space:
            ID = dim_space

        result.append(ID)
