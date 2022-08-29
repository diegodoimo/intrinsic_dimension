import numpy as np


def _loglik(d, mus, n1, n2, N, eps):
    one_m_mus_d = 1.0 - mus ** (-d)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps
    sum = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return sum - (N - 1) / d


def _argmax_loglik(dtype, d0, d1, mus, n1, n2, N, eps=1.0e-7):
    # mu can't be == 1 add some noise
    indx = np.nonzero(mus == 1)
    mus[indx] += 10*np.finfo(dtype).eps
    #mus[indx] += 1e-10  # np.finfo(dtype).eps

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

#
# def _likelihood_r2n(d, mu, n, N):
#     one_m_mus_d = 1. - mu ** (-d)
#     sum = np.sum(  ((1 - n) / one_m_mus_d + 2. * n - 1.) * np.log(mu)      )
#     return sum - N / d
#
# def _max_lik_r2n(d0, d1, mus, log_stepsize, N, eps = 1.e-7):
#     # mu can't be == 1 add some noise
#     indx = np.nonzero(mus == 1)
#     mus[indx] += np.finfo(mus.dtype).eps
#
#     l1 = _likelihood_r2n(d1, mus, log_stepsize, N)
#     while (abs(d0 - d1) > eps):
#         d2 = (d0 + d1) / 2.
#         l2 = _likelihood_r2n(d2, mus, log_stepsize, N)
#         if l2 * l1 > 0: d1 = d2
#         else:d0 = d2
#     d = (d0 + d1) / 2.
#
#     return d
#
# def _fisher_info_r2n(id_ml, mus, log_stepsize):
#     #to be better summarized
#
#     N = len(mus)
#     mu_d = mus**id_ml
#     log_mu = np.log(mus)
#
#     j0 = N/id_ml**2
#
#     num1 =  np.multiply(mu_d, log_mu)
#     den1 = mu_d -1
#     tmp1 = np.divide(num1, den1)
#     j1 = (log_stepsize-1)*np.sum(tmp1**2)
#
#     num2 = np.multiply(mu_d, log_mu**2)
#     tmp2 = np.divide(num2, mu_d-1)
#     j2 = -(log_stepsize-1)*np.sum(tmp2)
#
#     return j0+j1+j2

def compute_mus(dataset, x0, log_stepsize):

    dist = np.linalg.norm(dataset-x0, axis = 1)
    indx = np.nonzero(dist < np.finfo(x0.dtype).eps)[0]

    if len(indx)>1:
        print(f'there are {len(indx)-1} couples at 0 distance')
        dist[indx] = np.finfo(x0.dtype).eps

    dist.sort()
    step_size = np.array([2**i for i in range(log_stepsize+1)])
    mus = np.divide(dist[step_size[1:]], dist[step_size[:len(step_size)-1]])
    r1s = dist[step_size[:len(step_size)-1]]

    return mus, r1s



#old function
# def compute_mus(dataset, x0, log_stepsize, dimensions=None):
#
#     # if dimensions is not None:
#     #     delta = np.abs(dataset - x0)
#     #     delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
#     #     dist = np.sqrt((delta ** 2).sum(axis=-1))
#     # else:
#     dist = np.linalg.norm(dataset-x0, axis = 1)
#     #dist = np.sum((dataset-x0)**2, axis = 1)**0.5
#     indx = np.nonzero(dist < np.finfo(x0.dtype).eps)[0]
#
#     if len(indx)>1:
#         print(f'there are {len(indx)-1} couples at 0 distance')
#         dist[indx] = np.finfo(x0.dtype).eps
#     #dist = np.where(dist < np.finfo(float).eps, np.finfo(float).eps, dist)
#
#     dist.sort()
#     #step_size = np.array([2**i for i in range(int(math.log(ndata, 2)))]) #previous version ndata = log_stepsize
#     step_size = np.array([2**i for i in range(log_stepsize+1)])
#     mus = np.divide(dist[step_size[1:]], dist[step_size[:len(step_size)-1]])
#     r1s = dist[step_size[:len(step_size)-1]]
#
#     return mus, r1s
