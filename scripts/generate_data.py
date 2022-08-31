from utils.syntetic_datasets import *
from scipy.io import savemat
import numpy as np


#import rpy2.robjects as robjects
# from rpy2.robjects import r, pandas2ri
# from pandas import DataFrame
#
# import pandas as pd
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
#
# from rpy2.robjects.conversion import localconverter





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

    np.savetxt(f'{data_folder}/csv/{key}_{int(N/1000)}k_eps{eps}.csv', X, delimiter=",")
    
    mdict[key] =  X

# with localconverter(ro.default_converter + pandas2ri.converter):
#   r_from_pd_df = ro.conversion.py2rpy(df)
# r.assign("foo", r_from_pd_df)
# r("save(foo, file='here.gzip', compress=TRUE)")


savemat(f"{data_folder}/matlab_datasets.mat", mdict)
