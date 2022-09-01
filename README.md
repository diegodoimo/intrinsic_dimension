# Scaling analysis of intrinsic dimension with GRIDE

Code and models from the paper ["blabla"](some link to paper arxiv)

Platforms:

- Ubuntu 22.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html, or install the dependencies shown below manually.

```
conda create --name gride python numpy scipy scikit-learn seaborn 
```

```
conda activate gride
pip install dadapy                                      #official version of gride (and much more)
conda install pytorch torchvision cpuonly -c pytorch    #(cifar mnist datasets)
```

To reproduce the ESS estimator tests install R in the conda environment with: 
```
conda activate gride
conda install r-essentials r-base
```
Then open R from terminal typing:
```
R
```
and install the package intrinsicDimension:
```
install.packages("intrinsicDimension")
```
To reproduce the DANCo estimator tests you must have MATLAB installed. We used MATLAB version 2021b.


## Usage

**Dataset generation**
```
cd scripts
conda activate gride
python generate_datasets --syntetic --csv --eps 0.01    #create syntetic datasets in .csv format with noise std 0.01 (50MB required) 
python generate_datasets --syntetic --npy --eps 0.01    #create syntetic datasets in .npy format with noise std 0.01 (17MB required)
python generate_datasets --syntetic --mat --eps 0.01    #create syntetic datasets in .mat (matlab struct)  with noise std 0.01 (17MB required)

python generate_datasets --real                         #download real datasets
```

**Tests on syntetic datasets**
To reproduce decimation analysis on syntetic datasets (fig ...) with gride, twonn, mle (levina-bickel), geomle, with noise 0.01:
```
cd scripts
conda activate gride
python  syntetic_test.py --eps 0.01 --algo 'gride'      #less than 1min
python  syntetic_test.py --eps 0.01 --algo 'twonn'      #less than 1min
python  syntetic_test.py --eps 0.01 --algo 'mle'        #less than 1min
python  syntetic_test.py --eps 0.01 --algo 'geomle'     #(this may take between 10 and 20 minutes)
```

To reproduce decimation analysis on syntetic datasets (fig ...) with ess:
```
cd scripts
conda activate gride
python generate_datasets --syntetic --csv --eps 0.01    #create a set of data with noise std 0.01 (50MB required)
Rscript syntetic_ess.R                                  #(this may take between 20 and 30 minutes)
```

**Tests on real datasets (MNIST, ISOMAP, ISOLET)**
To reproduce decimation analysis on MNIST, ISOMAP, ISOLET datasets (fig ...) with gride, twonn, mle (levina-bickel), geomle:
```
cd scripts
conda activate gride
python  real_test.py --algo 'gride'            #less than 1min
python  real_test.py --algo 'twonn'            #less than 1min
python  real_test.py --algo 'mle'              #less than 1min
python  real_test.py --algo 'geomle'           #(this may take between 10 and 20 minutes)
```


**Time benchmark on cifar10**
To reproduce time benchmark with gride, twonn, mle (levina-bickel), geomle:
```
cd scripts
conda activate gride
python  benchmark_test.py --algo 'gride'            #less than ...
python  benchmark_test.py --algo 'twonn'            #less than ...
python  benchmark_test.py --algo 'mle'              #less than ...
python  benchmark_test.py --algo 'geomle'           #(this may take ...)
```



