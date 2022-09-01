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
To reproduce the DANCo estimator tests you must have MATLAB installed. We used MATLAB version 2021b


## Usage

Tests on syntetic datasets
To reproduce decimation analysis on syntetic datasets (fig ...) with gride, twonn, mle (levina-bickel), geomle, with noise 0.01:

```
cd scripts
conda activate gride
python  syntetic_test.py --eps 0.01 --algo 'gride'
python  syntetic_test.py --eps 0.01 --algo 'twonn'
python  syntetic_test.py --eps 0.01 --algo 'mle'
python  syntetic_test.py --eps 0.01 --algo 'geomle' #(this may take between 10 and 20 minutes)
```
