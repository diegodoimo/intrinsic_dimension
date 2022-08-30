# Scaling analysis of intrinsic dimension with GRIDE

Code and models from the paper ["blabla"](some link to paper arxiv)

Platforms:

- Ubuntu 22.04

## Install

You can get miniconda from https://docs.conda.io/en/latest/miniconda.html, or install the dependencies shown below manually.

```
conda create --name gride python numpy scipy scikit-learn
```

```
conda activate gride
pip install dadapy
```

To reproduce the ESS estimator tests install R in the conda environment with: 
```
conda activate gride
conda install r-essentials r-base
pip install rpy2
```
Then open R from terminal typing:
```
R
```
and install the package intrinsicDimension:
```
install.packages("intrinsicDimension")
```
To reproduce the DANCo estimator tests you must have MATLAB installed. We used MATLAB version ...


## Usage
