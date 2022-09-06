# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/ESS.R')
#
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/make_vec.R')
#
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/DimEst.R')
# library(yaImpute)
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/pointwise.R')


library(intrinsicDimension)
library(reticulate)
library(tidyverse)
library(here)
np <- import("numpy")

#tests on real data
getwd()

results_path<-"./results/real_datasets/ess_local/"
data_files = list('isolet.npy', 'mnist_ones.npy')
for(i in 1:length(data_files)) {

    mat <-np$load( paste0("../datasets/real/", data_files[i]) )

    ids <- data.frame(N=integer(),
                  ID=double(),
                  stringsAsFactors=FALSE)
    ndata<-nrow(mat)

    for (fraction in list(1, 2, 4, 8)){
      n <- ndata%/%fraction
      print(fraction)
      for (nrep in (1:fraction)){
        s <-sample(nrow(mat),size=n,replace=FALSE)
        X = mat[s, ]
        essPointwiseDimEst <- asPointwiseEstimator(essLocalDimEst, neighborhood.size=30, indices=NULL)
        ess.pw.res <- essPointwiseDimEst(X)
        id_val<-median(ess.pw.res$dim.est)
        print(id_val)
        #id<-essLocalDimEst(X, ver = 'a', d = 1)
        #id_val<-id$dim.est

        de<-c(n, id_val)
        ids<-rbind(ids, de)

        }
      }

    filename<-paste0(results_path, substring(data_files[i], 1, nchar(data_files[i])-4), ".txt")
    write.table(ids, filename, append = FALSE, sep = " ", dec = ".",
               row.names = FALSE, col.names = FALSE)
  }
