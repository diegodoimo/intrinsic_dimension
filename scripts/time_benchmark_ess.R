# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/ESS.R')
#
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/make_vec.R')
#
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/DimEst.R')
# library(yaImpute)
# source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/pointwise.R')


library(intrinsicDimension)
library(reticulate)
library(tictoc)
library(here)
np <- import("numpy")


#setNumThreads()
#tests on real data
getwd()

results_path<-"./results/real_datasets/ess_local/"

ids <- data.frame(N=integer(),
              time=double(),
              ID=double(),
              stringsAsFactors=FALSE)

#cifarP
ids <- data.frame(N=integer(),
              ID=double(),
              stringsAsFactors=FALSE)
for (p in list(4, 5, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181)){

  name<-paste0("../datasets/real/cifar_cat_",toString(p),"x", toString(p), ".npy")
  print(name)
  X <-np$load(name)
  ndata<-nrow(X)

  essPointwiseDimEst <- asPointwiseEstimator(essLocalDimEst, neighborhood.size=30, indices=NULL)
  start<-Sys.time()
  #time<-system.time({
  #id<-essLocalDimEst(X, ver = 'a', d = 1)
  #id_val<-id$dim.est
  ess.pw.res <- essPointwiseDimEst(X)
  id_val<-median(ess.pw.res$dim.est)
  #})
  end<-Sys.time()

  time<-end-start
  print(time)
  print(id_val)
  de<-c(p, time, id_val)
  ids<-rbind(ids, de)
}


filename<-paste0(results_path, "cifar_timeP.txt")
write.table(ids, filename, append = FALSE, sep = " ", dec = ".",
           row.names = FALSE, col.names = FALSE)




#cifarN
ids <- data.frame(N=integer(),
              time=double(),
              ID=double(),
              stringsAsFactors=FALSE)

mat <-np$load("../datasets/real/cifar_training.npy")
ndata<-nrow(mat)

for (fraction in list(1, 2, 4, 8, 16, 32, 64, 128, 256)){
  n <- ndata%/%fraction
  print(fraction)
  s <-sample(nrow(mat),size=n,replace=FALSE)
  X = mat[s, ]
  essPointwiseDimEst <- asPointwiseEstimator(essLocalDimEst, neighborhood.size=30, indices=NULL)

  start<-Sys.time()
  ess.pw.res <- essPointwiseDimEst(X)
  id_val<-median(ess.pw.res$dim.est)
  #id<-essLocalDimEst(X, ver = 'a', d = 1)
  #id_val<-id$dim.est
  end<-Sys.time()
  time<-end-start
  print(time, id_val)
  de<-c(n, time, id_val)
  ids<-rbind(ids, de)
  }

filename<-paste0(results_path, "cifar_timeN.txt")
write.table(ids, filename, append = FALSE, sep = " ", dec = ".",
           row.names = FALSE, col.names = FALSE)
