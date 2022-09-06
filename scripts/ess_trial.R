#source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/ESS.R')

#source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/make_vec.R')

#source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/DimEst.R')
#library(yaImpute)
#source('/home/diego/ricerca/past/IDestimators/intrinsicDimension/R/pointwise.R')
#tests on syntetic data
library(intrinsicDimension)
library(reticulate)
library(tidyverse)
library(here)
np <- import("numpy")
# data reading
getwd()
twonn_isolet <- t(np$load("../datasets/real/cifar_cat_11x11.npy"))


data_files<-list.files("../datasets/syntetic/csv")
results_path<-"./results/syntetic_datasets/ess/"

for(i in 1:length(data_files)) {
    mat <-read.csv(paste0("../datasets/syntetic/csv/", data_files[i]), header = FALSE)

    print(data_files[i])
    ids <- data.frame(N=integer(),
                  ID=double(),
                  stringsAsFactors=FALSE)
    ndata<-nrow(mat)

    for (fraction in list(1, 2, 4,8, 16, 32, 64, 128, 256, 512)){
      n <- ndata%/%fraction
      print(fraction)
      for (nrep in (1:1)){
        print(head(mat))
        print(dim(mat))
        s <-sample(nrow(mat),size=n,replace=FALSE)
        X = mat[s,]
        print(dim(X))
        #data <- swissRoll3Sph(300, 300)
        data_matrix<-as.matrix(sapply(X, as.numeric))
        print(dim(data_matrix))
        print(head(data_matrix))
        #print(is.matrix(data_matrix))

        #print(data_matrix)
        essPointwiseDimEst <- asPointwiseEstimator(essLocalDimEst, neighborhood.size=30, indices=NULL)
        #print(is.matrix(X))
        ess.pw.res <- essPointwiseDimEst(data_matrix)
        id_val<-ess.pw.res$dim.est

        #id<-essLocalDimEst(X, ver = 'a', d = 1)
        #id_val<-id$dim.est
        print(length(id_val))
        print(mean(id_val))
        print(median(id_val))

        de<-c(n, id_val)
        ids<-rbind(ids, de)

        }
      }
    }
