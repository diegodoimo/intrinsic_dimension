library(intrinsicDimension)


#tests on syntetic data

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
      for (nrep in (1:fraction)){
        s <-sample(nrow(mat),size=n,replace=FALSE)
        X = mat[s,]
        id<-essLocalDimEst(X, ver = 'a', d = 1)
        id_val<-id$dim.est
        de<-c(n, id_val)
        ids<-rbind(ids, de)

        }
      }

    filename<-paste0(results_path, substring(data_files[i], 1, nchar(data_files[i])-4), ".txt")
    write.table(ids, filename, append = FALSE, sep = " ", dec = ".",
                row.names = FALSE, col.names = FALSE)
  }
