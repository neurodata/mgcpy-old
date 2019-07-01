rm(list = ls())

require("mgc")
require("energy")

test_stats_list <- list()
for (i in 1:21) {
  data <- mgc.sims.spiral(500, 2)
  
  #df = as.data.frame(data$X)
  #write.table(df, file = paste("/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/", i, "_dataX.csv", sep=""),row.names=FALSE, na="",col.names=FALSE)
  #write.table(data$Y, file = paste("/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/", i, "_dataY.csv", sep=""),row.names=FALSE, na="",col.names=FALSE)
  
  test_stat <- dcor.test(data$X, data$Y, R=NULL)
  test_stats_list <- c(test_stats_list, list("test_stat"=test_stat$statistic))
}


print(t(matrix(unlist(test_stats_list))))