rm(list = ls())

require("mgc")
require("energy")
require("pracma")

alpha <- 0.05
repeats <- 1000
test_stats_null <- list()
test_stats_alternative <- list()
for (i in 5:100) {
  data <- mgc.sims.linear(i, 1)
  
  #df = as.data.frame(data$X)
  #write.table(df, file = paste("/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/", i, "_dataX.csv", sep=""),row.names=FALSE, na="",col.names=FALSE)
  #write.table(data$Y, file = paste("/Users/spanda/workspace/mgcpy/mgcpy/benchmarks/spiral_data/", i, "_dataY.csv", sep=""),row.names=FALSE, na="",col.names=FALSE)
  
  for (j in range(repeats)) {
    test_alt <- dcor.test(data$X, data$Y, R=1000)
    test_stats_alternative <- c(test_stats_alternative, list("test_stat"=test_alt$p.value))
    
    permuted_y <- sample(data$Y)
    test_null <- dcor.test(data$X, permuted_y, R=1000)
    test_stats_null <- c(test_stats_null, list("test_stat"=test_null$p.value))
  }
  
  test_stats_null_mat = matrix(unlist(test_stats_null), ncol = 1, byrow = TRUE)
  test_stats_alt_mat = matrix(unlist(test_stats_alternative), ncol = 1, byrow = TRUE)
  sorted_null = sort(test_stats_null_mat)
  cutoff = sorted_null[ceiling(repeats * (1 - alpha))]
  print(sum(test_stats_alt_mat >= cutoff) / repeats)
}

