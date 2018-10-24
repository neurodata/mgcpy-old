require("mgc")
require("microbenchmark")

print("Linear data (varying num_samples)")
print("num_samples time_taken(in secs)")

num_samples_range = seq(10, 150, by=10)
linear_data <- list()
i <- 1
for (num_samples in num_samples_range){
  data <- mgc.sims.linear(num_samples, 1, eps=0.1)

  #start_time <- Sys.time()
  #mgc.test(data$X, data$Y)
  #end_time <- Sys.time()

  #time_taken <- end_time - start_time
  #time_taken <- as.numeric(time_taken, units = "secs")

  time_taken <- microbenchmark(mgc.test(data$X, data$Y), times=5, unit="secs") # best of 5 executions

  print(num_samples)
  print(time_taken[1, 2]/(10^9))
  linear_data <- c(linear_data, list("num_samples"=num_samples, "time_taken"=time_taken[1, 2]/(10^9)))
  i <- i + 1
}

print(linear_data)



# Output
# "Linear data (varying num_samples)"
# "num_samples time_taken(in secs)"
# >
# [1] 10
# [1] 1.225284
# [1] 20
# [1] 3.336518
# [1] 30
# [1] 6.566329
# [1] 40
# [1] 11.25737
# [1] 50
# [1] 16.9009
# [1] 60
# [1] 23.86574
# [1] 70
# [1] 34.10607
# [1] 80
# [1] 42.56408
# [1] 90
# [1] 61.35755
# [1] 100
# [1] 71.12478
# [1] 110
# [1] 92.81968
# [1] 120
# [1] 99.42997
# [1] 130
# [1] 120.6133
# [1] 140
# [1] 131.1176
# [1] 150
# [1] 159.2373
#
