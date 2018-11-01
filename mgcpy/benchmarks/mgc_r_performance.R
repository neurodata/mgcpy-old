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
  times = seq(1, 5, by=1)
  for (t in times){
  time_taken <- microbenchmark(mgc.test(data$X, data$Y), times=1, unit="secs") # best of 5 executions
  print(num_samples)
  print(time_taken[1, 2]/(10^9))
  linear_data <- c(linear_data, list("num_samples"=num_samples, "time_taken"=time_taken[1, 2]/(10^9)))
  }
  
  i <- i + 1
}

print(linear_data)
