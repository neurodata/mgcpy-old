require("mgc")
  
base_dir = "/Users/pikachu/OneDrive - Johns Hopkins University/Mac Desktop/NDD I/mgcpy/mgcpy/independence_tests/unit_tests/mgc/data/"

simulations = c("linear_sim", "exp_sim", "cub_sim", "joint_sim", 
               "step_sim", "quad_sim", "w_sim", "spiral_sim", 
               "ubern_sim", "log_sim", "root_sim", "sin_sim", 
               "sin_sim_16", "square_sim", "two_parab_sim", 
               "circle_sim", "ellipsis_sim", "square_sim_", 
               "multi_noise_sim", "multi_indep_sim")

for (simulation in simulations){
  print(simulation)
  
  x = read.csv(paste(base_dir, paste(simulation, "_x.csv", sep=""), sep=""), header = FALSE)
  y = read.csv(paste(base_dir, paste(simulation, "_y.csv", sep=""), sep=""), header = FALSE)
  
  res = mgc.test(x$V1, y$V1)
  write.csv(res,paste(base_dir, paste(simulation, "_res.csv", sep=""), sep=""), row.names = FALSE)
}
  
