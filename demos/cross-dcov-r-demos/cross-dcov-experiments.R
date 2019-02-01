# Ronak Mehta
# Time Series Dependence Experiments

# Import functions: change for other users.
source('C:/Users/Ronak Mehta/Desktop/Jovo/cross-dcov-functions.R')

# Parameters shared among all experiments.
Sigma <- matrix(c(1, 0, 0, 1), 2, 2)
num_sims <- 100
alpha <- 0.05
num_boot <- 100
sample_sizes <- seq(from = 10, to = 510, by = 100)

# X_t independent of Y_t, but dependent on Y_{t-1}.
'
Phis <- list(matrix(c(0, 0.5, 0.5, 0), 2, 2))
Thetas <- list()
M <- 0
exp_name <- "AR(1), M = 0, Phi = [0, 0.5; 0.5, 0]"

run_experiment(Sigma, Phis, Thetas, M, num_sims, sample_sizes, num_boot, exp_name)


Phis <- list(matrix(c(0, 0.25, 0.25, 0), 2, 2))
Thetas <- list()
M <- 1
exp_name <- "AR(1), M = 1, Phi = [0, 0.25; 0.25, 0]"

run_experiment(Sigma, Phis, Thetas, M, num_sims, sample_sizes, num_boot, exp_name)
'

Phis <- list()
Thetas <- list(matrix(c(0.5, 0, 0, -0.5), 2, 2))
M <- 3
exp_name <- "AR(1), M = 3, Theta = [0.5, 0; 0, -0.5]"

run_experiment(Sigma, Phis, Thetas, M, num_sims, sample_sizes, num_boot, exp_name)



