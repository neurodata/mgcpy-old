# Ronak Mehta
# Time Series independence test experiements.

# Install and load dependencies.
deps <- c("boot", "energy", "stats", "MASS")
for (dep in deps) {
  if (!require(dep, character.only = TRUE)) {
    install.packages(dep)
    if (!require(dep, character.only = TRUE)) {
      stop(paste("Unable to install dependency '", dep, "', please check connection.", sep = ""))
    }
  }
}

# Full test of independence of univariate time series X and Y
# Using distance correlation and block bootstrapping.
test_independence <- function(X, Y, num_boot, alpha, unbiased = FALSE) {
  
  # Test statistic computation 
  # Considered a function of X holding Y fixed.
  test_stat <- function(X) {
    stat_type <- if (unbiased) "U" else "V"
    length(X)*dcor2d(X, Y, type = stat_type)
  }
  
  # Determine critical value.
  critical_value <- function(X) {
    # Block boostrap
    block.size <- ceiling(sqrt(length(X)))
    boot_result <- tsboot(tseries = X, 
                          statistic = test_stat, 
                          R = num_boot,
                          l = block.size,
                          sim = "fixed",
                          orig.t = FALSE)
    critical_value <- quantile(boot_result$t, 1 - alpha)
  }
  
  S <- test_stat(X)
  c <- critical_value(X)
  result <- list(reject = (S >= c),
                 test_stat = S,
                 critical_value = c,
                 unbiased = unbiased,
                 num_boot = num_boot,
                 alpha = alpha)
}

# Bivariate VAR model simulation.
varma <- function(n, Phis, Thetas, Sigma) {
  Z <- matrix(rep(0, 2*n), n, 2)
  innov <- mvrnorm(n, c(0,0), Sigma)
  Z[1,] <- innov[1,]
  for (t in 2:n) {
    # AR
    if (length(Phis)) {
      for (i in 1:length(Phis)) {
        if (t-i > 0) {
          Z[t,] <- Z[t,] + Z[t-i,] %*% Phis[[i]]
        }
      }
    }
    # MA
    if (length(Thetas)) {
      for (j in 1:length(Thetas)) {
        if (t-j > 0) {
          Z[t,] <- Z[t,] + innov[t-j,] %*% Thetas[[j]]
        }
      }
    }
    # Innovation
    Z[t,] <- Z[t,] + innov[t,]
  }
  colnames(Z) <- c("X", "Y")
  Z
}

# Run experiment with specified parameters.
run_experiment <- function(Sigma, 
                           Phis,   # List of Phi matrices for VARMA.
                           Thetas, # List of Phi matrices for VARMA. 
                           num_replicates, # Number of replicates to estimate power for fixed n 
                           sample.sizes,  
                           alpha, 
                           num_boot,
                           unbiased) {
  # Function and argument 'k' is used just for sapply.
  run_test <- function(k, n, Phis, Thetas, Sigma, num_boot, alpha, unbiased) {
    
    Z <- varma(n, Phis, Thetas, Sigma)
    X <- Z[,"X"]
    Y <- Z[,"Y"]
    
    result <- test_independence(X, Y, num_boot, alpha, unbiased)
    if (result$reject) return(1) else return(0)
  }
  estimate_power <- function(n, num_replicates, Phis, Thetas, Sigma, num_boot, alpha, unbiased) {
    # Run test num_replicates time to estimate power.
    num_rejects <- sum(sapply(1:num_replicates, run_test, n, Phis, Thetas, Sigma, num_boot, alpha, unbiased))
    cat(".")
    num_rejects/num_replicates
  }
  
  powers <- sapply(sample.sizes, estimate_power, num_replicates, Phis, Thetas, Sigma, num_boot, alpha, unbiased)
  cat("\n")
  powers
}

# Display power curve.
power_curve <- function(sample.sizes, powers, alpha, exp_num) {
  plot(sample.sizes,
       powers,
       type = "l",
       family = "serif",
       ylab = "Power",
       xlab = "n",
       ylim = c(0, 1),
       main = paste("Power as a function of sample size: Experiment ", exp_num))
  abline(h = alpha, lty = 2, col = "red")
  abline(h = 1, lty = 2, col = "blue")
}

# TESTS

# Specify bivariate VARMA processes by a list of matrices for Phi
# and list of matrices for Theta. For example, MA(2) would be empty list
# for Phis and list of 2 matrices for Thetas. Sigma is the covariance matrix
# for the white noise.

Sigma <- matrix(c(1, 0, 0, 1), 2, 2)
num_replicates <- 100
alpha <- 0.05
num_boot <- 100
sample.sizes <- seq(from = 10, to = 1510, by = 100)

# 1. White noise, unbiased estimator.

Phis <- list()
Thetas <- list()
unbiased <- TRUE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 1)
dev.copy(png,'exp1.png')
dev.off()

"
# 2. White noise, biased estimator.

Phis <- list()
Thetas <- list()
unbiased <- FALSE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 2)
dev.copy(png,'exp2.png')
dev.off()

# 3. Uncorrelated AR(1), unbiased estimator.

Phis <- list(matrix(c(0.5, 0, 0, 0.5), 2, 2))
Thetas <- list()
unbiased <- TRUE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 3)
dev.copy(png,'exp3.png')
dev.off()

# 4. Uncorrelated MA(1), biased estimator.

Phis <- list()
Thetas <- list(matrix(c(0.5, 0, 0, 0.5), 2, 2))
unbiased <- FALSE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 4)
dev.copy(png,'exp4.png')
dev.off()

# 5. Dependent AR(1), biased estimator.

Phis <- list()
Thetas <- list(matrix(c(0.2, 0.8, 0.8, 0.2), 2, 2))
unbiased <- FALSE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 5)
dev.copy(png,'exp5.png')
dev.off()

# 6. Dependent MA(1), unbiased estimator.

Phis <- list()
Thetas <- list(matrix(c(0.2, 0.8, 0.8, 0.2), 2, 2))
unbiased <- TRUE

powers <- run_experiment(Sigma, Phis, Thetas, num_replicates, sample.sizes, alpha, num_boot, unbiased)
power_curve(sample.sizes, powers, alpha, 6)
dev.copy(png,'exp6.png')
dev.off()
"
