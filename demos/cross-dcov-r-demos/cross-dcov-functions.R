# Ronak Mehta
# Cross-distance covariance functions.
# For testing independence of time series.

deps <- c("boot", "energy", "stats", "MASS")
for (dep in deps) {
  if (!require(dep, character.only = TRUE)) {
    install.packages(dep)
    if (!require(dep, character.only = TRUE)) {
      stop(paste("Unable to install dependency '", dep, "', please check connection.", sep = ""))
    }
  }
}

#' Determine critical value of test statistic on time series.
#' 
#' @param X Time series in vector form.
#' @param test_stat Function that outputs number from time series.
#' @param block_size Block size or mean block size.
#' @param type "circular" or "stationary" boostrap.
#' @param num_boot Number of bootstrapped samples.
#' @param alpha Significance level of test.
#' @return The estimated 1 - alpha quantile of the test statistic.
critical_value <- function(X, test_stat, block_size, type = "circular", num_boot = 100, alpha = 0.05) {
  if (type == "stationary") sim <- "geom" else sim <- "fixed"
  boot_result <- tsboot(tseries = X, 
                        statistic = test_stat, 
                        R = num_boot,
                        l = block_size,
                        sim = "fixed",
                        orig.t = FALSE)
  critical_value <- quantile(boot_result$t, 1 - alpha)
}

#' Full test of independence of time series X and Y.
#' 
#' @param X Time series as n length vector.
#' @param Y Time series as n ength vector.
#' @param M Maximum lag to consider for cross-dCov.
#' @param num_boot Number of bootstrapped samples.
#' @param alpha Significance level of test.
#' @param unbiased Whether to use the biased or unbiased estimate of dCov.
#' @param type "circular" or "stationary" boostrap.
#' @param block_size Block size or mean block size.
#' @return A list of decision, test stat, critical value, and params.
test_independence <- function(X, Y, M, num_boot, alpha, unbiased = FALSE, type = "circular", block_size = NA) {
  
  # Default block size is sqrt(n).
  if (is.na(block_size)) block_size <- floor(sqrt(length(X)))
  
  # Test statistic computation - a function of X holding Y fixed.
  # Must be defined internally to maintain Y in scope.
  # Uses Bartlett kernel to adapt FP statistic.
  test_stat <- function(X) {
    N <- length(X)
    p <- sqrt(N)
    stat_type <- if (unbiased) "U" else "V"
    
    result <- N*dcor2d(X, Y, type = stat_type)
    if (M) {
      for (j in 1:M) {
        x <- X[(1+j):N]
        y <- Y[1:(N-j)]
        result <- result + (N-j)*(1 - j/(p*(M+1)))*dcor2d(x, y, type = stat_type)
        
        x <- X[1:(N-j)]
        y <- Y[(1+j):N]
        result <- result + (N-j)*(1 - j/(p*(M+1)))*dcor2d(x, y, type = stat_type)
      }
    }
    result
  }
  
  S <- test_stat(X)
  c <- critical_value(X, test_stat, block_size, type, num_boot, alpha)
  
  result <- list(reject = (S >= c),
                 test_stat = S,
                 critical_value = c,
                 unbiased = unbiased,
                 num_boot = num_boot,
                 alpha = alpha)
}

#' Bivariate VARMA process simulation.
#' 
#' @param n Sample size.
#' @param Phis A list of 2 by 2 matrices, representing Phi coeffs.
#' @param Thetas A list of 2 by 2 matrices, representing Theta coeffs.
#' @param Sigma 2 x 2 covariance matrix for noise.
#' @return n x 2 matrix of observations, x dimensions then y dimensions.
varma <- function(n, Phis, Thetas, Sigma) {
  d <- nrow(Sigma)
  Z <- matrix(rep(0, d*n), n, d)
  innov <- mvrnorm(n, rep(0, d), Sigma)
  Z[1,] <- innov[1,]
  for (t in 2:n) {
    # AR.
    if (length(Phis)) {
      for (i in 1:length(Phis)) {
        if (t-i > 0) {
          Z[t,] <- Z[t,] + Z[t-i,] %*% Phis[[i]]
        }
      }
    }
    # MA.
    if (length(Thetas)) {
      for (j in 1:length(Thetas)) {
        if (t-j > 0) {
          Z[t,] <- Z[t,] + innov[t-j,] %*% Thetas[[j]]
        }
      }
    }
    # Innovation.
    Z[t,] <- Z[t,] + innov[t,]
  }
  colnames(Z) <- c("X", "Y")
  Z
}

#' Display power as a function of sample size.
#'
#' @param sample_sizes Vector of sample sizes at which to estimate power.
#' @param powers Vector of power estimates at each sample size.
#' @param alpha Significance level of test.
#' @param exp_name Name of experiment.
#' @return None.
power_function <- function(sample_sizes, powers, alpha, exp_name) {
  plot(sample_sizes,
       powers,
       type = "l",
       family = "serif",
       ylab = "Power",
       xlab = "n",
       ylim = c(0, 1),
       main = paste("Power as a function of sample size:", exp_name))
  abline(h = alpha, lty = 2, col = "red")
  abline(h = 1, lty = 2, col = "blue")
  dev.copy(png, paste(exp_name, '.png'))
  dev.off()
}

#' Run experiment and estimate power with specified parameters.
#' 
#' @param Sigma 2 x 2 covariance matrix for noise.
#' @param Phis A list of 2 by 2 matrices, representing Phi coeffs.
#' @param Thetas A list of 2 by 2 matrices, representing Theta coeffs.
#' @param M Maximum lag to consider for cross-dCov.
#' @param num_sims Number of simulations to estimate power.
#' @param sample_sizes Vector of sample sizes at which to estimate power.
#' @param num_boot Number of bootstrapped samples.
#' @param exp_name Name of experiment.
#' @param alpha Significance level of test.
#' @param unbiased Whether to use the biased or unbiased estimate of dCov.
#' @param verbose Whether to print
#' @return A vector of power estimates at each of the sample sizes.
run_experiment <- function(Sigma, Phis, Thetas, M, num_sims, sample_sizes, num_boot, exp_name, alpha = 0.05, unbiased = TRUE, verbose = TRUE) {
  
  if (verbose) cat("Experiment:", exp_name, "\n\n")
  
  # Function and argument 'k' is used just for sapply.
  run_test <- function(k) {
    
    if (k %% (num_sims/10) == 0) cat(".")
    Z <- varma(n, Phis, Thetas, Sigma)
    X <- Z[,"X"]
    Y <- Z[,"Y"]
    
    result <- test_independence(X, Y, M, num_boot, alpha, unbiased)
    if (result$reject) return(1) else return(0)
  }
  # Run test num_sims time to estimate power.
  powers <- rep(0, length(sample_sizes))
  for (i in 1:length(sample_sizes)) {
    
    n <- sample_sizes[i]
    if (verbose) cat("Estimating power at sample size", n)
    num_rejects <- sum(sapply(1:num_sims, run_test))
    powers[i] <- num_rejects/num_sims
    if (verbose) cat("\n")
  }
  
  power_function(sample_sizes, powers, alpha, exp_name)
  powers
}
