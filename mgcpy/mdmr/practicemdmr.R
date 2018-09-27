library(MDMR)
vignette("mdmr-vignette")

# Load data
data(mdmrdata)

# Compute distance matrix
D <- dist(Y.mdmr, method = "euclidean")

# Conduct MDMR
mdmr.res <- mdmr(X = X.mdmr, D = D)

# Check results
summary(mdmr.res)

# Study univariate effect sizes
delta.res <- delta(X.mdmr, Y = Y.mdmr, dtype = "euclidean", 
                   niter = 10, plot.res = T)