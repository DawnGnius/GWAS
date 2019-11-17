p <- 100
n <- 100
sig <- 2
p.nonzero <- 10
beta.nonzero <- 1

# Equal correlation
beta <- c(rep(beta.nonzero, p.nonzero), rep(0, p-p.nonzero))
Sigma <- matrix(rep(.5, p*p), p, p); diag(Sigma) <- rep(1, p)
dat <- MASS::mvrnorm(n, rep(0,p), Sigma)

mu <- unlist(base::lapply(X=1:p, FUN=function(ii) sqrt(n)*beta[ii]*sqrt(var(dat[, ii]))/sig))

# Z <- MASS::mvrnorm(1, mu, Sigma)

# FDP and FDP_lim
t <- 0.01
fdp <- function(ii){
  Z <- MASS::mvrnorm(1, mu, Sigma)
  pvalue <- unlist(base::lapply(X=1:p, FUN=function(ii) 1-pnorm(abs(Z[ii]))))
  tmp.pvalue <- pvalue[11:p]
  sum(which(tmp.pvalue > t)) / sum(which(pvalue > t))
}

snowfall::sfInit(parallel = TRUE, cpus = 10) # init
snowfall::sfLibrary(MASS)
snowfall::sfExport("p", "mu", "Sigma", "t")
fdp.repeat <- snowfall::sfLapply(1:500, fdp)
print(fdp.repeat)