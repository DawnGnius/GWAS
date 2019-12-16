## lnorm 
lnorm <- function(a, p = 1) {
  (sum(abs(a) ^ p)) ^ (1. / p)
}

## Vectorization a matrix by row
mstack <- function(S){
  as.vector(t(S))
}


