# data
n <- 1000
n.case <- 500
n.control <- 500
p <- 100000
fst <- 0.01

# p.afaf  ancestral population allele frequency p 
p.afaf <- runif(p, min=0.1, max=0.9)
# p.af    allele frequencies
rbeta1 <- function(p){
  rbeta(1, p*(1-fst)/fst, (1-p)*(1-fst)/fst)
}
p.af <- unlist(lapply(p.afaf, rbeta1))

# population
p1 <- 0.5
p2 <- 0.5

sample.case <- matrix(rep(0, n.case*p))
sample.control <- matrix(rep(0, n.control*p))

# case
sample.case.p1 <- matrix(rep(0, n.case*p*0.6))
sample.case.p2 <- matrix(rep(0, n.case*p*0.4))


# control
sample.control.p1 <- matrix(rep(0, n.control*p*0.4))
sample.control.p2 <- matrix(rep(0, n.control*p*0.6))