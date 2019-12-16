## lnorm 
lnorm <- function(a, p = 1) {
  (sum(abs(a) ^ p)) ^ (1. / p)
}

library(MASS)

# Autoregressive covariance structure
CorrAR <- function(p,rho){  
  Sigma <- matrix(nrow=p,ncol=p,NA)
  for(i in 1:p){
    for(j in 1:p){
      Sigma[i,j] <- rho^(abs(i-j))
    }
  }
  Sigma
}

# Compound symmetry covariance structure
CorrCS <- function(p,rho){  
  Sigma <- matrix(nrow=p,ncol=p,rho)
  diag(Sigma) <- 1
  Sigma
}

#' Simulation model 1
#' 
#' similar to the the simulation model in Chen, Chan, Stenseth (2012), JRSSB
#' 
#' @param n p q model dimensions
#' @param nrank model rank
#' @param s2n signal to noise ratio
#' @param sigma error variance. If specfied, then s2n has no effect
#' @param rho_X correlation parameter in the generation of predictors
#' @param rho_E correlation parameter in the generation of random errors
#' 
#' @return similated model and data
#' 
#' @export
simmodel1 <- function(n=50,p=25,q=25,nrank=3,s2n=1,sigma=NULL,rho_X=0.5,rho_E=0){

  Sigma=CorrAR
  
  U <- matrix(ncol=nrank,nrow=p)
  V <- matrix(ncol=nrank,nrow=q)
  
  U[,1]<-c(sample(c(-1,1),5,replace=T),rep(0,p-5))
  U[,2]<-c(rep(0,3),U[4,1],-U[5,1],sample(c(-1,1),3,replace=T),rep(0,p-8))
  U[,3]<-c(rep(0,8),sample(c(-1,1),2,replace=T),rep(0,p-10))    
  U[,1] <- U[,1]/lnorm(U[,1],2)
  U[,2] <- U[,2]/lnorm(U[,2],2)
  U[,3] <- U[,3]/lnorm(U[,3],2)
  
  #    V[,1]<-c(sample(c(1,-1),5,replace=T)*runif(5,0.3,1),rep(0,q-5))
  #    V[,2]<-c(rep(0,5),sample(c(1,-1),5,replace=T)*runif(5,0.3,1),rep(0,q-10))
  #    V[,3]<-c(rep(0,10),sample(c(1,-1),5,replace=T)*runif(5,0.3,1),rep(0,q-15))  
  #    V[,1] <- V[,1]/lnorm(V[,1],2)
  #    V[,2] <- V[,2]/lnorm(V[,2],2)
  #    V[,3] <- V[,3]/lnorm(V[,3],2)
  #
  #    D <- diag(c(20,15,10))
  
  V[,1]<-c(sample(c(1,-1),5,replace=T)*runif(5,0.5,1),rep(0,q-5))
  V[,2]<-c(rep(0,5),sample(c(1,-1),5,replace=T)*runif(5,0.5,1),rep(0,q-10))
  V[,3]<-c(rep(0,10),sample(c(1,-1),5,replace=T)*runif(5,0.5,1),rep(0,q-15))  
  V[,1] <- V[,1]/lnorm(V[,1],2)
  V[,2] <- V[,2]/lnorm(V[,2],2)
  V[,3] <- V[,3]/lnorm(V[,3],2)
  
  D <- diag(c(20,15,10))
  
  Xsigma <- Sigma(p,rho_X)
  X <- mvrnorm(n,rep(0,p),Xsigma)
  ##X <- diag(p)
  
  UU<-matrix(nrow=n,ncol=q,rnorm(n*q,0,1))
  if(rho_E!=0){
    for(t in 1:n) UU[t,] <- arima.sim(list(order=c(1,0,0), ar=rho_E), n=q)
  }
  
  C <- U%*%D%*%t(V)
  C3 <- U[,3]%*%t(V[,3])*D[3,3]
  
  Y3 <- X%*%C3  
  ##sigma <- sqrt(var(as.numeric(Y3))/var(as.numeric(UU))/s2n)  
  ##the same
  if(is.null(sigma)){
    sigma <- sqrt(sum(as.numeric(Y3)^2)/sum(as.numeric(UU)^2)/s2n)
  }
  UU <- UU*sigma
  
  Y <- matrix(nrow=n,ncol=q,NA)
  Y <- X%*%C + UU
  
  list(Y=Y,X=X,C=C,U=U,V=V,D=D,Xsigma=Xsigma)
  
}



#' Simulation model 2
#' 
#' similar to the the simulation model in Chen and Huang (2012), JASA
#' 
#' @param n sample size
#' @param p number of predictors
#' @param p0 number of relevant predictors 
#' @param q number of responses
#' @param q0 number of relevant responses
#' @param nrank model rank
#' @param s2n signal to noise ratio
#' @param sigma error variance. If specfied, then s2n has no effect
#' @param rho_X correlation parameter in the generation of predictors
#' @param rho_E correlation parameter in the generation of random errors
#' 
#' @return similated model and data
#' 
#' @export
simmodel2 <- function(n=100,p=50,p0=10,q=50,q0=10,nrank=3,s2n=1,sigma=NULL,rho_X=0.5,rho_E=0){

  Sigma=CorrCS
  
  A1 <- matrix(ncol=nrank,nrow=q0,rnorm(q0*nrank))
  A0 <- matrix(ncol=nrank,nrow=q-q0,0)
  A <- rbind(A1,A0)         
  B1 <- matrix(ncol=nrank,nrow=p0,rnorm(p0*nrank))
  B0 <- matrix(ncol=nrank,nrow=p-p0,0)
  B <- rbind(B1,B0)
  C <- B%*%t(A)   
  
  Xsigma <- Sigma(p,rho_X) 
  X <- mvrnorm(n,rep(0,p),Xsigma)
  
  UU <- mvrnorm(n,rep(0,q),Sigma(q,rho_E))
  
  #    ###Their definition is tr(CXC)/tr(E), which seems to be wrong
  #    sigma <- sqrt(sum(diag(t(C)%*%Sigma(p,rho_X)%*%C))/sum(diag(t(UU)%*%UU))/s2n)
  #    UU <- UU*sigma
  
  svdC <- svd(C)
  C3 <- svdC$u[,nrank]%*%t(svdC$v[,nrank])*svdC$d[nrank]
  Y3 <- X%*%C3  
  ##sigma <- sqrt(var(as.numeric(Y3))/var(as.numeric(UU))/s2n)  
  ##the same
  if(is.null(sigma)){
    sigma <- sqrt(sum(as.numeric(Y3)^2)/sum(as.numeric(UU)^2)/s2n)      
  }
  UU <- UU*sigma
  
  Y <- matrix(nrow=n,ncol=q,NA)
  Y <- X%*%C + UU
  
  list(Y=Y,X=X,C=C,A=A,B=B,U=B,V=A,sigma=sigma,Xsigma=Xsigma)
  
}



