#' Fitting reduced rank regression
#'
#' Given a response matrix and a covariate matrix, fits reduced rank
#' regression for a specified rank. It can also be used to do singular
#' value decomposition if the covariate matrix is the identity matrix.1
#' 
#' @param Y a matrix of response (n by q)
#' @param X a matrix of covariate (n by p)
#' @param nrank an integer specifying the desired rank
#' @param weight a square matrix of weight (q by q);
#'   The default is the identity matrix
#' @param coefSVD logical indicating the need for SVD for the
#'   coeffient matrix in the output; used in ssvd estimation
#' @return S3 \code{rrr} object, a list consisting of
#'   \item{coef}{coefficient of rrr}
#'   \item{coef.ls}{coefficient of least square}
#'   \item{fitted}{fitted value of rrr}
#'   \item{fitted.ls}{fitted value of least square}
#'   \item{A}{right singular matrix}
#'   \item{Ad}{a vector of sigular values}
#'   \item{nrank}{rank of the fitted rrr}
#' @examples
#' Y <- matrix(rnorm(400), 100, 4)
#' X <- matrix(rnorm(800), 100, 8)
#' rfit <- rrr.fit(Y, X)
#' @export
rrr.fit <- function(Y, X, nrank=1,
                    weight = NULL,
                    coefSVD = FALSE
) {
  q <- ncol(Y)
  n <- nrow(Y)
  p <- ncol(X)
  stopifnot(n == nrow(X))
  
  S_yx <- crossprod(Y, X)
  S_xx <- crossprod(X)
  
  ## FIXME: 0.1 is too arbitrary
  S_xx_inv <- tryCatch(ginv(S_xx),
                       error = function(e) solve(S_xx + 0.1 * diag(p)))
  
  ## FIXME: if weighted, this needs to be weighted too
  C_ls <- tcrossprod(S_xx_inv, S_yx)
  
  if (!is.null(weight)) {
    stopifnot(nrow(weight) == q && ncol(weight) == q)
    eigenGm <- eigen(weight) ## FIXME: ensure eigen success?
    ## sqrtGm <- tcrossprod(eigenGm$vectors * sqrt(eigenGm$values),
    ##                      eigenGm$vectors)
    ## sqrtinvGm <- tcrossprod(eigenGm$vectors / sqrt(eigenGm$values),
    ##                         eigenGm$vectors)
    sqrtGm <- eigenGm$vectors %*% (sqrt(eigenGm$values) * t(eigenGm$vectors))
    sqrtinvGm <- eigenGm$vectors %*% (1/sqrt(eigenGm$values)* t(eigenGm$vectors))
    
    XC <- X %*% C_ls %*% sqrtGm
    ## FIXME: SVD may not converge
    ## svdXC <- tryCatch(svd(XC,nu=nrank,nv=nrank),error=function(e)2)
    svdXC <- svd(XC, nrank, nrank)
    A <- svdXC$v[, 1:nrank]
    Ad <- (svdXC$d[1:nrank])^2
    AA <- tcrossprod(A)
    C_rr <-C_ls %*% sqrtGm %*% AA %*% sqrtinvGm
  } else { ## unweighted 
    XC <- X %*% C_ls
    svdXC <- svd(XC, nrank, nrank)
    A <- svdXC$v[, 1:nrank]
    Ad <- (svdXC$d[1:nrank])^2
    AA <- tcrossprod(A)
    C_rr <-C_ls %*% AA
  }
  
  ret <- list(coef = C_rr,
              coef.ls = C_ls,
              fitted = X %*% C_rr,
              fitted.ls = XC,
              A = A, Ad = Ad, nrank = nrank)
  if (coefSVD) {
    coefSVD <- svd(C_rr, nrank, nrank)
    coefSVD$d <- coefSVD$d[1:nrank]
    coefSVD$u <- coefSVD$u[, 1:nrank, drop = FALSE]
    coefSVD$v <- coefSVD$v[, 1:nrank, drop = FALSE]
    ret <- c(ret, list(coefSVD = coefSVD))
  }
  class(ret) <- "rrr"
  invisible(ret)
}



rrr.control <- function(sv.tol = 1e-7,
                        qr.tol = 1e-7) {
  list(  sv.tol = sv.tol          ## singular value tolerence
         , qr.tol = qr.tol        ## QR decomposition tolerence
  )
}

#' Solution path for reduced rank estimators
#' @param Y a matrix of response (n by q)
#' @param X a matrix of covariate (n by p)
#' @param penaltySVD the penalty to be used on the SVD:
#'   `count' means rank-constrainted estimation;
#'   `soft' means soft singular value thresholding;
#'   `ann' means adaptive soft singular value thresholding.
#' @param maxrank an integer of maximum desired rank.
#' @param gamma a scalar power parameter of the adaptive weights in penalty == "ann"
#' @param nlambda the number of lambda values; no effect if
#'   penalty == `count'.
#' @param lambda a vector of user-specified lambda values.
#' @param control a list of parameters for controlling the fitting process:
#'   `sv.tol' controls the tolerence of singular values;
#'   `qr.tol' controls the tolerence of QR decomposition for the LS fit
#' @return S3 \code{rrr.path} object, a list consisting of
#'   \item{A}{right singular matrix}
#'   \item{Ad}{a vector of singular values}
#'   \item{coef.ls}{coefficient estimate from LS}
#'   \item{Spath}{a matrix, each column containing shrinkage factors of the singular values of a solution}
#'   \item{df.exact}{the exact degrees of freedom}
#'   \item{df.naive}{the naive degrees of freedom as the effective number of parameters}
#'   \item{penaltySVD}{the method of low-rank estimation}
#' @examples
#' Y <- matrix(rnorm(400), 100, 4)
#' X <- matrix(rnorm(800), 100, 8)
#' rpath <- rrr.path(Y, X, maxrank = 3)
#' @export
rrr.path <- function(Y, X, 
                     penaltySVD = c('count', 'soft', 'ann'),
                     maxrank = min(dim(Y), dim(X)),                    
                     gamma = 2,     ## for 'ann'
                     nlambda = 100, ## no effect if 'count' 
                     lambda = NULL,
                     control = list() ) {
  q <- ncol(Y)
  n <- nrow(Y)
  p <- ncol(X)
  stopifnot(n == nrow(X))
  control <- do.call("rrr.control", control)
  
  ## Obtain the LS estimate
  qrX <- qr(X, tol = control$qr.tol)
  C_ls <- qr.coef(qrX, Y)
  C_ls <- ifelse(is.na(C_ls), 0, C_ls) ## FIXME
  rX <- qrX$rank
  
  nrank <- min(q, rX, maxrank)  ## nrank is the user specified upper bound
  rmax <- min(rX, q)            ## rmax is the maximum rank possible
  
  XC <- qr.fitted(qrX, Y)       ## X %*% C_ls
  svdXC <- svd(XC, nu = rmax, nv = rmax)
  A <- svdXC$v[, 1:rmax]
  Ad <- (svdXC$d[1:rmax])^2
  
  ## FIXME: Is the following necessary
  Ad <- Ad[Ad > control$sv.tol]
  rmax <- length(Ad)            ## modify rmax to be more practical
  
  
  ## f.d: weight function based on given thresholding rule
  ## f.d.derv: derivative of f.
  ## lambda: sequence of lambda values
  penaltySVD <- match.arg(penaltySVD)
  if (penaltySVD == "soft") {
    Adx <- Ad ^ 0.5
    if (is.null(lambda)) {
      lambda <- exp(seq(log(max(Adx)), log(Adx[min(nrank + 1, rmax)]),
                        length = nlambda))
    }
    f.d <- function(lambda, Ad){
      Adx <- Ad ^ (1/2)
      softTH(Adx, lambda) / Adx    
    }
    f.d.derv <- function(lambda, Ad) lambda / Ad 
  }
  
  if (penaltySVD == "ann"){
    Adx <- Ad ^ ((1 + gamma) / 2 )
    if (is.null(lambda)){
      lambda <- exp(seq(log(max(Adx)), log(Adx[min(nrank+1, rmax)]),
                        length = nlambda))        
    }
    f.d <- function(lambda, Ad) {
      Adx <- (Ad) ^ ((1 + gamma) / 2)
      softTH(Adx, lambda) / Adx    
    }
    f.d.derv <- function(lambda,Ad) lambda*(gamma+1)*Ad^((-gamma-2)/2)
  }
  
  if (penaltySVD =="count"){
    if (is.null(lambda)) {
      lambda <- 1:nrank  
    }
    f.d <- function(lambda,Ad) rep(1,lambda)
    f.d.derv <- function(lambda,Ad)rep(0,lambda)
  }
  
  nlam <- length(lambda)  
  
  ## Shrinkage factor
  Spath <- matrix(0, nrank, nlam)
  
  df.exact <- df.naive <- rep(0., nlam)
  for (i in 1:nlam){
    f <- f.d(lambda[i], Ad[1:nrank])
    Spath[1:length(f), i] <- f
    ##Spath[, i] <- f
    r <- sum(f > control$sv.tol)
    if (r >= 1) {
      f.derv <- f.d.derv(lambda[i], Ad[1:r])    
      term1 <- max(rX, q) * sum(f)                
      
      a <- vector()
      count = 1
      for(k in 1:r){  
        for(s in (r+1):rmax){
          a[count] <- (Ad[k]+Ad[s])*f[k]/(Ad[k]-Ad[s])
          count <- count +1
        }
      }    
      term2 <- sum(a)
      #h <- length(a)
      ##if(r==maxrank) term2 <- sum(a[-c(h-1,h)])
      if(r==rmax) term2 <- 0
      if(r==rmax & r==min(p,q)) term2 <- 0
      
      b <- vector()
      count = 1
      for(k in 1:r){      
        for(s in 1:r){
          if(s==k){
            b[count] <- 0
          }else{
            b[count] <- Ad[k]*(f[k]-f[s])/(Ad[k]-Ad[s])
          }
          count <- count +1
        }
      }
      term3 <- sum(b)    
      
      c <- Ad[1:r] ^ (1/2) * f.derv
      term4 <- sum(c)
      
      df.exact[i] <- term1 + term2 + term3 + term4
      df.naive[i] <- r*(rX+q-r)
    }
  }
  
  
  #   ###Shrinkage factor
  #   Spath <- matrix(nrow=nrank,ncol=nlam,0)
  #   
  #   df.exact <- vector()
  #   df.naive <- vector()
  #   for(i in 1:nlam){
  #     
  #     f <- f.d(lambda[i],Ad[1:nrank])
  #     Spath[1:length(f),i] <- f
  #     ##f <- c(f,rep(0,min(p,q)-length(f)))
  #     r <- sum(f>control$tol)
  #     if(r>=1){
  #       f.derv <- f.d.derv(lambda[i],Ad[1:r])    
  #       
  #       term1 <- max(rX,q)*sum(f)                
  #       
  #       a <- vector()
  #       count = 1
  #       for(k in 1:r){  
  #         for(s in (r+1):rmax){
  #           a[count] <- (Ad[k]+Ad[s])*f[k]/(Ad[k]-Ad[s])
  #           count <- count +1
  #         }
  #       }    
  #       term2 <- sum(a)
  #       #h <- length(a)
  #       ##if(r==maxrank) term2 <- sum(a[-c(h-1,h)])
  #       if(r==rmax) term2 <- 0
  #       if(r==rmax & r==min(p,q)) term2 <- 0
  #       
  #       b <- vector()
  #       count = 1
  #       for(k in 1:r){      
  #         for(s in 1:r){
  #           if(s==k){
  #             b[count] <- 0
  #           }else{
  #             b[count] <- Ad[k]*(f[k]-f[s])/(Ad[k]-Ad[s])
  #           }
  #           count <- count +1
  #         }
  #       }
  #       term3 <- sum(b)    
  #       
  #       #     c <- vector()
  #       #     for(k in 1:r){
  #       #       c[k] <- Ad[k]^(1/2)*f.derv[k]
  #       #       ##c[k] <- 2*lambda[i]*sqrt(Ad[k])^(-gamma)          
  #       #     }
  #       c <- Ad[1:r]^(1/2)*f.derv
  #       term4 <- sum(c)
  #       
  #       df.exact[i] <- term1 + term2 + term3 + term4
  #       df.naive[i] <- r*(rX+q-r)
  #       #############################################################  
  #     }else{
  #       df.exact[i] <- 0
  #       df.naive[i] <- 0      
  #     }
  #     
  #     ##df.exact[1] <- 0
  #     ##df.naive[1] <- 0
  #     
  #   }
  #   
  
  rval <- list(A=A, Ad=Ad, 
               coef.ls = C_ls, 
               Spath = Spath, ## singular value after thresholding
               df.exact = df.exact, 
               df.naive = df.naive,
               penaltySVD = penaltySVD)
  class(rval) <- "rrr.path"
  invisible(rval)
}


#' Select an rrr model from a solution path
#'
#' Select the best rrr model from a solution path fitted with \code{rrr.path}, 
#' according to an information criterion.
#'
#' @param fit a fitted object from \code{rrr.path}
#' @param Y response matrix
#' @param X covariate matrix
#' @param method the information criterion to be used;
#'   currently supporting `AIC', `BIC', `BICP', `GCV', and `GIC'
#' @param df.type the type of degrees of freedom, supporting
#'   `exact' and `naive', as returned from \code{rrr.path}
#' @return fitted object from the best model
#' @export
rrr.select <- function(fit, Y, X, 
                       method = c("AIC","BIC","BICP","GCV","GIC"), 
                       df.type = c("exact","naive")) {
  
  q <- ncol(Y)
  n <- nrow(Y)
  p <- ncol(X)
  method <- match.arg(method)
  df.type <- match.arg(df.type)
  
  C_ls <- fit$coef.ls
  A <- fit$A
  tempFit <- X %*% C_ls 
  
  nlam <- ncol(fit$Spath)
  rankall <-  sse <- rep(0., nlam)
  for(i in 1:nlam) {    
    dsth <- fit$Spath[,i]
    rank <- sum(dsth!=0)
    rankall[i] <- rank
    if (rank != 0){
      ## tempC <- tcrossprod(A[,1:rank] * dsth[1:rank], A[,1:rank])
      tempC <- A[,1:rank] %*% (dsth[1:rank] * t(A[,1:rank]))
      tempYhat <- tempFit %*% tempC
      sse[i] <- sum((Y - tempYhat)^2)
    } else {
      sse[i] <- sum(Y^2)
    }
  }
  
  logsse <- log(sse)
  df <- switch(df.type,
               "exact" = fit$df.exact,
               "naive" = fit$df.naive)
  ic <- switch(method,
               "GCV" = n*q*sse/(n*q-df)^2,
               "AIC" = n*q*log(sse/n/q) + 2*df,
               "BIC" = n*q*log(sse/n/q) + log(q*n)*df,
               "BICP" = n*q*log(sse/n/q)  + 2*log(p*q)*df,
               "GIC" = n*q*log(sse/n/q) + log(log(n*q))*log(p*q)*df)
  
  
  ##min.id <- which.min(ic[-c(which(sse < 1e-10)[1]:length(ic))])   
  min.id <- which.min(ic)
  rankest <- rankall[min.id]
  
  
  dsth <- fit$Spath[,min.id]
  ##dsth[dsth < tol] <- 0
  ##rank1 <- sum(round(dsth,3)!=0)
  if (rankest != 0){
    C <- C_ls%*%A[,1:rankest]%*%diag(dsth[1:rankest],nrow=rankest,ncol=rankest)%*%t(A[,1:rankest])
  }else{
    C <- matrix(nrow=p,ncol=q,0)
  }
  
  list(sse = sse,
       ic = ic,
       coef = C,
       s = dsth,
       rank = rankest)
}
