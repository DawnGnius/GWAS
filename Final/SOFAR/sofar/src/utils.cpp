#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;


// [[Rcpp::export]]
arma::mat kron_RcppArma(arma::mat A, arma::mat B) {
    return(arma::kron(A, B));
}


