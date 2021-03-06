% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rrr.R
\name{rrr.select}
\alias{rrr.select}
\title{Select an rrr model from a solution path}
\usage{
rrr.select(fit, Y, X, method = c("AIC", "BIC", "BICP", "GCV", "GIC"),
  df.type = c("exact", "naive"))
}
\arguments{
\item{fit}{a fitted object from \code{rrr.path}}

\item{Y}{response matrix}

\item{X}{covariate matrix}

\item{method}{the information criterion to be used;
currently supporting `AIC', `BIC', `BICP', `GCV', and `GIC'}

\item{df.type}{the type of degrees of freedom, supporting
`exact' and `naive', as returned from \code{rrr.path}}
}
\value{
fitted object from the best model
}
\description{
Select the best rrr model from a solution path fitted with \code{rrr.path}, 
according to an information criterion.
}

