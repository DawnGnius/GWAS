# # load data
load('./genevar/dat.RData')

# data preparation
CCT8.CEU2 <- express.CEU2[which(express.CEU2[,1] == "GI_6005726-S"), -1]
CCT8.CHB2 <- express.CHB2[which(express.CHB2[,1] == "GI_6005726-S"), -1]
CCT8.JPT2 <- express.JPT2[which(express.JPT2[,1] == "GI_6005726-S"), -1]
CCT8.YRI2 <- express.YRI2[which(express.YRI2[,1] == "GI_6005726-S"), -1]


#############

tmp1 <- colnames(CCT8.CEU2)
tmp2 <- colnames(genotype.CEU)
a <- unlist(lapply(tmp1, FUN=function(ii) which(tmp2 == ii)))
genotype.CEU2 <- genotype.CEU[, a]


tmp1 <- colnames(CCT8.CHB2)
tmp2 <- colnames(genotype.CHB)
a <- unlist(lapply(tmp1, FUN=function(ii) which(tmp2 == ii)))
genotype.CHB2 <- genotype.CHB[, a]

tmp1 <- colnames(CCT8.JPT2)
tmp2 <- colnames(genotype.JPT)
a <- unlist(lapply(tmp1, FUN=function(ii) which(tmp2 == ii)))
genotype.JPT2 <- genotype.JPT[, a]

tmp1 <- colnames(CCT8.YRI2)
tmp2 <- colnames(genotype.YRI)
a <- unlist(lapply(tmp1, FUN=function(ii) which(tmp2 == ii)))
genotype.YRI2 <- genotype.YRI[, a]


############

tmp1 <- colnames(CCT8.CEU2)
tmp2 <- colnames(genotype.CEU)
a <- unlist(lapply(tmp2, FUN=function(ii) which(tmp1 == ii)))
CCT8.CEU3 <- CCT8.CEU2[, a]

tmp1 <- colnames(CCT8.CHB2)
tmp2 <- colnames(genotype.CHB)
a <- unlist(lapply(tmp2, FUN=function(ii) which(tmp1 == ii)))
CCT8.CHB3 <- CCT8.CHB2[, a]

tmp1 <- colnames(CCT8.JPT2)
tmp2 <- colnames(genotype.JPT)
a <- unlist(lapply(tmp2, FUN=function(ii) which(tmp1 == ii)))
CCT8.JPT3 <- CCT8.JPT2[, a]

tmp1 <- colnames(CCT8.YRI2)
tmp2 <- colnames(genotype.YRI)
a <- unlist(lapply(tmp2, FUN=function(ii) which(tmp1 == ii)))
CCT8.YRI3 <- CCT8.YRI2[, a]