# load data
CHB <- read.csv('./Club6_FDP/genevar/CHB.csv')
CEU <- read.csv('./Club6_FDP/genevar/CEU_parents.csv')
JPT <- read.csv('./Club6_FDP/genevar/JPT.csv')
YRI <- read.csv('./Club6_FDP/genevar/YRI_parents.csv')
# microarray <- read.csv('./Club6_FDP/genevar/illumina_Human_WG-6_array_content.csv')
express.CHB <- read.csv('./Club6_FDP/genevar/CHB_180_gene_profile.txt', sep='\t')
express.CEU <- read.csv('./Club6_FDP/genevar/CEUp_240_gene_profile.txt', sep='\t')
express.JPT <- read.csv('./Club6_FDP/genevar/JPT_180_gene_profile.txt', sep='\t')
express.YRI <- read.csv('./Club6_FDP/genevar/YRIp_240_gene_profile.txt', sep='\t')

# data preparation
CHB <- CHB[, -1]; CEU <- CEU[, -1]; JPT <- JPT[, -1]; YRI <- YRI[, -1]
Asian <- cbind(CHB, JPT)
express.Asian <- rbind(express.CHB, express.JPT)
