# # load data
# normalized all together
express.CEU1 <- read.csv('./genevar/CEUp_240_gene_profile.txt', sep='\t')
express.CHB1 <- read.csv('./genevar/CHB_180_gene_profile.txt', sep='\t')
express.JPT1 <- read.csv('./genevar/JPT_180_gene_profile.txt', sep='\t')
express.YRI1 <- read.csv('./genevar/YRIp_240_gene_profile.txt', sep='\t')

# normalized independently for each group
express.CEU2 <- read.csv('./genevar/CEU_parents.csv')
express.CHB2 <- read.csv('./genevar/CHB.csv')
express.JPT2 <- read.csv('./genevar/JPT.csv')
express.YRI2 <- read.csv('./genevar/YRI_parents.csv')

# genotype data
genotype.CEU <- read.csv('./genevar/CEU.hmap', sep=' ')
genotype.CHB <- read.csv('./genevar/CHB.hmap', sep=' ')
genotype.JPT <- read.csv('./genevar/JPT.hmap', sep=' ')
genotype.YRI <- read.csv('./genevar/YRI.hmap', sep=' ')
