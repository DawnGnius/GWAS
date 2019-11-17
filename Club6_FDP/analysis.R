library(GenABEL)

idx <- which(express.Asian[,2] == 'GI_10047089-S')# find rows containning given SNP
express.Asian.snp <- express.Asian[idx, ]

GroupID.Asian <- colnames(Asian)
for (ii in 1:length(GroupID.Asian)) {
  # loop for 90/60/45 samples
  id.tmp11 <- paste(GroupID.Asian[ii], '_1_1', sep="")
  id.tmp12 <- paste(GroupID.Asian[ii], '_1_2', sep="")
  id.tmp21 <- paste(GroupID.Asian[ii], '_2_1', sep="")
  id.tmp22 <- paste(GroupID.Asian[ii], '_2_2', sep="")
  
  detection.id.tmp11 <- express.Asian.snp[which(express.Asian.snp[,1] == id.tmp11), 6]
  detection.id.tmp12 <- express.Asian.snp[which(express.Asian.snp[,1] == id.tmp12), 6]
  detection.id.tmp21 <- express.Asian.snp[which(express.Asian.snp[,1] == id.tmp21), 6]
  detection.id.tmp22 <- express.Asian.snp[which(express.Asian.snp[,1] == id.tmp22), 6]
  
  tmp <- which.max(c(detection.id.tmp11, detection.id.tmp12, detection.id.tmp21, detection.id.tmp22))
  if (tmp == 1){
    
  }
}