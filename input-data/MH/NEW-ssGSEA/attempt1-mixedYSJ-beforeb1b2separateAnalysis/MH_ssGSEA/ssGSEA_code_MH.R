#source("https://bioconductor.org/biocLite.R")
#biocLite("GSVA")
#BiocManager::install("GSVA")
library(GSVA)

# Before importing, need to add NAs to make lists equal length for read.table
GeneSet <- read.table("ssGSEA_genesets_combined.txt", header=TRUE)
GeneSet <- as.list(GeneSet)
GeneSet <- lapply(GeneSet, function(x) x[!is.na(x)]) # remove NAs

ExprSet <- read.table("MyBrCa_Batch1_and_2_counts_voomed_copy.txt", sep="\t", header=TRUE)
ExprMat <- as.matrix(ExprSet[,-1])
rownames(ExprMat)<-ExprSet[,1]

#generate ssGSEAparam for ssGSEA method 
ssGSEA_param <- ssgseaParam(ExprMat, GeneSet)
ssGSEA_out <- gsva(ssGSEA_param)
ssGSEA_res_df <- t(ssGSEA_out)
ssGSEA_res_df <- as.data.frame(ssGSEA_res_df)
ssGSEA_res_df <- format(round(ssGSEA_res_df, 4), nsmall = 4)
ssGSEA_res_df <- cbind(row.names(ssGSEA_res_df),ssGSEA_res_df)
colnames(ssGSEA_res_df)[1]<-c("ID")
write.table(ssGSEA_res_df, file='MyBRCA_ssGSEA_output_combinedGeneSet_MH.txt', quote=FALSE, sep='\t', row.names = F)