subtypetable <- read.table("/home/nono/Documents/cnncancer/tcga.subtype.txt",sep="\t",stringsAsFactors=FALSE)

sizelist <- read.table("sizeList.txt",stringsAsFactors=FALSE)

iii <- which(sizelist[,2] > 2e+5 & sizelist[,3]==2048 & sizelist[,4]==2048)
sizelist.filtered <- sizelist[iii,]

subtypeTable <- read.csv("/project/hikaku_db/data/tissue_images/tcga.luad.gene.expression.subtypes.20121025.csv",stringsAsFactors=FALSE)

typelist <- rep("",nrow(sizelist.filtered))
for(aa in 1:nrow(subtypeTable)){
    iii <- grep(subtypeTable[aa,1],sizelist.filtered[,1])
    typelist[iii] <- subtypeTable[aa,2]
}

writeMatrix(cbind(sizelist.filtered[,1],typelist),file="typelist.filterd")
