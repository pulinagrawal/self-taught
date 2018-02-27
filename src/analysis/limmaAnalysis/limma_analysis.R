library("limma")

limmaAnalysis <- function(dataMatrix, designMatrix){
  lmResult <- limma::lmFit(dataMatrix, designMatrix)
  eBayesResult <- limma:eBayes(lmResult)
  output <- topTable(eBayesResult)
  return(output)
}