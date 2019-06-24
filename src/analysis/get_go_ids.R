library(GO.db)
library(hash)

goterm=as.list(GOTERM)

get_terms=function(){
terms_dict = hash()
for(goid in goterm){
  term = Term(GOID(goid))[[GOID(goid)]]
  term = gsub("[^[:alnum:] ]", " ", term)
  terms_dict[[ paste("GO",chartr("-, ","___",toupper(term)), sep="_") ]] = GOID(goid)
  }
return(terms_dict)
}

prnt=function(x,y){
  for(i in seq(x,y)){
    term = Term(GOID(goterm[[i]]))[[GOID(goterm[[i]])]]
    term = gsub("[^[:alnum:] ]", " ", term)
    print(paste("GO",chartr(" ","_",toupper(term)), sep="_"))
  }
}

term_hash = get_terms()
# go_terms files is obtained by running the following bash command
# cat c5.bp.v6.1.entrez.gmt 
x <- scan("go_terms.txt", what="", sep="\n")

iterations = length(x)
variables = 2

output <- matrix(ncol=variables, nrow=iterations)

for(i in 1:iterations){
  go_id <- term_hash[[x[i]]]
  if (!is.null(go_id)){
    output[i,] <- c(x[i], go_id)
  }
  
  
}
output <- data.frame(output)

write.table(output, file="go_term_map.txt", row.names=F, col.names=F, sep=",")
