library(Biostrings)

dt_to_cosmic <- function(df){
    # restyle a set of mutation types from deeptumour style to cosmic

    # take only last 96 columns 
    df <- df[,55:150]

    # reverse complement purine contexts
    purines <- substr(names(df),2,2) %in% c("A", "G")
    ns <- lapply(strsplit(names(df), split = "..", fixed = T), DNAStringSet)
    ns[purines] <- lapply(ns[purines], reverseComplement)
    ns <- lapply(ns, as.character)

    # construct the mutation type from trinucs
    # use map to extract from desired list (column-wise)
    pos1 <- lapply(purrr::map(ns,1), substr, start=1, stop=1)
    ref <- lapply(purrr::map(ns,1), substr, start=2, stop=2)
    alt <- lapply(purrr::map(ns,2), substr, start=2, stop=2)
    pos3 <- lapply(purrr::map(ns,1), substr, start=3, stop=3)

    ns <- paste0(pos1, "[", ref, ">", alt, "]", pos3)
    names(df) <- ns
    return(df)
}

# PCAWG counts
pcawg <- read.csv('pcawg_mutation_types_raw_counts.csv', row.names =1)
pcawg <- pcawg[1:2778,]
pcawg <- dt_to_cosmic(pcawg)
write.csv(file ='pcawg_counts.csv', pcawg, quote=F)

# Hartwig counts
load('hmf.mut-type.RData')
hw <- dt_to_cosmic(mut.df.ct)
#rownames(hw) <- paste0('hw_', rownames(hw))
write.csv(file ='hartwig_counts.csv', hw, quote=F)
