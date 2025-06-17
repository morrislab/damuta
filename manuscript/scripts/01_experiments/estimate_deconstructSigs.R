library(deconstructSigs)
library(readr)
library(dplyr)
library(tidyr)

library(flock)


args <- commandArgs(trailingOnly=TRUE)

sigs_file <- args[1]
counts_file <- args[2]
id <- args[3]
out_file <- args[4]

sigs_lock <- lock(sigs_file)
counts_lock <- lock(counts_file)


mut96 <- colnames(signatures.cosmic)
cosmic_v3 <- read_tsv(sigs_file, show_col_types = FALSE) %>%
    pivot_longer(cols = -Type) %>% 
    pivot_wider(names_from = Type) %>%
    as.data.frame() %>%
    tibble::column_to_rownames('name')
cosmic_v3 <- cosmic_v3[mut96]
counts <- read_csv(counts_file, show_col_types = FALSE)
ids <- counts[["...1"]]
counts <- counts[mut96]
counts <- as.data.frame(counts)
rownames(counts) <- ids

unlock(sigs_lock)
unlock(counts_lock)


print(paste0('estimating activities on ', id))
y <- deconstructSigs::whichSignatures(counts, id, signatures.ref = cosmic_v3, contexts.needed=T)$weights %>%
    tibble::rownames_to_column() 

out_lock <- lock(out_file)
write_tsv(y, out_file, append = TRUE)
unlock(out_lock)

