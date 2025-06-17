#library(ggh4x)
library(cowplot)
library(gtable)
library(ggpubr)
#library(ggtext)
#library(entropy)
#library(rstatix)
#library(tidytext)
#library(dendsort)
library(circlize)
library(seriation) 
library(tidyverse)
library(ComplexHeatmap)


theme_ch <- function(fpt = 6) {
  theme_pubr() + 
    theme(
      axis.title = element_text(size = fpt),
      axis.text = element_text(size = fpt),
      plot.background = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(size = fpt),
      plot.title = element_text(size=fpt, hjust = 0.5),
      legend.text = element_text(size = fpt), 
      legend.title = element_text(size = fpt),
      axis.line = element_line(colour = 'black', size = 0.3),
      axis.ticks = element_line(colour = "black", size = 0.3),
      legend.position ='right'
    )
}

ht_opt(
  heatmap_row_names_gp = gpar(fontsize = 6),
  heatmap_column_names_gp = gpar(fontsize = 6),
  legend_grid_width = unit(3,'mm'), 
  legend_grid_height = unit(1, 'mm'),
  legend_title_position = "topleft",
  heatmap_row_title_gp = gpar(fontsize=6)
)

# Organ colours
lut = structure(c('#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#000000', '#505f37', 'yellow'),
                names = c('Breast', 'Colorectal', 'Bone_SoftTissue', 'Kidney', 'Lung', 'Prostate', 'Uterus', 'CNS', 'Ovary', 'Skin', 'Bladder', 'Pancreas', 'Liver', 'Lymphoid', 'Stomach', 'Esophagus', 'NET', 'Oral_Oropharyngeal', 'Biliary', 'Head_Neck', 'Thyroid', 'Cervix','Myeloid'))

subst_col <- c('C>A'='#02bced', 'C>G'='#010101', 'C>T'='#e22926', 'T>A'='#cac8c9', 'T>C'='#a0ce62', 'T>G'='#ecc6c5')

activity_col <-  c('Damage' = 'maroon3', 
                   'Misrepair' = 'aquamarine3', 
                   'Connect' = 'darkgoldenrod3',
                   'W' = 'darkgoldenrod4',
                   'COSMIC'= 'dodgerblue')
     

cohort_col <- c('PCAWG' = '#1D5E9E', 
                'GEL' = '#B83966',
                'HMF' = '#96B301',
                'ICGC' = '#1D5E9E')

connect_type_col <- c(
      "One-one" = "#187161",    # dark teal
      "Many-one" = "#E78540",   # dark orange
      "One-many" = "#1E7EC2",   # dark blue
      "Many-many" = "#B3415F"   # dark pink
    )

etiology_col <- c(
        "Clock-like" = "#66C2A5",       # teal
        "APOBEC" = "#FC8D62",      # salmon
        "DDRD" = "#8DA0CB",        # blue
        "Environmental" = "#E78AC3", # pink
        "ROS" = "#A6D854",         # green
        "Treatment" = "#FFD92F",    # yellow
        "Unknown" = "#E5C494",      # tan
        "Artifact" = "#B3B3B3"      # gray
      )
      
feature_set_col = c('DAMUTA' = 'coral', 'COSMIC' = 'dodgerblue', 'TMB' = 'grey')
sig_group_col = c('Both' = 'coral', 'Damage' = 'maroon3', 'Misrepair' = 'aquamarine4', 'None' = 'grey80')

#subst_col <- c('C>A'='#02bced', 'C>G'='#010101', 'C>T'='#e22926', 'T>A'='#cac8c9', 'T>C'='#a0ce62', 'T>G'='#ecc6c5')

apply_vegdist <- function(x,y,...) {
  vegdist(rbind(x,y), ...)[1]
}