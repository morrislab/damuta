source('plotting_header.R')

ann <- read_csv('data/phg_clinical_ann.csv') %>% rename(guid=`...1`) %>%
  mutate(cohort = factor(cohort, levels = c("GEL", "HMF", "PCAWG", "ICGC")))

cohort_samples <- ann %>% 
  group_by(cohort, organ) %>%
  summarize(n=n()) %>%
  ggplot(aes(x = reorder(organ, n), y=n, fill = cohort)) +
  geom_col(linewidth = 0.9) + 
  coord_flip() + 
  labs(y = 'Number of samples', x = 'Organ', fill= 'Dataset') +  
  theme_ch()  + theme(legend.position = c(0.8, 0.4), legend.key.size = unit(6,'pt'))

organs <- ann %>% 
  group_by(organ) %>%
  summarize(med_n_mut = median(n_mut)) %>%
  arrange(med_n_mut) %>%
  pull(organ) %>%
  unique() 

p = list()
bg = F

for (o in organs){
  p[[o]] <- ann %>%
    filter(organ == o) %>%
    mutate(med_n_mut = median(n_mut)) %>%
    ggplot(aes(x = reorder(guid, n_mut), y= log10(n_mut))) + 
    geom_hline(aes(yintercept = log10(med_n_mut), colour = 'red'), show.legend = F) +
    geom_point(size = 0.3, stroke = 0, shape = 16) + 
    theme_ch() + 
    theme(plot.margin = margin(0,-10,0,0, unit='pt'),
          axis.text = element_blank(),
          axis.ticks = element_blank(),
          axis.line.y = element_blank(),
          strip.placement = "outside",
          strip.text = element_text(angle = 45, 
                                    vjust = 0, hjust = 0,
                                    size = 6)
          ) + 
    coord_cartesian(clip = "off") + 
    labs(x = '', y = '') + 
    lims(y = c(0,6))  + 
    facet_wrap(~organ)
  
  if (bg) {
    p[[o]] <- p[[o]] + 
      theme(panel.background = element_rect(fill = 'grey', colour = 'grey',
                                            linewidth = 0.05))
  }
  
  bg = !bg
}

p$Thyroid <- p$Thyroid + 
  theme(axis.line.y = element_line(),
        axis.ticks.y = element_line(size = 0.3),
        axis.text.y = element_text(size = 6),
        axis.title.y = element_text(size = 6)
        ) + 
  labs(y = "log10(Number of mutations)")

p[['space']] = NA

pdf('figures/cohorts.pdf', width=7, height=4)
snakes <- plot_grid(plotlist=p, rel_widths = c(1.2, rep(1,22), 0.8), nrow = 1, align = "h", axis = "bt")
plot_grid(cohort_samples, snakes, nrow = 2, rel_heights = c(1.2,1), labels = c('a', 'b'), label_size = 8)
dev.off()

pdf('figures/snakes.pdf', width=7, height=2)
snakes
dev.off()



ann %>%
  group_by(cohort, organ) %>%
  summarize(value = n()) %>%
  pivot_wider(names_from=cohort) %>%
  mutate(across(-c(organ), ~replace_na(.,0))) %>%
  rowwise() %>%
  mutate(Total = sum(GEL, HMF, PCAWG, ICGC)) %>%
  arrange(desc(Total)) %>%
  rename(Organ = organ) %>% 
  as.data.frame() %>%
write.xlsx("results/figure_data/stables.xlsx", sheetName = "Table S1", col.names = TRUE, row.names = FALSE, append = FALSE)

