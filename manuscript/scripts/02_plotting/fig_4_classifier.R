source('scripts/03_plotting/plotting_header.R')
library(gtable)
library(ggpubr)

feature_set_col = c('DAMUTA' = 'coral', 'COSMIC' = 'dodgerblue', 'TMB' = 'grey')
col_dark = c('DAMUTA' = 'coral4', 'COSMIC' = 'dodgerblue4', 'TMB' = 'grey40')

### classifiers
rf <- read_csv('results/figure_data/full_classifier_metrics.csv') %>%
  arrange(baseline_acc) %>%
  filter(!task %in% c("pathway_OTHER", "pathway_GO_dna_metabolism", "pathway_GO_dna_repair")) %>%
  mutate(model = case_match(model, "model_0_2"~"COSMIC", "model_1_3"~"DAMUTA", 
          "model_1_4"~"Misrepair", "model_1_5"~"Damage", "model_5_0"~"TMB"),
         model = ordered(model, levels = c("DAMUTA",'Damage', 'Misrepair', 'COSMIC', "TMB")), 
         task =  str_sub(task,9,-1),
         task = ordered(task, levels = unique(task))) 


pvals <- rf %>%
  filter(model != 'TMB') %>%
  group_by(task) %>%
  summarise(
    pval = t.test(AP[model=="DAMUTA"], AP[model=="COSMIC"], paired=TRUE)$p.value
  ) %>%
  mutate(
    adj_pval = p.adjust(pval, method="BH"),
    signif = case_when(
      adj_pval < 0.001 ~ "***",
      adj_pval < 0.01 ~ "**", 
      adj_pval < 0.05 ~ "*",
      TRUE ~ "ns"
    )
  )

metrics <- rf %>%
  filter(model !%in% c('TMB', 'Damage', 'Misrepair')) %>%
  mutate(setting = 'Sample mutation status prediction') %>%
  ggplot(aes(x=task, y=AP)) + 
  geom_boxplot(aes(colour=model),outlier.shape=NA) + 
  geom_point(aes(color=model), size = 0.1,
             position=position_jitterdodge(dodge.width=0.75), show.legend = F) + 
  theme_ch() + 
  scale_colour_manual(values = c(feature_set_col, sig_group_col)) +
  labs(
    x = 'Repair pathway', y ='AUPRC of sample classification', 
    fill = 'Feature set', colour = 'Feature set') + 
  theme(
    legend.position = 'top',
    legend.key.size = unit(6, 'pt'),
    plot.margin = margin(0,5,0,5,'pt'),
    legend.box.spacing=unit(0,"pt")
  ) + 
  geom_text(data = pvals, aes(label = signif), y = max(rf$AP)+0.01, size = 6/.pt, show.legend = F) +
  coord_trans(clip='off')

#average performance for each task
metrics_df <- rf %>%
  group_by(task, model) %>%
  #mutate(r = row_number()) %>%
  #group_by(task, model, r) %>%
  summarize(across(where(is.numeric), ~mean(.))) %>%
  pivot_wider(id_cols = c(task), names_from = model, values_from = c(Accuracy, AP, baseline_acc))

t.test(metrics_df$Accuracy_DAMUTA, metrics_df$Accuracy_COSMIC, paired = T)
t.test(metrics_df$Accuracy_DAMUTA, metrics_df$baseline_acc_DAMUTA, paired = T)
t.test(metrics_df$Accuracy_COSMIC, metrics_df$baseline_acc_COSMIC, paired = T)
t.test(metrics_df$Accuracy_DAMUTA, metrics_df$Accuracy_TMB , paired = T)

t.test(metrics_df$AP_DAMUTA, metrics_df$AP_COSMIC , paired = T)
t.test(metrics_df$AP_DAMUTA, metrics_df$AP_TMB , paired = T)
t.test(metrics_df$AP_COSMIC, metrics_df$AP_TMB , paired = T)
t.test(metrics_df$AP_Misrepair, metrics_df$AP_TMB , paired = T)
t.test(metrics_df$AP_Damage, metrics_df$AP_TMB , paired = T)

# mean gain
mean((metrics_df$AP_DAMUTA/metrics_df$AP_COSMIC -1) * 100)

# average performance across pathways
rf %>%
  group_by(task, model) %>%
  summarize(across(where(is.numeric), ~mean(.))) %>%
  ggplot(aes(x = model, y = AP, colour = model)) + 
  geom_boxplot(outlier.shape = NA) +
  geom_line(aes(group = task), show.legend = F, colour = 'grey80', linetype = 'dashed') +
  geom_point(position=position_jitterdodge(dodge.width=0.75), show.legend = F) + 
  theme_ch() + 
  scale_colour_manual(values = c(feature_set_col, sig_group_col)) +
  stat_compare_means(
    comparisons = list(c("DAMUTA", "COSMIC"), c("DAMUTA", "TMB"), c("COSMIC", "TMB"), 
    c("TMB", "Misrepair"), c("TMB", "Damage")),
    method = "t.test",
    paired = TRUE,
    label = "p.signif"
  ) +
  #geom_text(data = pvals, aes(label = signif), y = max(rf$AP)+0.01, size = 6/.pt, show.legend = F) +
  labs(
    x = 'Feature set', y ='Mean AUPRC (10 seeds)', 
    fill = 'Feature set', colour = 'Feature set') + 
  theme(legend.position = 'right')


ggsave('figures/mean_performance_across_pathways.pdf', height = 4, width = 4)



rf %>%
  mutate(log_fc = log2(AP/baseline_ap)) %>%
  mutate(setting = 'Sample mutation status prediction') %>%
  ggplot(aes(x=model, y=log_fc)) + 
  geom_boxplot(aes(colour=model),outlier.shape=NA) + 
  geom_point(aes(color=model), size = 0.1,
             position=position_jitterdodge(dodge.width=0.75), show.legend = F) + 
  theme_ch() + 
  scale_colour_manual(values = c(feature_set_col, sig_group_col)) +
  labs(
    x = 'Repair pathway', y ='log2 fold change above baseline AUPRC', 
    fill = 'Feature set', colour = 'Feature set') + 
  theme(
    legend.position = 'top',
    legend.key.size = unit(6, 'pt'),
    plot.margin = margin(0,5,0,5,'pt'),
    legend.box.spacing=unit(0,"pt"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) + 
  facet_wrap(~task, nrow = 1) +
  theme_ch() + 
  theme(plot.background = element_rect(fill = 'white')) +
  coord_trans(clip='off') + 
  rotate_x_text(angle = 45)

ggsave('figures/pathway_classification.pdf', height = 4, width = 6)




# stat data 
class_balance <- read_csv('results/figure_data/class_balance.csv') %>% 
  select(-c(`...1`)) %>%
  pivot_longer(everything(), names_to = 'pathway') %>%
  mutate(pathway = str_sub(pathway, 9, -1)) %>%
  group_by(pathway) %>%
  summarize(
    n_mutated = sum(value>0),
    percent_mutated = n_mutated/n() * 100
  )

pathway_membership <- read_csv('data/DDR_pthw.csv') %>%
  group_by(pathway) %>%
  summarize(n_in_pthw = n())


n_sample_n_gene <- merge(class_balance, pathway_membership, by = 'pathway') %>%
  filter(!pathway %in% c('OTHER', 'GO_dna_metabolism', 'GO_dna_repair')) %>%
  pivot_longer(c(percent_mutated, n_in_pthw)) %>%
  mutate(
    name = ordered(name, levels = c('percent_mutated', 'n_in_pthw')),
    label = ifelse(name == 'percent_mutated', round(value, 1), value),
    value = ifelse(name == 'n_in_pthw', value/10, value),
    pathway = ordered(pathway, levels = levels(rf$task))
  ) %>%
  ggplot(aes(x = pathway, y = value, fill = name)) + 
  geom_bar(stat = 'identity', position = "dodge") + 
  geom_bar(stat = 'identity',position = "dodge") + 
  scale_y_continuous(name = "Percent of samples altered",
    sec.axis = sec_axis(transform=~.*10, name="Number of genes in pathway")
  ) + 
  geom_text(aes(label=label),size = 6/.pt, vjust = -0.5, position = position_dodge(width=0.9)) + 
  theme_ch() + 
  scale_fill_manual(values = c(percent_mutated='grey20', n_in_pthw='grey60'), 
                    labels = c(percent_mutated='Percent of samples altered', 
                               n_in_pthw='Number of genes in pathway')) + 
  theme(legend.position = c(0.7, 0.95),
        legend.key.size = unit(6,"pt"),
        plot.margin = margin(20,5,0,10,'pt')) + 
  labs(fill = '', x = '') + 
  coord_trans(clip='off')



### feature importances
ann = read_csv('data/phg_clinical_ann.csv')
W = read_csv('results/figure_data/h_W.csv') %>% 
  rename(guid=`...1`) %>%
  slice(match(ann$guid, guid))
ds = read_delim('results/figure_data/phg_deconstructsigs_activities.csv', delim='\t') %>% 
  rename(guid = Type) %>%
  slice(match(ann$guid, guid))

W_long <- W %>%
  pivot_longer(-c(guid)) %>%
  rename(sig=name) %>%
  separate(sig, into=c('damage', 'misrepair'), remove = F, sep= '_') 

theta <- W_long %>%
  group_by(guid, damage) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'damage', values_from = value) %>%
  ungroup() %>%
  mutate(across(where(is.numeric), ~ifelse(. < 0.05, 0, .))) %>%
  select(c(guid,paste0('D',1:18))) %>%
  slice(match(ann$guid, guid))

gamma <- W_long %>%
  group_by(guid, misrepair) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'misrepair', values_from = value) %>%
  ungroup() %>%
  mutate(across(where(is.numeric), ~ifelse(. < 0.05, 0, .))) %>%
  select(c(guid,paste0('M',1:6))) %>%
  slice(match(ann$guid, guid))


imps <- read_csv('results/figure_data/feature_importances.csv', col_select = -c(`...1`)) %>%
  filter(pathway %in% paste0('pathway_', levels(rf$task))) %>%
  mutate(model = case_match(model, "model_0_2"~"COSMIC", "model_1_3"~"DAMUTA", "model_5_0"~"TMB"),
         model = ordered(model, levels = c("DAMUTA", 'COSMIC', "TMB")),
         pathway =  str_sub(pathway,9,-1),
         pathway = ordered(pathway, levels = levels(rf$task)),
         feature = ordered(feature, levels = unique(feature))
  ) %>%
  group_by(model, pathway, feature) %>%
  summarize(
    split_importance = median(split_importance),
    sd_split_importance = sd(split_importance)
  ) 

heat_col = circlize::colorRamp2(c(0, 100), c("white", 'navy'))
mut_col = circlize::colorRamp2(c(2.4, 3.6), c("white", "black"))

# damuta split importances
med_n_mut <- theta %>% 
  bind_cols(gamma %>% select(-c(guid))) %>% 
  mutate(across(D1:M6, ~.*ann$n_mut),
         across(D1:M6, ~ifelse(.==0, NA, .))) %>% 
  pivot_longer(-c(guid)) %>% 
  group_by(name) %>% 
  summarize(value = median(value, na.rm=T)) %>%
  slice(match(c(paste0('D',1:18), paste0("M", 1:6)), name)) %>%
  column_to_rownames('name')  
  

hm <- imps %>%
  filter(model =='DAMUTA') %>%
  pivot_wider(names_from = pathway, values_from=split_importance) %>%
  column_to_rownames('feature') %>%
  select(levels(rf$task)) %>%
  as.matrix() %>%
  Heatmap(
    show_heatmap_legend = F,
    cluster_columns = F, cluster_row_slices = F,
    row_dend_reorder = T, row_names_side = 'left',
    row_split = factor(c( rep("Damage", 18), rep("Misrepair", 6)), levels = c('Misrepair', 'Damage')),
    col=heat_col, name = 'Split importance',
    heatmap_legend_param = list(
      labels_gp = gpar(fontsize=6), 
      title_gp = gpar(fontsize=6, fontface='bold'),
      grid_width = unit(3,'mm'), 
      grid_height = unit(1, 'mm'),
      legend_direction = "horizontal", 
      legend_width = unit(5, "cm")
    )
  ) + 
  rowAnnotation(`log10 median N mutations` = log10(med_n_mut$value),
                col = list(`log10 median N mutations` = mut_col),
                show_annotation_name =F,
                annotation_legend_param = list(direction = 'horizontal', 
                                               title_position = "topleft",
                                               #color_bar = "discrete",
                                               #nrow=1,
                                               labels_gp = gpar(fontsize=6), 
                                               title_gp = gpar(fontsize=6, fontface='bold'),
                                               grid_width = unit(3,'mm'), 
                                               grid_height = unit(1, 'mm')
                ),
                
                annotation_name_gp = gpar(fontsize=6),
                annotation_name_side = "bottom",
                simple_anno_size = unit(3, "mm"))

damuta_imp_hm_grob <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(hm, 
                       heatmap_legend_side = "bottom",
                       annotation_legend_side = "bottom")))), 
    unit(1, "null"), unit(1, "null"))
)



# cosmic split importances
ord <- 
  imps %>%
  filter(model =='COSMIC') %>%
  group_by(feature) %>%
  filter(sum(split_importance)>0) %>%
  pivot_wider(names_from = pathway, values_from=split_importance) %>%
  pull(feature)

med_n_mut <- ds %>%
  mutate(across(SBS1:SBS94, ~.*ann$n_mut),
         across(SBS1:SBS94, ~ifelse(.==0, NA, .))) %>% 
  pivot_longer(-c(guid)) %>% 
  group_by(name) %>% 
  summarize(value = median(value, na.rm=T)) %>%
  slice(match(ord, name)) %>%
  column_to_rownames('name') 


hm <- imps %>%
  filter(model =='COSMIC') %>%
  group_by(feature) %>%
  filter(sum(split_importance)>0) %>%
  pivot_wider(names_from = pathway, values_from=split_importance) %>%
  column_to_rownames('feature') %>%
  select(levels(rf$task)) %>%
  as.matrix() %>%
  Heatmap(
    row_names_side = 'left',
    cluster_columns = F,
    show_heatmap_legend = T,
    col=heat_col, name = 'Split importance',
    heatmap_legend_param = list(
      labels_gp = gpar(fontsize=6), 
      title_gp = gpar(fontsize=6, fontface='bold'),
      grid_width = unit(3,'mm'), 
      grid_height = unit(1, 'mm'),
      legend_direction = "horizontal", 
      legend_width = unit(5, "cm")
    )
  ) + 
  rowAnnotation(`log10 median N mutations` = log10(med_n_mut$value),
                col = list(`log10 median N mutations` = mut_col),
                show_legend = F,
                show_annotation_name =F, 
                annotation_name_gp = gpar(fontsize=6),
                annotation_name_side = "bottom",
                simple_anno_size = unit(3, "mm"))


cosmic_imp_hm_grob <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(hm, 
                       heatmap_legend_side = "top",
                       annotation_legend_side = "bottom")))), 
    unit(1, "null"), unit(1, "null"))
)


dm_imps <- imps %>% 
  filter(model == 'DAMUTA') %>%
  mutate(type = ifelse(str_sub(feature,1,1)=='D', 'Damage', "Misrepair")) %>%
  ggplot(aes(colour=type, x = type, y = split_importance)) + 
  geom_boxplot(outliers=F, show.legend = F) + 
  geom_point(aes(colour = type), show.legend = F, size = 1.5/.pt,
             position = position_jitterdodge(jitter.width = 0.3)) +
  theme_ch() + 
  scale_colour_manual(values = c('maroon3', 'aquamarine3')) + 
  labs(x = "Signature type", y = "Split importance", fill = 'Signature type') 



dm_cosmic_imps <- imps %>% 
  filter(model != 'TMB') %>%
  mutate(type = case_when(str_sub(feature,1,1)=='D'~'Damage',
                          str_sub(feature,1,1)=='M'~'Misrepair',
                          str_sub(feature,1,1)=='S'~'COSMIC')) %>%
  ggplot(aes(colour=type, x = type, y = split_importance)) + 
  geom_boxplot(outliers=F, show.legend = F) + 
  geom_point(aes(colour = type), show.legend = F, size = 1.5/.pt,
             position = position_jitterdodge(jitter.width = 0.3)) +
  theme_ch() + 
  scale_colour_manual(values = c(sig_group_col, feature_set_col)) + 
  labs(x = "Signature type", y = "Split importance", fill = 'Signature type') 



#tl <- plot_grid(n_sample_n_gene,  metrics,
#                 labels = c('a','b'), label_size = 8,
#                 nrow=2, align='v', axis = 'lr', rel_heights = c(1,1.5))
#tr <- plot_grid(dm_imps, ncol=1, labels = c('c'), label_size = 8)
#br <- plot_grid(damuta_imp_hm_grob, cosmic_imp_hm_grob, ncol=2, labels = c('d', 'e'), label_size = 8)
#t <- plot_grid(tl, tr, rel_widths = c(1.3,1))
#b <- plot_grid(br, rel_widths = c(1,2))
#plot_grid(t, b, nrow=2, rel_heights = c(1,1.5))
#ggsave('figures/fig_6.pdf', width=7, height=9.5)


l <- plot_grid(n_sample_n_gene,  metrics, dm_imps + coord_flip(),
               labels = c('a','b','c'), label_size = 8,
               nrow=3, align='v', axis = 'lr', rel_heights = c(0.8,1.5,1))
r <- plot_grid(damuta_imp_hm_grob, cosmic_imp_hm_grob, ncol=1, labels = c('d', 'e'), label_size = 8)
plot_grid(l, r, nrow=1, rel_widths = c(1.8,1))
ggsave('figures/fig_4.pdf', width=7, height=9.5)

