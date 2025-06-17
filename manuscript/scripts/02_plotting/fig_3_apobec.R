source('scripts/03_plotting/plotting_header.R')

ann = read_csv('data/apobec_ann.csv') %>% mutate(
  Cell_Line = sub("-.*", "", MutationType),
  Tissue = recode(Cell_Line, JSC = "BCell Lymphoma", BT = "Breast", 
                  MDA=  "Breast", BC= "BCell Lymphoma", HT= "Bladder"),
  Tissue = ordered(Tissue, levels = c('BCell Lymphoma', 'Breast', 'Bladder')),
  Experiment = ordered(Experiment, levels = c('WT', 'A3B_KO', 'A3A_KO', 'A3A_A3B_KO', 'UNG2_KO', 'UNG-GFP', 'SMUG_KO', 'REV1_KO')),
  Condition = case_when(Experiment == "UNG2_KO" ~ "UNG low",
                        Experiment == "UNG-GFP" ~ "UNG high",
                        Cell_Line == "BT" & Experiment== "WT" ~ "UNG high",
                        Cell_Line == "MDA" & Experiment== "WT" ~ "UNG high",
                        Cell_Line == "BC" & Experiment== "WT" ~ "UNG low",
  )
) %>% 
  mutate(
    apobec_condition=ifelse(Experiment %in% c("UNG2_KO", "UNG-GFP"), "WT", levels(Experiment)[Experiment]),
    apobec_condition = gsub('_', ' ', apobec_condition),
    apobec_condition = gsub('A3A A3B', 'A3A+A3B', apobec_condition),
    ung_condition=ifelse(Cell_Line =='BC', 'UNG low', 'UNG high'), 
    ung_condition=ifelse(Experiment == 'UNG-GFP', 'UNG high', ung_condition),
    ung_condition=ifelse(Experiment == 'UNG2_KO', 'UNG low', ung_condition)
  )

W <- readr::read_csv('results/figure_data/apobec_h_W.csv')
colnames(W) = c('guid', paste0('D', rep(1:18, each=6), '_M', rep(1:6, times=18)))
W <- W %>% slice(match(ann$MutationType, guid))

W_long <- W %>%
  pivot_longer(-c(guid)) %>%
  rename(sig=name) %>%
  separate(sig, into=c('damage', 'misrepair'), remove = F, sep= '_')  

theta <- W_long %>%
  group_by(guid, damage) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'damage', values_from = value) %>%
  ungroup() %>%
  select(c(guid,paste0('D',1:18)))

gamma <- W_long %>%
  group_by(guid, misrepair) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'misrepair', values_from = value) %>%
  ungroup()

connect_activities = read_csv('results/figure_data/apobec_connect_activities.csv') %>%
  rename_with(~gsub('connect_sig_', 'C', .x))
  
sbs_activities = read_csv('data/apobec_ds_acts.csv') %>% 
  slice(match(ann$MutationType, rowname))

#############################
# connect sig activities 
mut_col = circlize::colorRamp2(c(2.5, 4), c("white", "black"))

apobec_condition_col = c(WT = 'grey65', 'A3B KO' = 'orangered', 'A3A KO' = 'deepskyblue',  'A3A+A3B KO' = 'purple')
ung_condition_col  = c(`UNG low` = 'sienna2', `UNG high` = 'sienna4')
both_cond_col = c(WT = 'grey65', 'A3B KO' = 'orangered', 'A3A KO' = 'deepskyblue',  
                  'A3A+A3B KO' = 'purple', UNG2_KO = 'sienna2', `UNG-GFP` = 'sienna4')
tissue_col = c(`BCell Lymphoma` = 'darkred', Breast='pink2')
cellline_col = c(BT = 'pink3', MDA = 'pink', BC = 'darkred')


d_col = circlize::colorRamp2(c(0, 0.5, 1), c("white", adjustcolor("maroon3", alpha.f = 0.5), "maroon4"))
m_col = circlize::colorRamp2(c(0, 0.5, 1), c("white", adjustcolor("aquamarine3", alpha.f = 0.5), "aquamarine4"))
c_col = circlize::colorRamp2(c(0, 0.5, 1), c("white", adjustcolor("darkgoldenrod2", alpha.f = 0.5), "darkgoldenrod3"))


bw_col = circlize::colorRamp2(c(0, 0.5, 0.75, 1), c("navy", 'white', 'orange', 'red'))

###############


df = bind_cols(select(theta, -guid), select(gamma, -guid)) 
sel = grepl(ann$Experiment, pattern='A3') | grepl(ann$Experiment, pattern='UNG') | ann$Experiment == 'WT'
sel = sel & (ann$`In_vitro_time (days)` != 'NA(parent)')
sel = sel & ((ann$Cell_Line=='BT') | (ann$Cell_Line=='MDA') | (ann$Cell_Line=='BC') )
apobec_only = df[sel,] %>% mutate(across(D1:M6, ~ifelse(.<0.05, 0, .)))

dm_hm <- Heatmap(as.matrix(select(apobec_only, D1:D18) %>% select_if(~ !is.numeric(.) || sum(.) != 0)) , 
                 width = 18*unit(3.5/50, "in"),
                 row_split = ann[sel,]$Experiment, cluster_rows = F,
                 column_title_gp = gpar(fontsize = 6),
                 row_title_gp = gpar(fontsize = 6),
                 name='Damage Activity', col=d_col, column_title='Damage',
                 heatmap_legend_param = list(
                   labels_gp = gpar(fontsize=6), 
                   title_gp = gpar(fontsize=6, fontface='bold'),
                   grid_width = unit(3,'mm'), 
                   grid_height = unit(1, 'mm')
                 )) + 
  Heatmap(as.matrix(select(apobec_only, M1:M6) %>% select_if(~ !is.numeric(.) || sum(.) != 0)) , 
          width = 6*unit(3.5/50, "in"),
          row_split = ann[sel,]$Experiment, cluster_rows = F, column_title_gp = gpar(fontsize = 6),
          name='Misrepair Activity', col=m_col, column_title='Misrepair',
          heatmap_legend_param = list(
            labels_gp = gpar(fontsize=6), 
            title_gp = gpar(fontsize=6, fontface='bold'),
            grid_width = unit(3,'mm'), 
            grid_height = unit(1, 'mm')
          )) + 
  rowAnnotation(`log10 N mutations` = log10(ann[sel,]$n_mut), 
                `Cell line` = ann[sel,]$Cell_Line, 
                #`Experiment` = ann[sel,]$Experiment,
                `APOBEC Condition` = ann[sel,]$apobec_condition,
                `UNG Condition` = ann[sel,]$ung_condition,
                col=list(`log10 N mutations`=mut_col, `Cell line`=cellline_col,
                         `Experiment` = both_cond_col,
                         `APOBEC Condition` = apobec_condition_col, 
                         `UNG Condition`= ung_condition_col), 
                annotation_legend_param = list(direction = 'horizontal',  
                                               #nrow = 4,
                                               title_position = "topleft",
                                               color_bar = "discrete",
                                               labels_gp = gpar(fontsize=6), 
                                               title_gp = gpar(fontsize=6, fontface='bold'),
                                               grid_width = unit(3,'mm'), 
                                               grid_height = unit(1, 'mm')
                ),
                
                annotation_name_gp = gpar(fontsize=6),
                annotation_name_side = "top",
                simple_anno_size = unit(3, "mm"))


petljack_dm_hm <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(dm_hm, 
                       heatmap_legend_side = "right",
                       annotation_legend_side = "right")))), 
    unit(1, "null"), unit(1, "null"))
)

# pdf('figures/petljack_dm_hm.pdf', width=3.5, height=7)
# plot_grid(petljack_dm_hm, labels = 'a', label_size = 8)
# dev.off()


sel = grepl(ann$Experiment, pattern='A3') | grepl(ann$Experiment, pattern='UNG') | ann$Experiment == 'WT'
sel = sel & (ann$`In_vitro_time (days)` != 'NA(parent)')
sel = sel & ((ann$Cell_Line=='BT') | (ann$Cell_Line=='MDA') | (ann$Cell_Line=='BC') )


mat <- df %>%
  select(M1:M6) %>% 
  subset(sel) %>%
  proxy::dist(method = 'cosine', pairwise = T, diag = F, upper = T) %>%
  as.matrix()

o = seriate(mat, method = "PCA")

ha <- HeatmapAnnotation(
    `log10 N mutations` = log10(ann[sel,]$n_mut), 
    `Cell line` = ann[sel,]$Cell_Line, 
    `APOBEC Condition` = ann[sel,]$apobec_condition,
    `UNG Condition` = ann[sel,]$ung_condition,
    #`Experiment` = ann[sel,]$Experiment,
    #`Tissue` = ann[sel,]$Tissue,
    col=list(
        `log10 N mutations`=mut_col, 
        `Cell line`=cellline_col,
        #`Experiment` = both_cond_col,
        `APOBEC Condition` = apobec_condition_col, 
        `UNG Condition`= ung_condition_col
        #`Tissue` = tissue_col
    ),
    show_legend = F,
    annotation_name_gp = gpar(fontsize=6),
    annotation_name_side = "right",
    simple_anno_size = unit(3, "mm")
  )

hm <- Heatmap(1-mat, top_annotation = ha,
              name = 'Cosine similarity',
              height = ncol(mat)*unit(6/600, "in"), width = ncol(mat)*unit(6/600, "in"), 
              row_order = get_order(o, 1), column_order = get_order(o, 2),
              col=bw_col, 
              row_labels = rep('', sum(sel)), column_labels = rep('', sum(sel)),
              heatmap_legend_param = list(
                labels_gp = gpar(fontsize=6), 
                title_gp = gpar(fontsize=6, fontface='bold'),
                grid_width = unit(3,'mm'), 
                grid_height = unit(1, 'mm')
              ))
 

misrepair_pca_hm <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(hm, 
                       heatmap_legend_side = "right",
                       annotation_legend_side = "right")))), 
    unit(1, "null"), unit(1, "null"))
)

#pdf('figures/apobec_misrepair_pca_hm.pdf', width=6, height=8)
#plot_grid(misrepair_pca_hm, labels = 'c', label_size = 8)
#dev.off()

mat <- df %>%
  select(D1:D18) %>% 
  subset(sel) %>%
  proxy::dist(method = 'cosine', pairwise = T, diag = F, upper = T) %>%
  as.matrix()

o = seriate(mat, method = "PCA")

ha <- HeatmapAnnotation(
  `log10 N mutations` = log10(ann[sel,]$n_mut), 
  `Cell line` = ann[sel,]$Cell_Line, 
  `APOBEC Condition` = ann[sel,]$apobec_condition,
  `UNG Condition` = ann[sel,]$ung_condition,
  #`Experiment` = ann[sel,]$Experiment,
  #`Tissue` = ann[sel,]$Tissue,
  col=list(
    `log10 N mutations`=mut_col, 
    `Cell line`=cellline_col,
    #`Experiment` = both_cond_col,
    `APOBEC Condition` = apobec_condition_col, 
    `UNG Condition`= ung_condition_col
    #`Tissue` = tissue_col
  ),
  show_legend = F,
  annotation_name_gp = gpar(fontsize=6),
  annotation_name_side = "right",
  simple_anno_size = unit(3, "mm")
)

hm <- Heatmap(1-mat,
              top_annotation = ha,
              name = 'Cosine similarity',
              height = ncol(mat)*unit(6/600, "in"), width = ncol(mat)*unit(6/600, "in"), 
              row_order = get_order(o), column_order = get_order(o),
              col=bw_col,
              
              row_labels = rep('', sum(sel)), column_labels = rep('', sum(sel)),
              heatmap_legend_param = list(
                labels_gp = gpar(fontsize=6), 
                title_gp = gpar(fontsize=6, fontface='bold'),
                grid_width = unit(3,'mm'), 
                grid_height = unit(1, 'mm')
              ))

damage_pca_hm <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(hm, 
                       heatmap_legend_side = "right",
                       annotation_legend_side = "right")))), 
    unit(1, "null"), unit(1, "null"))
)

#pdf('figures/damage_pca_hm.pdf', width=6, height=6)
#plot_grid(damage_pca_hm, labels = 'c', label_size = 8)
#dev.off()

mat <- sbs_activities %>%
  select(-rowname) %>%
  subset(sel) %>%
  proxy::dist(method = 'cosine', pairwise = T, diag = T, upper = T) %>%
  as.matrix()

o = seriate(mat, method = "PCA")

ha <- HeatmapAnnotation(
  `log10 N mutations` = log10(ann[sel,]$n_mut), 
  #`Cell line` = ann[sel,]$Cell_Line, 
  `APOBEC Condition` = ann[sel,]$apobec_condition,
  `UNG Condition` = ann[sel,]$ung_condition,
  #`Experiment` = ann[sel,]$Experiment,
  `Tissue` = ann[sel,]$Tissue,
  col=list(
    `log10 N mutations`=mut_col, 
    #`Cell line`=cellline_col,
    #`Experiment` = both_cond_col,
    `APOBEC Condition` = apobec_condition_col, 
    `UNG Condition`= ung_condition_col,
    `Tissue` = tissue_col
  ),
  show_legend = F,
  annotation_name_gp = gpar(fontsize=6),
  annotation_name_side = "right",
  simple_anno_size = unit(3, "mm")
)

hm <- Heatmap(as.matrix(1-mat), 
              top_annotation = ha,
              name = 'Cosine similarity',
              height = ncol(mat)*unit(6/300, "in"), width = ncol(mat)*unit(6/300, "in"), 
              row_order = get_order(o, 1), 
              column_order = get_order(o, 1),
              col=bw_col,
              
              row_labels = rep('', sum(sel)), column_labels = rep('', sum(sel)),
              heatmap_legend_param = list(
                labels_gp = gpar(fontsize=6), 
                title_gp = gpar(fontsize=6, fontface='bold'),
                grid_width = unit(3,'mm'), 
                grid_height = unit(1, 'mm')
              )
)


cosmic_pca_hm <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(hm, 
                       heatmap_legend_side = "right",
                       annotation_legend_side = "right")))), 
    unit(1, "null"), unit(1, "null"))
)

#pdf('plots/cosmic_pca_hm.pdf', width=4, height=4)
#plot_grid(cosmic_pca_hm, labels = 'd', label_size = 8)
#dev.off()



df = bind_cols(connect_activities[2:31], theta[2:19], gamma[2:7]) 


# etiology
SBS2_C14 <- sbs_activities[sel,] %>%
  bind_cols(df[sel,], ann[sel,]) %>%
  ggplot(aes(x = SBS2, y = C14)) + 
  geom_point() + 
  geom_smooth(method='lm') + 
  stat_cor(label.x = 0, label.y = .6, size = 6/.pt) +
  stat_regline_equation(label.x = 0, label.y = .65, size = 6/.pt) + 
  geom_abline(slope=1, intercept=0, linetype='dashed') + 
  lims(x=c(0,0.75),y=c(0,0.75)) +
  theme_ch()


SBS13_C3 <- sbs_activities[sel,] %>%
  bind_cols(df[sel,], ann[sel,]) %>%
  ggplot(aes(x = SBS13, y = C3)) + 
  geom_point() + 
  geom_smooth(method='lm') + 
  stat_cor(label.x = 0, label.y = .6, size = 6/.pt) +
  stat_regline_equation(label.x = 0, label.y = .65, size = 6/.pt) + 
  geom_abline(slope=1, intercept=0, linetype='dashed') + 
  lims(x=c(0,0.75),y=c(0,0.75)) +
  theme_ch()


plot_grid(SBS2_C14, SBS13_C3, label_size = 8)
ggsave('figures/apobec_sbs_corr.png', width=4, height=2)

########### 
# ii) can attribute the differences seen between those groups to damage activity 
comparisons <- list(c("WT", "A3A KO"), c("WT", "A3B KO"), c("A3A KO", "A3B KO"), c("A3A KO", "A3A+A3B KO"), c("WT", "A3A+A3B KO"), c("A3B KO", "A3A+A3B KO"))
comparisons <- list(c("WT", "A3A KO"), c("WT", "A3A+A3B KO"), c("WT", "A3B KO"))

stat.test <- (df) %>%
  bind_cols(ann) %>%
  filter(Tissue=="Breast", `In_vitro_time (days)` != 'NA(parent)',
         Experiment %in% c('WT', 'A3B_KO', 'A3A_KO', 'A3A_A3B_KO')) %>%
  mutate(Experiment = gsub('_', ' ', Experiment),
         Experiment = gsub("A3A A3B", 'A3A+A3B', Experiment)
  ) %>%
  pivot_longer(c("D2", "D5", "D10", "M3", "M4", "M6")) %>%
  mutate(type = ifelse(str_sub(name ,1,1) == "D", "Damage", "Misrepair")) %>%
  group_by(name) %>%
  #group_by(type) %>%
  t_test(value~Experiment, comparisons = comparisons) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance() %>%
  mutate(p.adj.signif = ifelse(p.adj.signif == "ns", "", p.adj.signif)) %>%
  mutate(name = factor(name, levels=c("D2", "D5", "D10", "M3", "M4", "M6")))



## DAMUTA activity
apobec_split <- (df) %>%
  bind_cols(ann) %>%
  filter(Cell_Line %in% c('BT', 'MDA'), `In_vitro_time (days)` != 'NA(parent)',
         Experiment %in% c('WT', 'A3B_KO', 'A3A_KO', 'A3A_A3B_KO')) %>%
  mutate(Experiment = gsub('_', ' ', Experiment),
         Experiment = gsub("A3A A3B", 'A3A+A3B', Experiment)
  ) %>%
  pivot_longer(c("D2", "D5", "D10", "M4", "M3", "M6")) %>%
  mutate(type = ifelse(str_sub(name ,1,1) == "D", "Damage", "Misrepair"),
         name = factor(name, levels = c("D2", "D5", "D10", "M3", "M4","M6"))) %>%
  ggplot(aes(x = Experiment, y = value, colour = Experiment)) + 
  #geom_bar(aes(fill=Experiment), alpha = 0.2, position = 'dodge', 
  #         stat = "summary", width = 0.8, fun = 'mean') + 
  geom_boxplot(outliers = F) + 
  geom_point(aes(x=Experiment), size = 0.1, 
            position = position_jitterdodge(jitter.width = 1)) +
  facet_wrap(~name, nrow=1) +
  scale_colour_manual(values = apobec_condition_col) + 
  labs(y = "Activity") + 
  theme_ch() + 
  theme(legend.position='top',
        legend.key.size = unit(8, 'pt'),
        legend.margin = margin(0, 0, 0, 0)) + 
  stat_pvalue_manual(stat.test, label.size = 6/.pt, hide.ns = F,
                     label = "p.adj.signif", tip.length = 0,
                     y.position = c(.75, .8, .85),
                     inherit.aes=F) + 
  coord_trans(clip = 'off') + 
  labs(fill = 'APOBEC condition') + 
  rotate_x_text(45)

#pdf('../plots/apobec_split.pdf', width=4.5, height=3)
#plot_grid(apobec_split, labels  = 'a', label_size = 8)
#dev.off()



comparisons <- list( c("UNG high", "UNG low"))
stat.test <- df %>%
  bind_cols(ann) %>%
  filter(Cell_Line %in% c("BC", "BT", "MDA"),
         Condition %in% c("UNG high", "UNG low"),
         `In_vitro_time (days)` != 'NA(parent)'
         ) %>%
  pivot_longer(c("D2", "D5", "D10", "M3", "M4", "M6")) %>%
  mutate(type = ifelse(str_sub(name ,1,1) == "D", "Damage", "Misrepair")) %>%
  group_by(Tissue, name) %>%
  t_test(value~Condition, comparisons = comparisons) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance() %>%
  mutate(p.adj.signif = ifelse(p.adj.signif == "ns", "", p.adj.signif)) %>%
  add_xy_position(x = "Tissue", dodge = 0.8) %>%
  mutate(name = factor(name, levels=c("D2", "D5", "D10", "M3", "M4", "M6")))



ung_split<- df %>%
  bind_cols(ann) %>%
  filter(Cell_Line %in% c("BT", "BC", "MDA")) %>%
  pivot_longer(c("D2", "D5", "D10", "M3", "M4","M6")) %>%
  mutate(type = ifelse(str_sub(name ,1,1) == "D", "Damage", "Misrepair"),
         name = factor(name, levels = c("D2", "D5", "D10", "M3", "M4","M6"))) %>%
  filter(Experiment %in% c('WT', 'UNG2_KO', 'UNG-GFP'),
         `In_vitro_time (days)` != 'NA(parent)',
         ) %>% 
  ggplot(aes(x = Tissue, y = value, colour = Condition)) + 
  geom_boxplot(outliers=F) + 
  geom_point(position = position_jitterdodge(jitter.width = 0.2), size = 0.05) + 
  facet_wrap(~name, nrow = 1) + 
  #scale_fill_manual(values = ung_condition_col) + 
  scale_colour_manual(values = ung_condition_col) +
  stat_pvalue_manual(stat.test, label = "p.adj.signif", tip.length = 0,
                     y.position = .85, label.size=6/.pt,
                     inherit.aes=F) + 
  theme_ch() + 
  theme(legend.position = 'top',
        legend.key.size = unit(8, 'pt'),
        legend.margin = margin(0, 0, 0, 0)) + 
  rotate_x_text(45) + 
  labs(fill = 'UNG condition', y = 'Activity') + 
  coord_trans(clip = 'off')


#pdf('figures/ung_split.pdf', width=4.5, height=3)
#plot_grid(ung_split, labels='c', label_size = 8)
#dev.off()



# proportion of samples in each condition with active connection
apobec_bipart <- W %>%
  mutate(across(D1_M1:D18_M6, ~ifelse(. <= 0.05, NA, .)), 
         Cell_Line = ann$Cell_Line, Experiment = ann$Experiment, 
         `In_vitro_time (days)`=ann$`In_vitro_time (days)`) %>%
  filter(Cell_Line %in% c("BT", 'MDA'), 
         Experiment %in% c("WT", "A3A_KO", "A3B_KO", "A3A_A3B_KO"),
         `In_vitro_time (days)` != 'NA(parent)'
  )%>%
  mutate(Experiment = gsub('_', ' ', Experiment),
         Experiment = gsub("A3A A3B", 'A3A+A3B', Experiment)
  ) %>%
  mutate(Experiment = gsub('_', ' ', Experiment)) %>%
  group_by(Experiment) %>% mutate(n_exp =n()) %>%
  pivot_longer(-c(guid, Cell_Line, Experiment,`In_vitro_time (days)`, n_exp)) %>%
  separate(name, into=c('Damage', 'Misrepair'), remove = F, sep= '_') %>%
  filter(Damage %in% c("D2", "D5", "D10") & Misrepair %in% c("M3", "M4", "M6")) %>%
  arrange(Misrepair) %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  group_by(guid, Experiment, D) %>% mutate(d_size = (sum(value, na.rm=T)>0.05), 
                                           is_first = row_number()==1, 
                                           d_size = ifelse(d_size & is_first, 1,0)) %>%
  group_by(guid, Experiment, M) %>% mutate(m_size = (sum(value, na.rm=T)>0.05),
                                           is_first = row_number()==1, 
                                           m_size = ifelse(m_size & is_first, 1,0)) %>%
  group_by(Experiment, D) %>% mutate(d_size = sum(d_size)) %>%
  group_by(Experiment, M) %>% mutate(m_size = sum(m_size)) %>%
  group_by(Experiment, name) %>%
  summarize(Proportion = sum(value>0, na.rm=T),
            Activity = mean(value, na.rm=T),
            across(c(n_exp, d_size, m_size, D, M, Damage, Misrepair), ~first(.))) %>%
  mutate(across(c(Proportion, d_size, m_size), ~ifelse(.==0,NA,./n_exp) )) %>%
  ggplot() +
  geom_segment(aes(x=0, y=D, xend=1, yend=M, linewidth=Proportion, 
                   alpha = Activity), lineend = 'round') + 
  scale_linewidth(range = c(0,2), breaks = seq(0.25, 1, by=0.25)) + 
  scale_size_area(max_size=2, breaks = seq(0.25, 1, by=0.25)) +
  scale_alpha_continuous(breaks = seq(0.1, 0.25, by=0.05), limits = c(0, .25))+
  geom_point(aes(0, D, size = d_size), colour = 'maroon3') + 
  geom_point(aes(1, M, size = m_size), colour = 'aquamarine3') + 
  geom_text(aes(0, D, label=Damage), nudge_x=-0.3, size=6/.pt, check_overlap = T) + 
  geom_text(aes(1, M, label=Misrepair), nudge_x=0.3, size=6/.pt,check_overlap = T) + 
  facet_wrap(~Experiment, scales = 'free', ncol = 2) + 
  theme_void() + 
  theme(legend.position = 'right', 
        legend.direction = 'vertical',
        legend.box = 'vertical',
        legend.key.size = unit(6, 'pt'),
        legend.spacing.y = unit(1.0, 'mm'),
        legend.box.margin = margin(c(5,0,2,0)),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 6),
        strip.text = element_text(size = 6, margin=unit(c(0,0,2,0), 'mm')),
        plot.margin = unit(c(2,2,2,2), 'mm'),
        panel.spacing = unit(5, "mm"),
        legend.title.align=0.5
        ) +
  coord_trans(clip='off') + 
  guides(size = guide_legend(override.aes = list(colour = 'black') ))  + 
  labs(size = 'Proportion of samples\n with activity >0.05', linewidth = 'Proportion of samples\n with activity >0.05', alpha = 'Mean activity')



# proportion of samples in each condition with active connection
ung_bipart <- W %>%
  mutate(across(D1_M1:D18_M6, ~ifelse(. <= 0.05, NA, .))) %>%
  #mutate(across(D1_M1:D18_M6, ~.*ann$n_mut)) %>%
  mutate(Cell_Line = ann$Cell_Line, 
         Experiment = factor(ann$Condition, levels = c('UNG low', 'UNG high')),
         Tissue = ann$Tissue,`In_vitro_time (days)`=ann$`In_vitro_time (days)`
  ) %>%
  filter(Cell_Line %in% c("BT", "BC", "MDA"), 
         Experiment %in% c("UNG high", "UNG low"),
         `In_vitro_time (days)` != 'NA(parent)'
  )%>%
  group_by(Tissue, Experiment) %>% mutate(n_exp =n()) %>%
  pivot_longer(-c(guid, Tissue, Cell_Line, Experiment,`In_vitro_time (days)`, n_exp)) %>%
  separate(name, into=c('Damage', 'Misrepair'), remove = F, sep= '_') %>%
  filter(Damage %in% c("D2", "D5", "D10") & Misrepair %in% c("M3", "M4", "M6")) %>%
  arrange(Misrepair) %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  group_by(guid, Tissue, Experiment, D) %>% mutate(d_size = (sum(value, na.rm=T)>0.05), 
                                           is_first = row_number()==1, 
                                           d_size = ifelse(d_size & is_first, 1,0)) %>%
  group_by(guid, Tissue, Experiment, M) %>% mutate(m_size = (sum(value, na.rm=T)>0.05),
                                           is_first = row_number()==1, 
                                           m_size = ifelse(m_size & is_first, 1,0)) %>%
  group_by(Tissue, Experiment, D) %>% mutate(d_size = sum(d_size)) %>%
  group_by(Tissue, Experiment, M) %>% mutate(m_size = sum(m_size)) %>%
  group_by(Tissue, Experiment, name) %>%
  summarize(Proportion = sum(value>0, na.rm=T),
            Activity = mean(value, na.rm=T),
            across(c(n_exp, d_size, m_size, D, M, Damage, Misrepair), ~first(.))) %>%
  mutate(across(c(Proportion, d_size, m_size), ~ifelse(.==0,NA,./n_exp) )) %>%
  ggplot() +
  geom_segment(aes(x=0, y=D, xend=1, yend=M, linewidth=Proportion, 
                   alpha = Activity), lineend = 'round') + 
  scale_linewidth(range = c(0,2), breaks = seq(0.25, 1, by=0.25)) + 
  scale_size_area(max_size=2, breaks = seq(0.25, 1, by=0.25)) +
  scale_alpha_continuous(breaks = seq(0.1, 0.25, by=0.05), limits = c(0, .25))+
  geom_point(aes(0, D, size = d_size), colour = 'maroon3') + 
  geom_point(aes(1, M, size = m_size), colour = 'aquamarine3') + 
  geom_text(aes(0, D, label=Damage), nudge_x=-0.3, size=6/.pt, check_overlap = T) + 
  geom_text(aes(1, M, label=Misrepair), nudge_x=0.3, size=6/.pt,check_overlap = T) + 
  facet_grid(Experiment~Tissue) + 
  theme_void() + 
  theme(legend.position = 'right', 
        legend.direction = 'vertical',
        legend.box = 'vertical',
        legend.key.size = unit(6, 'pt'),
        legend.spacing.y = unit(1.0, 'mm'),
        legend.title = element_text(size = 6),
        legend.text = element_text(size = 6),
        strip.text.x = element_text(size = 6, margin=unit(c(0,0,2,0), 'mm')),
        strip.text.y = element_text(size = 6, angle = 90, margin=unit(c(0,0,2,2), 'mm')),
        plot.margin = unit(c(2,2,2,2), 'mm'),
        panel.spacing = unit(5, "mm"),
        legend.title.align=0.5,
        strip.clip = "off"
        )  +
  coord_trans(clip='off') + 
  guides(size = guide_legend(override.aes = list(colour = 'black') ))  + 
  labs(size = 'Proportion of samples\n with activity >0.05', linewidth = 'Proportion of samples\n with activity >0.05', alpha = 'Mean activity')

# plot all figures
split <- plot_grid(apobec_split, ung_split, labels = c('d', 'f'), label_size = 8, ncol=1, align = 'v', axis='lr')
lg <- get_legend(apobec_bipart + theme(legend.box = 'horizontal', legend.margin = margin(0, 0, 0, 0)))
apobec_bipart <- apobec_bipart + theme(legend.position='none', plot.margin = unit(0.5 * c(1.5,1,0,1), "cm"))
ung_bipart <- ung_bipart +  theme(legend.position='none', plot.margin = unit(0.5*c(0,1,4,1), "cm"))


bipart <- plot_grid(apobec_bipart, lg, ung_bipart, labels = c('e', '', 'g'),
          label_size = 8, ncol=1, align = 'v',
          axis='lr', rel_heights = c(1, 0.5, 1)
          )



r <- plot_grid(damage_pca_hm, misrepair_pca_hm,
          ncol=1, label_size = 8, labels = c('b', 'c'),
          align = 'hv', axis='lr')
t <- plot_grid(petljack_dm_hm, r, label_size = 8, labels = c('a', ''),
               align = 'h', axis='tb', rel_widths = c(1,1))
b <- plot_grid(split, bipart,
          ncol=2, label_size = 8,
          align = 'hv', axis='lr', rel_widths = c(2,1,2,1))
plot_grid(t, b, nrow=2)
ggsave('figures/fig_3.pdf', width=7, height=10)
ggsave('figures/fig_3.png', width=7, height=10)

