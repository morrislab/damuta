source('scripts/03_plotting/plotting_header.R')

ann = read_csv('data/phg_clinical_ann.csv')
connect_activities = read_csv('results/figure_data/connect_acts.csv') %>% 
  slice(match(ann$guid, guid))

# plot choice of N for connect nmf 
df <- read_csv('results/figure_data/n_compare.csv') %>% select(-`...1`)
p <- df %>% 
  rename(`Silhouette score`=sil, `MSE`=mse, `Frobenious norm`=frob, 
         `N COSMIC with >0.8 similarity`=n_cosmic_covered) %>%
  pivot_longer(c(`Silhouette score`, `MSE`, `Frobenious norm`, `N COSMIC with >0.8 similarity`)) %>%
  filter(name!='N COSMIC with >0.8 similarity') %>%
  ggplot(aes(x = rank, y = value)) + 
  geom_line(linewidth = 0.2) + 
  geom_point(size=0.2) +  
  geom_vline(xintercept = 30, alpha = 0.3, linewidth = 2, colour='red') + 
  facet_wrap(~name, scales = 'free_y', nrow=3, strip.position="left") + 
  labs(x='Rank of NMF decomposition', y='') + 
  theme_ch() + 
  theme(strip.placement = "outside")

# plot choice of N for connect nmf 
cosmic_cover <- df %>% 
  ggplot(aes(x = rank, y = n_cosmic_covered)) + 
  geom_line(linewidth = 0.2) + 
  geom_point(size=0.2) +  
  geom_vline(xintercept = 30, alpha = 0.3, linewidth = 2, colour = 'red') + 
  annotate(geom = 'text', label = 'x = 30', x=35, y = 0, colour = 'red', size=6/.pt) + 
  geom_abline(intercept = 0, slope = 1, linetype='dashed', linewidth=0.2, colour = 'grey40') + 
  annotate(geom = 'text', label = 'x = y', x=41, y = 37, colour = 'grey40', size = 6/.pt) + 
  labs(x='Rank of NMF decomposition', y='Number of COSMIC with >0.8 similarity') + 
  theme_ch() 

  
pdf('figures/nmf_n_sigs.pdf', width=7, height=4)
plot_grid(cosmic_cover, p, labels = c('a','b'), label_size = 8)
dev.off()


# connects

# create table with matchings
connect_ann <- read_csv('results/figure_data/connect_sig_ann.csv') %>%
  select(-`...1`) %>% 
  arrange(as.numeric(str_extract(connect, "\\d+$"))) %>%
  mutate(connect = ordered(connect, levels=unique(connect))) %>%
  left_join(
    read_csv('data/cosmic_etiologies.csv'),
    by = join_by(closest_cosmic==sig)
  ) 

connect <- read_csv('results/figure_data/connect_sigs.csv') %>% 
  rename(sig=`...1`) %>%
  arrange(as.numeric(str_extract(sig, "\\d+$"))) %>%
  mutate(sig = ordered(sig, levels=unique(sig)))

single_thresh = 2/3
connect_ann <- connect %>%
  pivot_longer(-sig) %>%
  separate(name, into=c('damage', 'misrepair'), remove = F, sep= '_') %>%
  group_by(sig, damage) %>% mutate(sum_damage = (sum(value))) %>%
  group_by(sig, misrepair) %>% mutate(sum_misrepair= (sum(value))) %>%
  group_by(sig) %>% #filter(sig=='C11') %>% print(n=Inf)
    summarize(
      max_damage = max(sum_damage),
      max_misrepair = max(sum_misrepair),
      connect_type = case_when(
        (max_damage>single_thresh) == T & (max_misrepair>single_thresh) == T ~ 'One-one',
        (max_damage>single_thresh) == T & (max_misrepair>single_thresh) == F ~ 'One-many',
        (max_damage>single_thresh) == F & (max_misrepair>single_thresh) == T ~ 'Many-one',
        (max_damage>single_thresh) == F & (max_misrepair>single_thresh) == F ~ 'Many-many'), 
    ) %>% 
    right_join(connect_ann, by = join_by(sig==connect))


connect_ann  %>%
  group_by(connect_type) %>%
  count()

connect_ann  %>%
  group_by(connect_type, etiology_l1) %>%
  count()


# for each tumour count number of connection types
connect_ann %>%
  ggplot(aes(x=etiology_l1, fill=connect_type)) +
  geom_bar() +
  scale_fill_manual(values = connect_type_col) +
  theme_ch() 

ggsave('figures/connect_type_by_organ.pdf', width=10, height=10)

# all connections
connect %>%
  #mutate(across(D1_M1:D18_M6, ~ifelse(. <= 0.01, NA, .))) %>%
  pivot_longer(-c(sig)) %>%
  arrange(as.numeric(str_extract(sig, "\\d+$"))) %>%
  mutate(sig = ordered(sig, levels=unique(sig))) %>%
  separate(name, into=c('Damage', 'Misrepair'), remove = F, sep= '_') %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  group_by(sig, D) %>% mutate(d_size = sum(value, na.rm=T)) %>%
  group_by(sig, M) %>% mutate(m_size = sum(value, na.rm=T)) %>%
  #filter(d_size > 0.01, m_size > 0.01) %>%
  mutate(across(value, ~ifelse(d_size<0.05,NA,.))) %>%
  mutate(across(value, ~ifelse(m_size<0.05,NA,.))) %>%
  mutate(across(c(d_size, m_size), ~ifelse(.<0.05,NA,.))) %>%
  ggplot() +
  geom_segment(aes(x=0, y=D, xend=1, yend=M, linewidth=value, alpha = value), lineend = 'round') + 
  scale_linewidth(range = c(0,3), breaks = c(0.2, 0.4, 0.6, 0.8)) + 
  scale_size_area(max_size=3, breaks = c(0.2, 0.4, 0.6, 0.8)) +  
  scale_alpha_continuous(breaks = c(0.2, 0.4, 0.6, 0.8)) + 
  geom_point(aes(0, D, size = d_size), colour = 'maroon3') + 
  geom_point(aes(1, M, size = m_size), colour = 'aquamarine3') + 
  geom_text(aes(0, D, label=Damage), nudge_x=-0.2, size=5/.pt) + 
  geom_text(aes(1, M, label=Misrepair), nudge_x=0.2, size=5/.pt) + 
  geom_text(aes(0.8, 2, label=paste0('Type: ', connect_type, '\nMatched to ', hungarian_cosmic, '\n(cossim ', round(hungarian_sim,2), ')' )),
            data=connect_ann, size=6/.pt) +
  theme_void() + 
  facet_wrap(~sig, ncol = 5) + 
  guides(size = guide_legend(override.aes = list(colour = 'black') ),
         colour = guide_colorbar()) + 
  theme(legend.position = 'top', panel.spacing = unit(2, "lines"),
        plot.margin = margin(6,6,6,6, "pt"))  + 
  labs(size = 'Activity', linewidth = 'Activity', alpha = 'Activity') + 
  coord_trans(clip = 'off')

ggsave('figures/connect_sigs.pdf', width=7, height=10)








# connects
cosmic <- read_delim('data/COSMIC_v3.2_SBS_GRCh37.txt', delim='\t') %>% 
  select(-c('SBS27', 'SBS43', paste0('SBS', 45:60))) %>%
  select(all_of(c('Type',connect_ann$closest_cosmic)))

connect_tau <- read_csv('results/figure_data/connect_sigs_tau.csv') %>% rename(sig=`...1`) %>%
  pivot_longer(-c(sig)) %>%
  pivot_wider(names_from = sig) %>%
  slice(match(cosmic$Type, name))

proxy::simil(select(cosmic, starts_with("SBS")), select(connect_tau, starts_with("C")), 
             by_rows = F, method = 'cosine', diag = T, upper = T) %>%
  as.matrix() -> mat


# cosim heatmap
get_luminance_from_hex <- function(hex_color) {
  rgb <- strtoi(substring(sub("^#", "", hex_color), c(1,3,5), c(2,4,6)), 16)
  (0.299 * rgb[1] + 0.587 * rgb[2] + 0.114 * rgb[3]) / 255
}

bw_col = circlize::colorRamp2(c(0.5, 0.79, 0.8, 1), c('white', '#96D489', '#327180', 'black'))
o = seriate(tibble(mat), method = "BEA_TSP")

sort_index <- apply(t(mat), 2, which.max)

ra = rowAnnotation(
  "Connection Type" = connect_ann$connect_type,
  col = list(
    "Connection Type" = connect_type_col
  ),
  show_legend = TRUE,
  annotation_name_gp=gpar(fontsize = 8),
  width = unit(1, "mm")
)



ca =  columnAnnotation(
    "Etiology" = connect_ann %>% slice(match(rownames(mat), closest_cosmic)) %>% pull(etiology_l1),
    col = list('Etiology'=etiology_col),
    show_legend = TRUE,
    height = unit(5, "cm"),
    annotation_name_gp = gpar(fontsize = 8)
  )

connect_cosmic_sim <- Heatmap(
  t(mat), name = "Cosine\nsimilarity", 
  col=bw_col, 
  #cluster_rows = F,
  #row_split = connect_ann$single_majority, 
  #cluster_columns = dendsort(hclust(dist((mat)))),
  #column_order = get_order(o, 1),
  #row_order = get_order(o, 1), 
  left_annotation = ra,
  row_names_side = "left",
  cluster_rows = F,
  column_order = order(sort_index),
  show_row_dend = F, show_column_dend = F,
  top_annotation = ca,
  width = unit(5, 'cm')
  #cell_fun = function(i, j, x, y, width, height, fill) {
  #  lum <- get_luminance_from_hex(fill)
  #  grid.text(ifelse(names(cosmic)[i+1]==pull(connect_ann[j,],'hungarian_cosmic'), "*", ""),
  #                   x, y, gp = gpar(fontsize = 10, col=ifelse(lum > 0.5, 'black', 'white')), vjust = 0.75)
  #}
)  



pdf('figures/connect_cosmic_sim.pdf', height = 4, width=7)
print(connect_cosmic_sim)
dev.off()


connect_et <- connect %>%
  pivot_longer(-sig) %>%
  separate(name, into=c('Damage', 'Misrepair'), remove = F, sep= '_') %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  group_by(sig, D) %>% mutate(d_size = sum(value, na.rm=T)) %>%
  group_by(sig, M) %>% mutate(m_size = sum(value, na.rm=T)) %>%
  mutate(d_active = d_size > 0.05, m_active = m_size > 0.05) %>%
  left_join(connect_ann, by = join_by(sig==sig)) %>%
  separate_rows(closest_cosmic, sep = ',') %>%
  select(-c(etiology_l1, etiology_l2)) %>%
  left_join(read_csv('data/cosmic_etiologies.csv'), by=join_by(closest_cosmic==sig)) %>%
  ungroup()


p1 <- connect_et %>%
  filter(d_active) %>%
  distinct(Damage, etiology_l1, sig) %>%
  ggplot(aes(x = Damage, fill = etiology_l1)) + 
  geom_bar(position = 'fill') + 
  #scale_fill_manual(values = et_col) + 
  theme(legend.position = 'none') + 
  labs(x = 'Damage', y = 'Count')

p2 <- connect_et %>%
  filter(m_active) %>%
  distinct(Misrepair, etiology_l1, sig)  %>%
  ggplot(aes(x = Misrepair, fill = etiology_l1)) + 
  geom_bar(position = 'fill') + 
  #scale_fill_manual(values = et_col) + 
  labs(x = 'Misrepair', y = 'Count')

cowplot::plot_grid(p1, p2, labels = c('a','b'), label_size = 8)
ggsave('foo.png', width=15, height=4)



connect_examples <- connect %>%
  filter(sig %in% c('C9', 'C14', 'C3', 'C16')) %>%
  pivot_longer(-c(sig)) %>%
  arrange(as.numeric(str_extract(sig, "\\d+$"))) %>%
  mutate(sig = ordered(sig, levels=unique(sig))) %>%
  separate(name, into=c('Damage', 'Misrepair'), remove = F, sep= '_') %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  group_by(sig, D) %>% mutate(d_size = sum(value, na.rm=T)) %>%
  group_by(sig, M) %>% mutate(m_size = sum(value, na.rm=T)) %>%
  #filter(d_size > 0.01, m_size > 0.01) %>%
  mutate(across(value, ~ifelse(d_size<0.05,NA,.))) %>%
  mutate(across(value, ~ifelse(m_size<0.05,NA,.))) %>%
  mutate(across(c(d_size, m_size), ~ifelse(.<0.05,NA,.))) %>%
  # renumber D and M for plotting
  group_by(sig) %>%
  filter(!is.na(value)) %>%
  mutate(D = match(Damage, unique(Damage)),
         M = match(Misrepair, unique(Misrepair)) + (length(unique(Damage)) - length(unique(Misrepair)))/2) %>%
  ggplot() +
  geom_segment(aes(x=0, y=D, xend=1, yend=M, linewidth=value, alpha = value), lineend = 'round') + 
  scale_linewidth(range = c(0,5), breaks = c(0.2, 0.4, 0.6, 0.8)) + 
  scale_size_area(max_size=5, breaks = c(0.2, 0.4, 0.6, 0.8)) + 
  scale_alpha_continuous(breaks = c(0.2, 0.4, 0.6, 0.8)) + 
  geom_point(aes(0, D, size = d_size), colour = 'maroon3') + 
  geom_point(aes(1, M, size = m_size), colour = 'aquamarine3') + 
  geom_text(aes(0, D, label=Damage), nudge_x=-0.2, size=5/.pt) + 
  geom_text(aes(1, M, label=Misrepair), nudge_x=0.2, size=5/.pt) + 
  geom_text(aes(0.5, c(4,2,4,4), label=paste0(connect_type, '\n(', closest_cosmic, ', ', round(hungarian_sim,2), ')' )),
            data=filter(connect_ann, sig %in% c('C3', 'C9', 'C14', 'C16')) , size=6/.pt) +
  theme_void() + 
  facet_wrap(~sig, ncol = 2) + 
  guides(size = guide_legend(override.aes = list(colour = 'black') )) + 
  theme(legend.position = 'top', #strip.text = element_blank(),
        legend.text=element_text(size=6), 
        legend.title = element_text(size=6),
        plot.margin = margin(6,6,6,6, "pt"))  + 
  #lims(y=c(-1,8)) + 
  coord_trans(clip = 'off') + 
  labs(size = 'Connection', linewidth = 'Connection',alpha = 'Connection')

pdf('figures/connect_examples.pdf', width=4, height=4)
plot_grid(connect_examples, labels = 'b', label_size = 8)
dev.off()



connect_cosmic_sim <- (
  gtable_matrix("hm_gtbl", matrix(list(
    grid.grabExpr(draw(connect_cosmic_sim, 
                       heatmap_legend_side = "right",
                       annotation_legend_side = "right")))), 
    unit(1, "null"), unit(1, "null"))
)




pdf('figures/fig_5.pdf', width=7, height=4)
plot_grid(
  plot_grid(connect_examples, connect_cosmic_sim, labels = c('a','b'), label_size = 8), 
   rel_widths=c(1,1.5), ncol=1, labels = c('', 'c'), label_size = 8)
dev.off()


