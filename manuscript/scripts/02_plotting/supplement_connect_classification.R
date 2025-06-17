source('scripts/03_plotting/plotting_header.R')

cosmic <- read_delim('data/COSMIC_v3.2_SBS_GRCh37.txt', delim='\t') %>% 
  select(-c('SBS27', 'SBS43', paste0('SBS', 45:60)))

# connects
connect_tau <- read_csv('results/figure_data/connect_sigs_tau.csv') %>% rename(sig=`...1`) %>%
  pivot_longer(-c(sig)) %>%
  pivot_wider(names_from = sig) %>%
  slice(match(cosmic$Type, name))

# redund info
damage_sim <- cosmic %>% 
  pivot_longer(-c(Type)) %>%
  rename(sig = name, name=Type) %>%
  mutate(trinuc=paste0(str_sub(name, 1,1), str_sub(name, 3,3), str_sub(name, 7,7)),
         subst=paste0(str_sub(name, 3,5))
  ) %>%
  arrange(subst, trinuc) %>%
  mutate(trinuc = ordered(trinuc, levels=unique(trinuc)),
         subst = ordered(subst, levels=unique(subst)),
         sig = ordered(sig, levels=unique(sig))
  ) %>%
  group_by(sig, trinuc) %>%
  summarize(value=sum(value)) %>%
  pivot_wider(names_from = sig) %>%
  select(-c(trinuc)) %>%
  as.matrix() %>%
  lsa::cosine() 

damage_sim[upper.tri(damage_sim, diag=T)] <- NA
damage_redund <- damage_sim %>% 
  as_tibble() %>% 
  mutate(across(everything(), ~(.>0.75))) %>%
  pivot_longer(everything()) %>%
  group_by(name) %>%
  summarize(n_redund=sum(value, na.rm=T))

misrepair_sim <- cosmic %>% 
  pivot_longer(-c(Type)) %>%
  rename(sig = name, name=Type) %>%
  mutate(trinuc=paste0(str_sub(name, 1,1), str_sub(name, 3,3), str_sub(name, 7,7)),
         subst=paste0(str_sub(name, 3,5))
  ) %>%
  arrange(subst, trinuc) %>%
  mutate(trinuc = ordered(trinuc, levels=unique(trinuc)),
         subst = ordered(subst, levels=unique(subst)),
         sig = ordered(sig, levels=unique(sig))
  ) %>%
  group_by(sig,subst) %>%
  summarize(value=sum(value)) %>%
  mutate(base=str_sub(subst,1,1)) %>%
  group_by(sig, base) %>%
  mutate(value = value/sum(value)) %>%
  pivot_wider(names_from = sig) %>%
  ungroup() %>%
  select(-c(subst, base)) %>%
  as.matrix() %>%
  lsa::cosine() 

misrepair_sim[upper.tri(misrepair_sim, diag=T)] <- NA
misrepair_redund <- misrepair_sim %>% 
  as_tibble() %>% 
  mutate(across(everything(), ~(.>0.75))) %>%
  pivot_longer(everything()) %>%
  group_by(name) %>%
  summarize(n_redund=sum(value, na.rm=T))


proxy::simil(select(cosmic, starts_with("SBS")), select(connect_tau, starts_with("C")), 
             by_rows = F, method = 'cosine', diag = T, upper = T) %>%
  as.matrix() -> mat

# create table with matchings
thresh = 0.05
connect_ann <- read_csv('results/figure_data/connect_sigs.csv') %>% 
  rename(sig=`...1`) %>% 
  pivot_longer(-sig) %>%
  separate(name, into=c('damage', 'misrepair'), remove = F, sep= '_') %>%
  group_by(sig,damage) %>% mutate(on_damage = ifelse(value>thresh, damage,NA)) %>%
  group_by(sig,misrepair) %>% mutate(on_misrepair = ifelse(value>thresh, misrepair,NA)) %>%
  group_by(sig) %>% 
  summarize(n_damage = sum(!is.na(unique(on_damage))), 
            n_misrepair = sum(!is.na(unique(on_misrepair))),
            entropy = entropy::entropy(value),
            single_majority = any(value > 0.5) 
            ) %>%
  mutate(
    connect_type = case_when(
      (n_damage==1) & (n_misrepair==1) ~ 'One-one',
      (n_damage>1) & (n_misrepair==1) ~ 'Many-one',
      (n_damage==1) & (n_misrepair>1) ~ 'One-many',
      (n_damage>1) & (n_misrepair>1) ~ 'Many-many',
    )
  ) %>%
  left_join(
    read_csv('results/figure_data/connect_sig_ann.csv') %>%
      select(-`...1`),
    join_by(sig==connect)
  ) %>%
  arrange(as.numeric(str_extract(sig, "\\d+$"))) %>%
  mutate(sig = ordered(sig, levels=unique(sig)))


# cosim heatmap
get_luminance_from_hex <- function(hex_color) {
  rgb <- strtoi(substring(sub("^#", "", hex_color), c(1,3,5), c(2,4,6)), 16)
  (0.299 * rgb[1] + 0.587 * rgb[2] + 0.114 * rgb[3]) / 255
}

bw_col = circlize::colorRamp2(c(0.5, 0.75, 1), c('white', 'lightblue', 'navy'))
o = seriate(tibble(mat), method = "BEA_TSP")

sort_index <- apply(t(mat), 2, which.max)

# Sort the matrix rows
sorted_mat <- mat[, ]


ha <- HeatmapAnnotation(
  'Damage redundancy' = damage_redund$n_redund,
  'Misrepair redundancy' = misrepair_redund$n_redund,
  annotation_legend_param = list(
    'Damage redundancy' = list(at = 0:10, labels = 0:10),
    'Misrepair redundancy' = list(at = 0:10, labels = 0:10)
  )
)

bar_data <- matrix(c(damage_redund$n_redund, misrepair_redund$n_redund), ncol = 2)

ha <- HeatmapAnnotation(
  Redundancy = anno_barplot(bar_data, 
                     beside = TRUE, 
                     attach = TRUE, 
                     gp = gpar(fill = c("maroon3", "aquamarine4")))
)

connect_cosmic_sim <- Heatmap(
  t(mat), name = "Cosine\nsimilarity", 
  col=bw_col, 
  #cluster_rows = F,
  #row_split = connect_ann$single_majority, 
  #cluster_columns = dendsort(hclust(dist((mat)))),
  #column_order = get_order(o, 1),
  #row_order = get_order(o, 1), 
  row_names_side = "left",
  cluster_rows = F,
  column_order = order(sort_index),
  show_row_dend = F, show_column_dend = F,
  cell_fun = function(i, j, x, y, width, height, fill) {
    lum <- get_luminance_from_hex(fill)
    grid.text(ifelse(names(cosmic)[i+1]==pull(connect_ann[j,],'hungarian_cosmic'), "*", ""),
                     x, y, gp = gpar(fontsize = 10, col=ifelse(lum > 0.5, 'black', 'white')), vjust = 0.75)
  }
  #top_annotation = ha
) 


pdf('figures/connect_cosmic_sim.pdf', height = 4, width=7)
print(connect_cosmic_sim)
dev.off()



# from the COSMIC signature perspective
mat %>% 
  unclass() %>% as.data.frame() %>% 
  rownames_to_column('cosmic') %>%
  as_tibble() %>%
  pivot_longer(-cosmic, names_to = 'sig') %>%
  group_by(cosmic) %>%
  filter(value == max(value)) %>%
  left_join(connect_ann) %>%
  mutate(close = value>0.8) %>%
  ggplot(aes(x = connect_type, fill =  close)) +
  geom_bar()
  










cosmic <- read_delim('data/COSMIC_v3.2_SBS_GRCh37.txt', delim='\t') %>% 
  pivot_longer(-c(Type)) %>%
  rename(sig = name, name=Type) %>%
  filter(!sig %in% c('SBS27', 'SBS43', paste0('SBS', 45:60), 'SBS95')) %>%
  mutate(trinuc=paste0(str_sub(name, 1,1), str_sub(name, 3,3), str_sub(name, 7,7)),
         subst=paste0(str_sub(name, 3,5))
  ) %>%
  arrange(subst, trinuc) %>%
  mutate(trinuc = ordered(trinuc, levels=unique(trinuc)),
         subst = ordered(subst, levels=unique(subst)),
         sig = ordered(sig, levels=unique(sig))
  ) 


damage_sim <- cosmic %>% 
  group_by(sig, trinuc) %>%
  summarize(value=sum(value)) %>%
  pivot_wider(names_from = sig) %>%
  select(-c(trinuc)) %>%
  proxy::simil('cosine', by_rows=F) %>%
  as.matrix() 

misrepair_sim <- cosmic %>% 
  group_by(sig,subst) %>%
  summarize(value=sum(value)) %>%
  mutate(base=str_sub(subst,1,1)) %>%
  group_by(sig, base) %>%
  mutate(value = value/sum(value)) %>%
  pivot_wider(names_from = sig) %>%
  ungroup() %>%
  select(-c(subst, base)) %>%
  proxy::simil('cosine', by_rows=F) %>%
  as.matrix() 

damage_sim[upper.tri(damage_sim, diag=T)] <- NA 
damage_redund <- damage_sim %>%
  as_tibble() %>%
  pivot_longer(everything()) %>%
  filter(!is.na(value)) %>%
  group_by(name) %>%
  summarise(n_redund=sum(value>0.8))

misrepair_sim[upper.tri(misrepair_sim, diag=T)] <- NA 
misrepair_redund <- misrepair_sim %>%
  as_tibble() %>%
  pivot_longer(everything()) %>%
  filter(!is.na(value)) %>%
  group_by(name) %>%
  summarise(n_redund=sum(value>0.8))

  

connect_ann %>%
  #left_join(damage_redund, by = join_by(hungarian_cosmic==name)) %>%
  #left_join(misrepair_redund, by = join_by(hungarian_cosmic==name)) %>%
  ggplot(aes(x = single_majority, fill = hungarian_cosmic==closest_cosmic)) +
  geom_bar() + 
  theme_ch() + 
  scale_fill_manual(values = c(`TRUE`='black', `FALSE`='grey')) + 
  labs(x= 'Connect type', y = 'Count', fill ='Matched to COSMIC signature with >0.8 similarity')
  

connect_ann %>%
  left_join(damage_redund, by = join_by(hungarian_cosmic==name)) %>%
  #left_join(misrepair_redund, by = join_by(hungarian_cosmic==name)) %>%
  ggplot(aes(y = n_redund, x = connect_type)) +
  geom_boxplot() +
  geom_point() + 
  theme_ch() + labs(title = 'Matched COSMIC Damage redundancy')




pdf('figures/redund_connect.pdf', h=4, w=7)
connect_ann %>%
  rename(all_close = 'all >= 0.8') %>%
  mutate(n_close = str_count(all_close, ','), 
         n_close = ifelse(all_close == '[]', n_close, n_close+1),
         n_close_cat = ifelse(n_close>1, 2, n_close) 
  ) %>%
    ggplot(aes(x = connect_type, fill = hungarian_sim>0.8)) +
  geom_bar() + 
  #facet_wrap(~n_close_cat) +
  theme_ch() + 
  scale_fill_manual(values = c(`TRUE`='black', `FALSE`='grey')) + 
  labs(x= 'Connect type', y = 'Count', fill ='Matched to COSMIC signature with >0.8 similarity')
dev.off()


connect_activities %>%
  mutate(
    across(-guid, ~.>0.05),
    organ = ann$organ,
    ) %>%
  pivot_longer(-c(guid, organ)) %>%
  left_join(connect_ann, by = join_by(name==sig)) %>%
  filter(value==T) %>%
  group_by(organ, connect_type) %>%
  count()
