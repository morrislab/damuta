source('scripts/03_plotting/plotting_header.R')
library(compositions)
library(vegan)

#library(radEmu) don't load in global scope because it masks dplyr::select

# Tissue specificity plot
# Load figure data
ann = read_csv('data/phg_clinical_ann.csv') 
W = read_csv('results/figure_data/h_W.csv') %>% rename(guid=`...1`) %>% slice(match(ann$guid, guid))
connect_activities = read_csv('results/figure_data/connect_acts.csv') %>% slice(match(ann$guid, guid))
ds = read_delim('results/figure_data/phg_deconstructsigs_activities.csv', delim='\t') %>% rename(guid = Type) %>% slice(match(ann$guid, guid))

W_long <- W %>%
  pivot_longer(-c(guid)) %>%
  rename(sig=name) %>%
  separate(sig, into=c('damage', 'misrepair'), remove = F, sep= '_') 

theta <- W_long %>%
  group_by(guid, damage) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'damage', values_from = value) %>%
  ungroup() %>%
  select(c(guid,paste0('D',1:18))) %>%
  #mutate(across(-c(guid), ~ifelse(.<0.05, NA, .))) %>%
  slice(match(ann$guid, guid))

gamma <- W_long %>%
  group_by(guid, misrepair) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'misrepair', values_from = value) %>%
  ungroup() %>%
  select(c(guid,paste0('M',1:6))) %>%
  #mutate(across(-c(guid), ~ifelse(.<0.05, NA, .))) %>%
  slice(match(ann$guid, guid))


df <- bind_cols(ann, 
               select(ds[match(ann$guid,ds$guid),],SBS1:SBS94),
               select(W[match(ann$guid,W$guid),],D1_M1:D18_M6),
               select(theta[match(ann$guid,theta$guid),],D1:D18),
               select(gamma[match(ann$guid,gamma$guid),],M1:M6),
               select(connect_activities[match(ann$guid,connect_activities$guid),],C1:C30),
) 

# number of misrepair signatures active in >50% of samples
df %>%
  select(guid, organ, D1:M6) %>%
  pivot_longer(-c(guid, organ)) %>%
  mutate(value = ifelse(value<0.05,NA,value)) %>%
  mutate(type = ifelse(str_sub(name, 1,1) == "D", "Damage", "Misrepair")) %>%
  group_by(organ, name, type) %>%
  summarize(prop_active = length(na.omit(value))/n()) %>%
  filter(prop_active > 0.5) %>%
  group_by(organ, type) %>%
  summarize(n_prop_active = length(prop_active)) %>%
  pivot_wider(names_from = type, values_from = n_prop_active) %>%
  mutate(all_active = Misrepair == 6) %>%
  pull(all_active) %>% sum()

# number of damage signatures active in >50% of samples
df %>%
  pivot_longer(D1:M6) %>% 
  mutate(value = ifelse(value<0.05,NA,value)) %>%
  mutate(type = ifelse(str_sub(name, 1,1) == "D", "Damage", "Misrepair")) %>%
  group_by(organ, name, type) %>%
  summarize(prop_active = length(na.omit(value))/n()) %>%
  filter(prop_active > 0.5) %>%
  group_by(organ, type) %>%
  summarize(n_prop_active = length(prop_active)) %>%
  pivot_wider(names_from = type, values_from = n_prop_active) %>%
  pull(Damage) %>% median()

# number of mutations for specifc sigs, organs
df %>%
  filter(organ == 'Skin') %>%
  mutate(
    M6_majority = M6 > 0.5,
    M3_majority = M3 > 0.5,
    M3_M6_majority = M3 + M6 > 0.5
  ) %>%
  select(guid, M6_majority, M3_majority, M3_M6_majority) %>%
  pivot_longer(-guid) %>%
  group_by(name) %>%
  summarize(sum = sum(value, na.rm=T), n = n(), prop = sum/n)

df %>%
  filter(organ == 'Lung') %>%
  mutate(
    M4_majority = M4 > 0.5,
    M5_majority = M5 > 0.5,
    M4_M5_majority = M4 + M5 > 0.5
  ) %>%
  select(guid, M4_majority, M5_majority, M4_M5_majority) %>%
  pivot_longer(-guid) %>%
  group_by(name) %>%
  summarize(sum = sum(value, na.rm=T), n = n(), prop = sum/n)

df %>%
  filter(organ == 'Esophagus') %>%
  select(guid, M1:M6) %>%
  pivot_longer(-guid) %>%
  group_by(guid) %>%
  filter(value == max(value)) %>%
  group_by(name) %>% 
  summarise(sum = n(), n = nrow(.), prop = sum/n)

df %>%
  filter(organ == 'Esophagus') %>%
  mutate(
    M2_majority = M2 > 0.5
  ) %>%
  select(guid, M2_majority) %>%
  pivot_longer(-guid) %>%
  group_by(name) %>%
  summarize(sum = sum(value, na.rm=T), n = n(), prop = sum/n)


df %>%
  filter(organ == 'Esophagus') %>%
  ggplot(aes(x=M2)) +
  geom_histogram() 

ggsave('foo.png')

# Median number of mutations per tissue
tissue_bubble <- df %>%
  mutate(across(D1:M6, ~ifelse(.<0.05,NA,.*n_mut))) %>%
  select(guid, organ, D1:M6) %>%
  pivot_longer(-c(guid, organ)) %>%
  mutate(
    name = ordered(name, levels = c(paste0('D', 1:18), paste0('M', 1:6))),
    type = ifelse(str_sub(name, 1,1) == "D", "Damage", "Misrepair")
  ) %>%
  group_by(organ, name, type) %>%
  summarize(med_mut = median(value, na.rm=T),
            prop_active = length(na.omit(value))/n()*100,
            .groups = 'drop') %>%
  mutate(organ = gsub('_', ' and ', organ)) %>%
  ggplot(aes(x=name, y=organ, size=prop_active, colour=log10(med_mut))) + 
  geom_point() +
  scale_size_continuous(breaks = c(0, 25, 50, 75, 100), 
                        limits = c(5, 100), range = c(0, 3)) +  
  scale_colour_gradientn(colours = c('grey80', 'grey80', 'purple', 'orange', 'red','red'), 
                         guide = guide_colourbar(barwidth=unit(8, 'pt'))) +
  labs(x ="", y = '', 
       size = 'Percentage of samples\nwith active signature', 
       colour = 'Log10 median\n# of mutations ') + 
  guides(size = guide_legend(override.aes = list(shape = 21, fill='grey', colour = 'grey'))) +
  theme_ch() + 
  rotate_x_text(45) + 
  facet_grid(~type, space='free', scales = 'free_x') + 
  theme(legend.position = 'right',
        #legend.direction = 'horizontal',
        #legend.box = 'vertical',
        legend.key.height = unit(0.5, 'cm'),
        legend.margin = margin(0, 0, 0, 0, "in"),
        legend.spacing = unit(5, "pt"),
        plot.margin = margin(0,0,0,0, "in")
  ) 


#ggsave('figures/tissue_bubble.pdf', width=4, height=4)


# NMI
############

dist_method = 'cosine'
hclust_method = 'complete'
label_col = 'organ'

dend_damage <- df %>%
  select(D1:D18) %>%
  #mutate(across(everything(), ~ifelse(.<0.05,0,.))) %>%
  proxy::dist(method = dist_method) %>%
  hclust(hclust_method) %>%
  dendsort::dendsort()

dend_misrepair <- df %>%
  select(M1:M6) %>%
  #mutate(across(everything(), ~ifelse(.<0.05,0,.))) %>%
  proxy::dist(method = dist_method) %>%
  hclust(hclust_method) %>%
  dendsort::dendsort()

dend_dm <- df %>%
  select(D1:M6) %>%
  #mutate(across(everything(), ~ifelse(.<0.05,0,.))) %>%
  proxy::dist(method = dist_method) %>%
  hclust(hclust_method) %>%
  dendsort::dendsort()


dend_connect <- df %>%
  select(starts_with('C', ignore.case=F)) %>%
  proxy::dist(method = dist_method) %>%
  hclust(hclust_method) %>%
  dendsort::dendsort()

dend_cosmic <- df %>%
  select(starts_with('SBS')) %>%
  #mutate(across(everything(), ~ifelse(.<0.05,0,.))) %>%
  proxy::dist(method = dist_method) %>%
  hclust(hclust_method) %>%
  dendsort::dendsort()


clustering_metrics <- tibble(k = 2:200) %>%
  rowwise() %>%
  mutate(
    NMI_Misrepair =aricode::NMI(cutree(dend_misrepair,k), ann[[label_col]], variant='sqrt'),
    NMI_Damage = aricode::NMI(cutree(dend_damage,k),       ann[[label_col]], variant='sqrt'),
    "NMI_Both" = aricode::NMI(cutree(dend_dm,k),       ann[[label_col]], variant='sqrt'),
    #NMI_W = NMI(cutree(dend_w,k),       ann[[label_col]], variant='sqrt'),
    NMI_Connect = aricode::NMI(cutree(dend_connect,k),     ann[[label_col]], variant='sqrt'),
    NMI_COSMIC = aricode::NMI(cutree(dend_cosmic,k),       ann[[label_col]], variant='sqrt'),
    NMI_Random = aricode::NMI(sample(1:k, length(ann[[label_col]]), replace = T), ann[[label_col]], variant='sqrt')
  ) %>% 
  pivot_longer(-k) %>%
  separate(name, into = c('metric', 'Signature Type'), sep = '_') %>%
  pivot_wider(names_from = metric) 


nmi <- clustering_metrics %>% 
  filter(k==length(unique(ann$organ))) %>%
  filter(`Signature Type` != 'Connect') %>%
  ggplot(aes(x = reorder(`Signature Type`, NMI), y = NMI, fill = `Signature Type`)) + 
  geom_col() + 
  theme_ch() + 
  labs(x = '', y = 'Tissue specificity (NMI)') + 
  scale_fill_manual(
    values =  c('Damage' = 'maroon3', 
               'Misrepair' = 'aquamarine3',
               'Both' = 'coral',
               'COSMIC'= 'dodgerblue', 
               'Random' = 'grey')
  ) + 
  theme(
    legend.position = 'none',
    plot.margin = margin(0.3,0.1,0.1,0.1, "in")
  )

clustering_metrics %>% 
  filter(`Signature Type` %in% c('Damage/Misrepair concat', 'Damage')) %>%
  rename(group = `Signature Type`) %>%
  compare_means(NMI ~ group, data=., method = 't.test', paired = T)


clustering_metrics %>%
  mutate(
    `Signature Type` = factor(`Signature Type`,
                              levels = c('Damage/Misrepair concat', 'Damage', 'Misrepair', 'Connect', 'COSMIC'))
  ) %>%
  ggplot(aes(x = k, y = NMI, colour = `Signature Type`)) + 
  geom_point(size = 0.5) + 
  geom_path() + 
  theme_ch() + 
  geom_vline(xintercept = length(unique(ann[[label_col]])), linewidth = 1, linetype = 'dashed', colour = 'grey') +
  scale_colour_manual(values = c('Damage' = 'maroon3', 
                                 'Misrepair' = 'aquamarine3', 
                                 'Connect' = 'darkgoldenrod3',
                                 'Damage + Misrepair' = 'orange',
                                 'W' = 'darkgoldenrod4',
                                 'COSMIC'= 'dodgerblue',
                                 'Random' = 'grey')) +
  labs(x = "Number of clusters from dendrogram cut",
       y='Normalized mutual information with organ clusters',
       colour = 'Feature set') + 
  theme(legend.position = 'right')

ggsave('figures/nmi_extended.pdf', height = 4, width = 6)




# subclone annotation file is derived from pcawg restricted-access data and not included in the repo   
ann = read_csv('data/pcawg_subclone_ann.csv') %>%
  mutate(clone = as.numeric(str_extract(guid, "(?<=_S)\\d+")))
W = read_csv('~/damuta-figures/figure_data/subclone_h_W.csv') %>% 
  rename(guid=`...1`) %>%
  slice(match(ann$guid, guid))

colnames(W) = c('guid', paste0('D', rep(1:18, each=6), '_M', rep(1:6, times=18)))

W_long <- W %>%
  pivot_longer(-c(guid)) %>%
  rename(sig=name) %>%
  separate(sig, into=c('damage', 'misrepair'), remove = F, sep= '_') 

theta <- W_long %>%
  group_by(guid, damage) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'damage', values_from = value) %>%
  ungroup() %>%
  select(c(guid,paste0('D',1:18))) %>%
  slice(match(ann$guid, guid))

gamma <- W_long %>%
  group_by(guid, misrepair) %>%
  summarize(value = sum(value, na.rm = T)) %>%
  pivot_wider(names_from = 'misrepair', values_from = value) %>%
  ungroup() %>%
  select(c(guid,paste0('M',1:6))) %>%
  slice(match(ann$guid, guid))

# Get tibbles for first and last clones
first <- ann %>%
  subset(duplicated(index, fromLast=T)) %>%
  filter(clone==1) %>%
  arrange(index) %>%
  bind_cols(select(theta[match(.$guid,theta$guid),],D1:D18),
            select(gamma[match(.$guid,gamma$guid),],M1:M6)
  ) #%>% mutate(across(D1:M6, ~ifelse(. < 0.05, 0, .)))

last <- ann %>%
  subset(duplicated(index)) %>%
  group_by(index) %>%
  filter(clone == max(clone)) %>%
  ungroup() %>%
  arrange(index) %>% 
  bind_cols(select(theta[match(.$guid,theta$guid),],D1:D18),
            select(gamma[match(.$guid,gamma$guid),],M1:M6)
  ) #%>% mutate(across(D1:M6, ~ifelse(. < 0.05, 0, .)))

stopifnot(all(first$index == last$index))




# RadEmu to test subclonal changes in each organ
fl <- bind_rows(first, last) %>%
  bind_rows(bind_rows(first, last) %>% mutate(organ = 'Pan-cancer')) %>%
  mutate(
    organ = factor(organ),
    is_first_clone = factor(clone==1),
    name = paste0(guid, '_', organ)
  ) %>%
  column_to_rownames('name') 


# skip slow simulation if already run
if (!file.exists('results/figure_data/damage_emu_act.rds') & !file.exists('results/figure_data/misrepair_emu_act.rds')) {

  damage_emu_act <- list()
  misrepair_emu_act <- list()

  for (o in unique(fl$organ)) {  
    print(o)
    fl_o <- fl %>% filter(organ == o)
    damage_emu_act[[o]] <- radEmu::emuFit(formula = ~ is_first_clone, 
                        data = fl_o, 
                        Y = dplyr::select(fl_o, D1:D18),
                        run_score_tests = T,
                        test_kj = data.frame(k=2, j=1:18)
                        ) 
    misrepair_emu_act[[o]] <- radEmu::emuFit(formula = ~ is_first_clone, 
                        data = fl_o, 
                        Y = dplyr::select(fl_o, M1:M6),
                        run_score_tests = T,
                        test_kj = data.frame(k=2, j=1:6)
                        ) 
  }
    # save result
    write_rds(damage_emu_act, 'results/figure_data/damage_emu_act.rds')
    write_rds(misrepair_emu_act, 'results/figure_data/misrepair_emu_act.rds')

} else {
  damage_emu_act <- read_rds('results/figure_data/damage_emu_act.rds')
  misrepair_emu_act <- read_rds('results/figure_data/misrepair_emu_act.rds')
}


# check convergence
sapply(damage_emu_act, function(i) i$estimation_converged)
sapply(misrepair_emu_act, function(i) i$estimation_converged)

# Combine all damage_emu coefficients
damage_coefs <- bind_rows(
  lapply(damage_emu_act, function(i) i$coef),
  .id = "organ"
  ) %>% as_tibble() %>%
  mutate(Type = 'Damage')

misrepair_coefs <- bind_rows(
  lapply(misrepair_emu_act, function(i) i$coef),
  .id = "organ"
  ) %>% as_tibble() %>%
  mutate(Type = 'Misrepair')


subclone_bubble <- bind_rows(damage_coefs, misrepair_coefs) %>%
  mutate(organ = gsub('_', ' and ', organ)) %>%
  rename(p_val = pval, log_fold_change = estimate, name = category) %>%
  mutate(
    pval_bh = p.adjust(p_val, method = 'BH'),
    pval_bonferroni = p.adjust(p_val, method = 'bonferroni')
  ) %>%
  filter(pval_bh < 0.05) %>%
  group_by(organ) %>%
  mutate(n = n()) %>%
  arrange(n) %>%
  ungroup() %>%
  mutate(organ = ordered(organ, levels = c(setdiff(unique(organ), "Pan-cancer"), "Pan-cancer"))) %>%
  mutate(name = ordered(name, levels = c(
    'D17', 'D1', 'D7', 'D6', 'D3', 'D15', 'D8', 'D12', 'D13', 'D16', 'D9','D11', 
    'D18', 'D4', 'D14',  'D5', 'D10', 'D2', 'M3', 'M1', 'M6', 'M2', 'M5', 'M4'))) %>%
  ggplot(aes(x = name, y = organ, colour = log_fold_change, size = -log10(pval_bh))) +
  geom_point() +
  theme_pubr() +
  facet_grid(~Type, scales = 'free_x', space = 'free_x') +
  scale_colour_gradient2(
    low = 'orangered', high = 'cornflowerblue', 
    mid = 'white', midpoint = 0, 
    limits = c(-.3,.3), oob = scales::squish
  ) +
  theme_ch() +
  rotate_x_text(45) +
  theme(
    strip.background = element_blank(),
    legend.margin = margin(0, 0, 0, 0),
    legend.position = 'top',
    legend.direction = 'horizontal',
    legend.box = 'vertical',
    plot.margin = unit(c(0.2, 0, 0.1, 0), 'cm'),
    legend.key.height = unit(0.3, 'cm')
  ) +
  scale_size_continuous(range=c(0,6), breaks = c(2, 5, 10)) + 
  guides(size = guide_legend(override.aes = list(stroke = 1, fill = "grey"))) +
  labs(
    colour = 'Mean log-fold change from\nclonal to subclonal mutations    ',
    x = 'Signature', 
    y = '',
    size = '-log10(p-value)'
  ) + 
  coord_cartesian(clip = "off", expand = TRUE) +
  scale_x_discrete(expand = expansion(add = 0.8))


# pairwise changes in damage/misrepair for those with a difference
df <- as_tibble(last %>% select(D1:M6) - first %>% select(D1:M6)) %>%
  mutate(index = first$index, organ = first$organ) %>%
  filter(organ %in% c('Ovary', 'Pancreas', 'Colorectal', 'Bone_SoftTissue', 'Breast'))  %>%
  mutate(organ = gsub('_', ' and ', organ)) %>%
  mutate(organ = ordered(organ, levels = c('Ovary', 'Breast', 'Colorectal', 'Pancreas',  'Bone and SoftTissue'))) 

# Contour plots
subclone_contour <- df %>%
  ggplot(aes(x = M3, y = M4, colour = organ)) +
  geom_hline(yintercept = 0, colour = 'grey40') +
  geom_vline(xintercept = 0, colour = 'grey40') +
  geom_point(size = 0.02) +
  geom_density2d(linewidth = 0.02) +
  #geom_smooth(method = 'lm', se = F, aes(colour = organ)) +
  #stat_cor(label.y.npc = 'top', size = 6/.pt) +
  facet_wrap(~organ, ncol=1) +
  scale_colour_manual(values = lut) + 
  labs(colour = 'Organ', 
    x = 'M3 activity change\nfrom clonal to subclonal', 
    y = 'M4 activity change from clonal to subclonal') + 
  geom_text(
    data = tibble(
      organ = ordered(
        c('Ovary', 'Ovary', 'Ovary', 'Ovary')
      ),
      x = c(-0.2, -0.2, 0.25, 0.25),
      y = c(0.3, 0.2, -0.17, -0.27),
      label = c('M4↑', 'M3↓', 'M4↓', 'M3↑')
    ),
    aes(x = x, y = y, label = label),
    color = "black", size = 6/.pt
  ) +
  lims(x = c(-0.3, 0.3), y = c(-0.3, 0.3)) +
  theme_ch() + 
  theme(
    legend.position = 'none', 
    strip.background = element_blank(),
    plot.margin = unit(c(0.1, 0, 0.1, 0.1), 'cm')
  ) + 
  coord_trans(clip = F)
# Create contingency tables for each organ and test distribution in quadrants
# Calculate quadrant counts and chi-square test
# Create tile plot for each organ's quadrant distribution
subclone_contingency <- df %>%
  group_by(organ) %>%
  summarise(
    q1 = sum(M3 > 0 & M4 > 0),  # top right
    q2 = sum(M3 < 0 & M4 > 0),  # top left
    q3 = sum(M3 < 0 & M4 < 0),  # bottom left
    q4 = sum(M3 > 0 & M4 < 0),  # bottom right
    chisq_pval = chisq.test(c(q1, q2, q3, q4))$p.value
  ) %>%
  ungroup() %>%
  mutate(
    chisq_pval_bonferroni = p.adjust(chisq_pval, method = 'bonferroni'),
    pval_display = case_when(
      chisq_pval_bonferroni < 0.001 ~ "<0.001", 
      chisq_pval_bonferroni < 0.01 ~ "<0.01", 
      chisq_pval_bonferroni < 0.05 ~ "<0.05", 
      TRUE ~ as.character(round(chisq_pval_bonferroni, 3)),
    ),
    pval_display = paste0(organ, ' chi-sq p', pval_display)
  ) %>%
  arrange(organ) %>%
  mutate(pval_display = ordered(pval_display, levels = unique(pval_display))) %>%
  pivot_longer(cols = c(q1, q2, q3, q4), names_to = "quadrant", values_to = "count") %>%
  mutate(
    x = case_when(
      quadrant %in% c("q1", "q4") ~ "M3↑",
      quadrant %in% c("q2", "q3") ~ "M3↓"
    ),
    y = case_when(
      quadrant %in% c("q1", "q2") ~ "M4↑",
      quadrant %in% c("q3", "q4") ~ "M4↓"
    )
  ) %>%
  group_by(organ) %>%
  mutate(prop = count / sum(count)) %>%
  ggplot(aes(x = reorder(x, -prop), y = reorder(y, prop), fill = prop)) +
  geom_tile() +
  geom_text(aes(label = count), color = "black", size = 10/.pt) +
  facet_wrap(~pval_display, ncol = 1, scales = 'free_x') +
  scale_fill_gradient(low = "white", high = "cornflowerblue") +
  theme_ch() +
  theme(
    strip.background = element_blank(),
    axis.title = element_text(size = 10),
    legend.position = "none",
    plot.margin = unit(c(0.1, 0.1, 0.1, 0), 'cm')
  ) +
  labs(x = "", y = "") + 
  coord_trans(clip = F) 




# assemble

t <- cowplot::plot_grid(tissue_bubble + theme(plot.margin = unit(c(0.1, 0.1, 0.1, -0.2), 'cm')), nmi + rotate_x_text(45), rel_widths = c(2.5, 1), nrow = 1, labels = c('a', 'b'), label_size = 8)
l <- cowplot::plot_grid(subclone_contour, subclone_contingency, ncol = 2, align = 'hv', axis = 'lr', labels = c('d', 'e'), label_size = 8)
b <- cowplot::plot_grid(subclone_bubble + theme(plot.margin = unit(c(0.1, 0.1, 0.4, -0.2), 'cm')), l, ncol = 2, rel_widths = c(2.3, 1), labels = c('c'), label_size = 8)
cowplot::plot_grid(t, b, rel_heights = c(1, 1.4), nrow = 2) 
ggsave('figures/fig_6.pdf', height = 9, width = 7)




