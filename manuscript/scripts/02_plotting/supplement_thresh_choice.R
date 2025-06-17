source('plotting_header.R')

ds = read_delim('results/figure_data/phg_all_deconstructsigs_activities.csv', delim='\t') %>% rename(guid = Type) 
ann = read_csv('data/phg_clinical_ann.csv') %>% rename(guid = `...1`)
C <- read_csv('results/figure_data/connect_acts.csv') %>% rename(guid=`...1`) %>% slice(match(ann$guid, guid))
W = read_csv('results/figure_data/h_W.csv') %>% rename(guid=`...1`)
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


# Distribution of total activity per sample
tot <- bind_rows(
  (ds %>%
    pivot_longer(-c(guid)) %>%
    group_by(guid) %>%
    summarize(value = sum(value)) %>%
    mutate(type='COSMIC')),
  (theta %>%
     group_by(guid) %>%
     pivot_longer(-c(guid)) %>%
     filter(value >0.05) %>%
     summarize(value = sum(value)) %>%
     mutate(type='Damage')),
  (gamma %>%
     group_by(guid) %>%
     pivot_longer(-c(guid)) %>%
     filter(value >0.05) %>%
     summarize(value = sum(value)) %>%
     mutate(type='Misrepair')),
  (W %>%
     group_by(guid) %>%
     pivot_longer(-c(guid)) %>%
     filter(value >0.01) %>%
     summarize(value = sum(value)) %>%
     mutate(type='W')),
  (C %>%
     group_by(guid) %>%
     pivot_longer(-c(guid)) %>%
     filter(value >0.05) %>%
     summarize(value = sum(value)) %>%
     mutate(type='Connect')),
) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
  ggplot(aes(x=value, fill = type)) + 
  geom_histogram(bins = 30) + 
  facet_wrap(~type, ncol=1, scales = 'free_y') + 
  theme_ch() + 
  scale_fill_manual(values = c(COSMIC = 'dodgerblue', 
                               Connect = "goldenrod1",
                               W = 'goldenrod3',
                               Damage = 'maroon3', 
                               Misrepair = 'aquamarine3')) + 
  labs(x = 'Total activity', y = "Number of samples", fill = 'Signature type') + 
  theme(legend.position = 'none')

# Distribution of number of active signatures per sample
count <- bind_rows(
  (ds %>%
     pivot_longer(-c(guid)) %>%
     group_by(guid) %>%
     summarize(value = sum(value>0.05)) %>%
     mutate(type='COSMIC')),
  (theta %>%
     pivot_longer(-c(guid)) %>%
     group_by(guid) %>%
     summarize(value = sum(value>0.05)) %>%
     mutate(type='Damage')),
  (gamma %>%
     pivot_longer(-c(guid)) %>%
     group_by(guid) %>%
     summarize(value = sum(value>0.05)) %>%
     mutate(type='Misrepair')),
  (W %>%
     pivot_longer(-c(guid)) %>%
     group_by(guid) %>%
     summarize(value = sum(value>0.1)) %>%
     mutate(type='W')),
  (C %>%
     pivot_longer(-c(guid)) %>%
     group_by(guid) %>%
     summarize(value = sum(value>0.05)) %>%
     mutate(type='Connect')),
) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
  ggplot(aes(x=value, fill = type)) + 
  geom_histogram(bins = 30) + 
  facet_wrap(~type, ncol=1, scales = 'free_y') + 
  theme_ch() + 
  scale_fill_manual(values = c(COSMIC = 'dodgerblue', 
                               Connect = "goldenrod1",
                               W = 'goldenrod3',
                               Damage = 'maroon3', 
                               Misrepair = 'aquamarine3')) + 
  labs(x = 'Number of active signatures per sample', y = "Number of samples", fill = 'Signature type')+
  theme(legend.position = 'none')



count <- bind_rows(
  ds %>% 
    pivot_longer(-guid) %>% 
    mutate(type='COSMIC'),
  theta %>% 
    pivot_longer(-guid) %>% 
    mutate(type='Damage'),
  gamma %>% 
    pivot_longer(-guid) %>% 
    mutate(type='Misrepair'),
  W %>% 
    pivot_longer(-guid) %>% 
    mutate(type='W'),
  C %>% 
    pivot_longer(-guid) %>% 
    mutate(type='Connect'),
  
) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
  ggplot(aes(x=value, fill = type)) + 
  geom_histogram(bins = 100) + 
  facet_wrap(~type, ncol=1, scales = 'free_y') + 
  scale_y_log10() + 
  theme_ch() + 
  scale_fill_manual(values = c(COSMIC = 'dodgerblue', 
                               Connect = "goldenrod1",
                               W = 'goldenrod3',
                               Damage = 'maroon3', 
                               Misrepair = 'aquamarine3')) + 
  labs(x = 'Activity', y = "Count", fill = 'Signature type')+
  theme(legend.position = 'none')
  


# CDF of median number of active signatures and threshold
get_med <- function(acts) {
  acts %>% 
    pivot_longer(-c(guid)) %>%
    mutate(`0.5` = value > 0.5, `0.3` = value > 0.3, `0.1` = value > 0.1, 
           `0.05` = value > 0.05, `0.01` = value > 0.01, `0.005` = value > 0.005, 
           `0.001` = value > 0.001, `0` = value > 0) %>%
    group_by(guid) %>% 
    summarise(`0.5` = sum(`0.5`), `0.3` = sum(`0.3`), `0.1` = sum(`0.1`), 
              `0.05` = sum(`0.05`), `0.01` = sum(`0.01`), `0.005` = sum(`0.005`), 
              `0.001` = sum(`0.001`), `0` = sum(`0`)) %>%
    summarise(across(starts_with("0."), ~median(.))) %>%
    pivot_longer(everything()) 
    
}

thresh_choice <- bind_rows(
  ds %>% 
    get_med() %>%
    mutate(type = 'COSMIC'), 
  theta %>% 
    get_med() %>%
    mutate(type = 'Damage'),
  gamma %>% 
    get_med() %>%
    mutate(type = 'Misrepair'),
  W %>% 
    get_med() %>%
    mutate(type = 'W'),
  C %>% 
    get_med() %>%
    mutate(type = 'Connect'),
  
) %>%
  filter(!(type == 'COSMIC' & name %in% c(0, 0.001, 0.005, 0.01))) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
ggplot(aes(x = name, y = value, colour = type, group=type)) +
  geom_path() +
  geom_point(size = 0.8) +
  theme_ch() + 
  facet_wrap(~type, scales = 'free_y', ncol=1) + 
  labs(x = 'Activity threshold', colour = 'Signature type', 
       y = 'Median number of active signatures per sample') + 
  scale_colour_manual(values = c(COSMIC = 'dodgerblue', 
                                Connect = "goldenrod1",
                                W = 'goldenrod3',
                                Damage = 'maroon3', 
                                Misrepair = 'aquamarine3')) + 
  theme(legend.position = 'none') + 
  rotate_x_text(45)
  



#pdf('../plots/thresh_choice.pdf', width=2, height=5)
#plot_grid(thresh_choice, label_size = 8)
#dev.off()

pdf('figures/thresh_choice.pdf', width=5.5, height=4)
plot_grid(thresh_choice, tot, count, labels = c('a','b','c'),
          align='h', 
          rel_widths = c(1,1,1.2), ncol=3,label_size = 8)
dev.off()



(W %>%
    pivot_longer(-c(guid)) %>%
    group_by(guid) %>%
    summarize(value = sum(value>0.01)) %>%
    mutate(type='W')) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
  summarize(median(value))

(W %>%
    pivot_longer(-c(guid)) %>%
    group_by(guid) %>%
    summarize(value = sum(value>0.05)) %>%
    mutate(type='W')) %>%
  mutate(type = ordered(type, levels = unique(type))) %>%
  summarize(median(value))
