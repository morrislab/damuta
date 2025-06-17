source('scripts/03_plotting/plotting_header.R')

df = read_csv('results/figure_data/class_balance.csv')
colSums(select(df, -c(`...1`)))
round(colSums(select(df, -c(`...1`))) / nrow(df), 2) * 100

df = read_csv('data/DDR_pthw.csv')
df %>%
  group_by(pathway) %>%
  summarise(n())


# how many C > T?
phi <- read_csv('results/figure_data/h_phi.csv') %>% 
  rename(sig=`...1`) %>%
  pivot_longer(cols = -c(sig)) %>% 
  mutate(base = substr(name, 2,2)) %>%
  arrange(base, name) 

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
         sig = ordered(sig, levels=unique(sig)),
         base = substr(subst, 1,1)
  ) 


df <- bind_rows(
  cosmic %>% 
    group_by(sig, base) %>%
    summarize(value = sum(value))%>%
    mutate(
      type = 'COSMIC', 
      sig = as.character(sig)),
  phi %>%
    group_by(sig, base) %>%
    summarize(value = sum(value)) %>%
    mutate(
      type = 'Damage',
      sig = as.character(sig))
) %>%
  group_by(sig, type) %>%
  summarize(ct_ratio = log2(first(value)/last(value)))

p <- df %>%
  ggplot(aes(x = ct_ratio)) + 
  geom_histogram(bins = 30, aes(fill = type)) + 
  geom_text(aes(label = sig, x = ct_ratio,
                y = c(2,2,7,6,6.4)), data = . %>% filter(sig %in% c('SBS3', 'SBS5', 'SBS40', 'SBS2', 'SBS17b')),
            size = 6/.pt, nudge_x = 0.5, angle=45) + 
  #geom_vline(aes(xintercept =ifelse(sig %in% c('SBS3', 'SBS5', 'SBS40', 'SBS2', 'SBS17b'), ct_ratio, NA)),
   #              linetype = 'dashed', colour ='grey') + 
  geom_segment(
    data = data.frame(x = c(-.5,-.5,.5,.5),
                     xend = c(-4,-4,4,4),
                     y = c(8.1,2.1,8.1,2.1),
                     yend = c(8.1,2.1,8.1,2.1), 
                     type = c('COSMIC', 'Damage','COSMIC', 'Damage')),
    aes(x = x, xend = xend, y = y, yend = yend),
    arrow = arrow(length = unit(0.2, "cm")),
    color = "black"
  ) +
  geom_text(
    data = data.frame(x = c(-2,-2,2,2),
                      y = c(8.4,2.2,8.4,2.2),
                     type = c('COSMIC', 'Damage','COSMIC', 'Damage'),
                     label = c("T-biased","T-biased", "C-biased", "C-biased")),
    aes(x = x, y = y, label = label),
    color = "black",
    size = 6/.pt
  ) +
  theme_ch() + 
  facet_wrap(~type, ncol=2, scales = 'free_y') + 
  scale_fill_manual(values = activity_col) + 
  theme(legend.key.size = unit(6, 'pt')) + 
  labs(x = 'log2(C/T)', y = 'Count', fill = 'Signature set') +  
  coord_trans(clip='off')

plot_grid(p, NULL, rel_widths = c(1,0.05))
ggsave('figures/ct_balance.pdf', height = 4, width = 7)


df %>%
  mutate(balance = case_when(
    ct_ratio > 1 ~ 'C-biased',
    ct_ratio < -1 ~ 'T-biased',
    .default = 'Balanced'
  )) %>%
  group_by(type, balance) %>%
  summarize(n = n())
  

eta <- read_csv('results/figure_data/h_eta.csv')

eta %>%
  pivot_longer(-label) %>%
  mutate(
    base = substr(name, 1,1),
    active = value > 0.05) %>%
  group_by(label, base) %>%
  summarize(n_active = sum(active)) %>%
  group_by(base) %>%
  summarize(n_sig = sum(n_active == 3))

