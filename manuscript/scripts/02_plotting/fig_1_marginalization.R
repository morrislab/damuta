source('scripts/02_plotting/plotting_header.R')
library(ggtext)

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

# Example marginalization
cosmic_marg <- cosmic %>% 
  filter(sig %in% c('SBS4', 'SBS22', 'SBS2', 'SBS13')) %>%
  mutate(sig = ordered(sig, levels=c('SBS4', 'SBS22', 'SBS2', 'SBS13'))) %>%
  group_by(sig) %>%
  mutate(ylab = 0.9*max(value)) %>%
  mutate(name = str_replace_all(name, '(.).(C>A).(.)', "\\1[<span style='color:lightblue;'>C>A</span>]\\3")) %>%
  mutate(name = str_replace_all(name, '(.).(C>G).(.)', "\\1[<span style='color:black;'>C>G</span>]\\3")) %>%
  mutate(name = str_replace_all(name, '(.).(C>T).(.)', "\\1[<span style='color:red;'>C>T</span>]\\3")) %>%
  mutate(name = str_replace_all(name, '(.).(T>A).(.)', "\\1[<span style='color:lightgreen;'>T>A</span>]\\3")) %>%
  mutate(name = str_replace_all(name, '(.).(T>C).(.)', "\\1[<span style='color:grey;'>T>C</span>]\\3")) %>%
  mutate(name = str_replace_all(name, '(.).(T>G).(.)', "\\1[<span style='color:pink;'>T>G</span>]\\3")) %>%
  mutate(name = ordered(name, levels = unique(name))) %>%
  ggplot(aes(x = name, y=value, fill=subst)) + 
  geom_bar(stat='identity') +
  facet_wrap(~sig, scales = 'free_y', ncol=1) + 
  theme_pubr() + rotate_x_text(hjust=0) + 
  scale_fill_manual(values = c('lightblue','black', 'red', 'lightgreen', 'grey', 'pink')) + 
  geom_text(aes(x=5,y=ylab,label=sig), 
            data = . %>% group_by(sig) %>% slice(1),
            size = (6/.pt)) +
  guides(fill=FALSE) + 
  labs(x='Mutation Type', y = 'Proportion', title = 'COSMIC') +
  theme_ch() + rotate_x_text(hjust=0) + 
  theme(strip.background = element_blank(), strip.text = element_blank()) +
  theme(axis.text.x = element_markdown(size = 3),
        axis.text.y = element_text(size=6),
        plot.margin = unit(c(6,0,6,0), "pt")
        ) + 
  scale_y_continuous(n.breaks = 3)
        

damage_marg <- cosmic %>% 
  filter(sig %in% c('SBS4', 'SBS22', 'SBS2', 'SBS13')) %>%
  mutate(sig= ordered(sig, levels =  c('SBS4', 'SBS22', 'SBS2', 'SBS13'))) %>%
  group_by(sig, trinuc) %>%
  summarize(value=sum(value)) %>%
  mutate(base = str_sub(trinuc,2,2)) %>%
  mutate(trinuc = str_replace_all(trinuc, '(.)(C)(.)', "\\1<span style='color:maroon2;'>C</span>\\3")) %>%
  mutate(trinuc = str_replace_all(trinuc, '(.)(T)(.)', "\\1<span style='color:maroon4;'>T</span>\\3")) %>%
  mutate(trinuc = ordered(trinuc, levels = unique(trinuc))) %>%
  group_by(sig) %>%
  mutate(ylab = 0.9*max(value)) %>%
  ggplot(aes(x = trinuc, y=value, fill=base)) + 
  geom_bar(stat='identity') +
  scale_fill_manual(values=c('maroon2', 'maroon4')) + 
  guides(fill=FALSE) + 
  geom_text(aes(x=3,y=ylab,label=sig), 
            data = . %>% group_by(sig) %>% slice(1),
            size = (6/.pt)) + 
  facet_wrap(~sig, scales = 'free_y', ncol=1) +   
  labs(x = 'Trinucleotide Context', y='Proprtion', title = 'Marginalized for trinucleotide context') + 
  theme_ch() + rotate_x_text(hjust=0) + 
  theme(strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.y = element_text(size=6),
        axis.text.x = element_markdown(),
        plot.margin = unit(c(6,0,6,0), "pt")
        )  + 
  scale_y_continuous(n.breaks = 3)
  

misrepair_marg <- cosmic %>% 
  filter(sig %in% c('SBS4', 'SBS22', 'SBS2', 'SBS13')) %>%
  mutate(sig= ordered(sig, levels =  c('SBS4', 'SBS22', 'SBS2', 'SBS13'))) %>%
  group_by(sig,subst) %>%
  summarize(value=sum(value)) %>%
  mutate(base=str_sub(subst,1,1)) %>%
  group_by(sig, base) %>%
  mutate(value = value/sum(value)) %>%
  mutate(subst = str_replace_all(subst, 'C>(.)', "<span style='color:aquamarine2;'>C</span>>\\1")) %>%
  mutate(subst = str_replace_all(subst, 'T>(.)', "<span style='color:aquamarine4;'>T</span>>\\1")) %>%
  ggplot(aes(x = subst, y=value, fill=base)) + 
  geom_bar(stat='identity') +
  scale_fill_manual(values=c('aquamarine2', 'aquamarine4')) + 
  guides(fill=FALSE) +
  facet_wrap(~sig, ncol=1) +  
  labs(x = 'Substitution', y='', title = 'Marginalized for substitution') + 
  theme_ch() + rotate_x_text(hjust=0) + 
  theme(strip.background = element_blank(), 
        strip.text = element_blank(),
        axis.text.x = element_markdown(),
        axis.text.y = element_text(size=6),
        plot.margin = unit(c(6,0,6,0), "pt")
  ) + 
  scale_y_continuous(n.breaks = 3)

plot_grid(cosmic_marg, 
          plot_grid(damage_marg, misrepair_marg, ncol = 2, rel_widths = c(3,1.5)), 
          rel_heights = c(1, 1), ncol = 1, labels = 'c', label_size = 8)
ggsave('figures/sigs_marg.pdf', width=4.4, height=4.2)


# Histogram of redundancy
sbs_sim <- cosmic %>%
  pivot_wider(names_from = sig) %>%
  select(-c(name, trinuc, subst)) %>%
  as.matrix() %>%
  lsa::cosine()

damage_sim <- cosmic %>% 
  group_by(sig, trinuc) %>%
  summarize(value=sum(value)) %>%
  pivot_wider(names_from = sig) %>%
  select(-c(trinuc)) %>%
  as.matrix() %>%
  lsa::cosine()

misrepair_sim <- cosmic %>% 
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

median(sbs_sim[upper.tri(sbs_sim)])
median(damage_sim[upper.tri(damage_sim)])
median(misrepair_sim[upper.tri(misrepair_sim)])

cosmic_v2 <- read_delim('data/COSMIC_v2_SBS_GRCh37.txt', delim='\t')
sbs_v2_sim <- cosmic_v2 %>%
  select(-c(Type)) %>%
  as.matrix() %>%
  lsa::cosine()

median(sbs_v2_sim[upper.tri(sbs_v2_sim)])
tibble(value = sbs_v2_sim[upper.tri(sbs_v2_sim)]) %>%
  mutate(redund = value >= 0.8) %>%
  summarize(redund = 100*round(sum(redund)/n(),2))


# marginalization
labs_df = data.frame(lab = c('SBS4/22','SBS4/22','SBS4/22','SBS2/13','SBS2/13','SBS2/13'), 
                     x = c(sbs_sim['SBS22',]['SBS4'], damage_sim['SBS22',]['SBS4'], 
                     misrepair_sim['SBS22',]['SBS4'], sbs_sim['SBS2',]['SBS13'], 
                     damage_sim['SBS2',]['SBS13'], misrepair_sim['SBS2',]['SBS13']),
                     name = ordered(c("COSMIC", "Marginalized for trinucleotide context", "Marginalized for substitution",
                                      "COSMIC", "Marginalized for trinucleotide context", "Marginalized for substitution"),
                              levels = c('COSMIC', 'Marginalized for trinucleotide context', 'Marginalized for substitution'))
                     )
                     
 
                     
sims_dist <- tibble(COSMIC = sbs_sim[upper.tri(sbs_sim)], 
  `Marginalized for trinucleotide context` = damage_sim[upper.tri(damage_sim)],
  `Marginalized for substitution` = misrepair_sim[upper.tri(misrepair_sim)]) %>%
  pivot_longer(everything()) %>%
  mutate(name = ordered(name, levels = c('COSMIC', 'Marginalized for trinucleotide context', 'Marginalized for substitution'))) %>%
  mutate(redund = value >= 0.8) %>%
  group_by(name) %>%
  ggplot(aes(x = value, y= after_stat(scaled), fill=name, colour=name)) + 
  geom_density(alpha = 0.4, linewidth = 1/.pt) + 
  geom_vline(xintercept = 0.8, linetype='dashed') +
  geom_segment(aes(x = 0.9, y = 0.7, xend = 1, yend = 0.7),
               arrow = arrow(length = unit(5, "pt")), colour = 'black', linewidth = 0.8/.pt) + 
  geom_text(data = . %>% summarize(redund = 100*round(sum(redund)/n(),2)),
            size = 5/.pt, colour = 'black', aes(x = .95, y =0.8, 
                            label = paste0(redund, '% redundant'))) +
  geom_text(aes(label= lab, x = x, y = 0), color = 'black', size = 5/.pt, 
            hjust = 'center', angle = 45, nudge_y = 0.15, data = labs_df) + 
  facet_wrap(~name, ncol=1, scales='free_x') + 
  geom_point(aes(x = x, y = 0), data = labs_df) + 
  scale_colour_manual(values = c('dodgerblue', 'maroon3', 'aquamarine4')) + 
  scale_fill_manual(values = c('dodgerblue', 'maroon3', 'aquamarine4'))  + 
  labs(y = 'Proportion', x = 'Cosine similarity') +
  theme_ch() + theme(legend.position = 'none') + 
  scale_x_continuous(limits=c(0,1), breaks=c(0,.25,.5,.75,1)) +
  coord_trans(clip='off')


plot_grid(sims_dist, labels = 'd', label_size = 8)
ggsave('figures/sims_dist.pdf', width=2.6, height=4.2)


left <- plot_grid(cosmic_marg, 
                  plot_grid(damage_marg, misrepair_marg, ncol = 2, rel_widths = c(3,1.5)), 
                  rel_heights = c(1, 1), ncol = 1, labels = 'c', label_size = 8)

right <- plot_grid(sims_dist, labels = 'd', label_size = 8)


plot_grid(left, right, ncol=2, rel_widths = c(1.7, 1))
ggsave('figures/cosmic_marginalization.pdf', width=7, height=4.3)
