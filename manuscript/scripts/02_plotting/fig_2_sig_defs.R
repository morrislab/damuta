source('scripts/03_plotting/plotting_header.R')
library(ggtext)

# Signature definition plots
custom_labels <- function(x) {
  # from chatgpt 27/03/24
  ifelse(x == 0, "0", ifelse(x>0.999, '1', sprintf("%.2f", x)))
}

damage_sigs <- read_csv('results/figure_data/h_phi.csv') %>% 
  rename(label = `...1`) %>% 
  mutate(label=ordered(paste0('D',1:18), levels=paste0('D',1:18)) ) %>%
  pivot_longer(cols = -c(label)) %>% 
  mutate(base = substr(name, 2,2)) %>%
  arrange(base, name) %>%
  mutate(name = str_replace_all(name, '(.)(C)(.)', "\\1<span style='color:maroon2;'>C</span>\\3")) %>%
  mutate(name = str_replace_all(name, '(.)(T)(.)', "\\1<span style='color:maroon4;'>T</span>\\3")) %>%
  mutate(name = ordered(name, levels = unique(name))) %>%
  group_by(label) %>%
  mutate(ylab = 0.9*max(value)) %>%
  ggplot(aes(x = name, y=value, fill=base)) + 
  geom_bar(stat = 'identity') + 
  facet_wrap(~label, ncol = 2, scales = 'free_y') + 
  theme_pubr() + 
  scale_fill_manual(values=c('maroon2', 'maroon4')) + 
  geom_text(aes(x=2.5,y=ylab,label=label), 
            data = . %>% group_by(label) %>% slice(1),
            size = 6/.pt, fontface='bold') +
  guides(fill=FALSE) + 
  labs(x = 'Trinucleotide Context', y = 'Proportion') + 
  scale_y_continuous(breaks = function(x) c(min(x), max(x)), expand = c(0, 0),
                     labels = custom_labels) + 
  theme_ch() + 
  rotate_x_text(hjust=0) + 
  theme(axis.text.x = element_markdown(),
        strip.text = element_blank()) + 
  coord_trans(clip = 'off')


misrepair_sigs <- read_csv('results/figure_data/h_eta.csv') %>% 
  rename(label = `...1`) %>% 
  mutate(label=ordered(paste0('M',1:6), levels=paste0('M',1:6))) %>%
  pivot_longer(cols = -c(label)) %>% 
  mutate(base = substr(name, 1,1)) %>%
  arrange(base, name) %>%
  mutate(name = str_replace_all(name, 'C>(.)', "<span style='color:aquamarine2;'>C</span>>\\1")) %>%
  mutate(name = str_replace_all(name, 'T>(.)', "<span style='color:aquamarine4;'>T</span>>\\1")) %>%
  mutate(name = ordered(name, levels = unique(name))) %>%
  ggplot(aes(x = name, y=value, fill=base)) + 
  geom_bar(stat = 'identity') + 
  facet_wrap(~label, ncol = 1) + 
  theme_pubr() + 
  scale_fill_manual(values=c('aquamarine2', 'aquamarine4')) + 
  geom_text(aes(x=6,y=.95,label=label), 
            data = . %>% group_by(label) %>% slice(1),
            size = 6/.pt, fontface='bold') + 
  guides(fill=FALSE) + 
  labs(x = 'Substitution', y = 'Proportion') + 
  scale_y_continuous(breaks = function(x) c(min(x), max(x)), expand = c(0, 0),
                     labels = custom_labels) + 
  theme_ch() + 
  rotate_x_text(hjust=0) + 
  theme(axis.text.x = element_markdown(),
        strip.text = element_blank()) + 
  coord_trans(clip = 'off') 
  

#pdf('figures/sig_defs.pdf', width=7, height=4)
#plot_grid(damage_sigs, misrepair_sigs, NULL, ncol=3, labels = c('a', 'b', ''), rel_widths = c(5,1,0.6), label_size = 8)
#dev.off()


pdf('figures/fig_2.pdf', width=7, height=4)
plot_grid(damage_sigs, misrepair_sigs, NULL, ncol=3, 
                      labels = c('a', 'b', ''), rel_widths = c(5,1,0.6), label_size = 8)
#b <- plot_grid(bubble, nmi, rel_widths = c(1.4,1), labels = c('c', 'd'), label_size = 8)
#plot_grid(t, b, nrow=2, rel_heights = c(0.9,1))
dev.off()


