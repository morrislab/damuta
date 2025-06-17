source('plotting_header.R')

phi_stab <- read_csv('results/figure_data/stability_phis.csv')
eta_stab <- read_csv('results/figure_data/stability_etas.csv')

get_avg_silhouette <- function(df, sel, k){
  df %>%
    do({
      data_subset <- select(., sel)
      dist_matrix <- as.matrix(proxy::dist(data_subset, method = "cosine"))
      clust <- kmeans(dist_matrix, centers = k)
      silhouette_scores <- cluster::silhouette(clust$cluster, dist_matrix)
      avg_sil_width <- mean(silhouette_scores[, "sil_width"])
      data.frame(avg_sil_width = avg_sil_width)
    })
}

  
phi_sil <- lapply(1:50, function(x){
  phi_stab %>% 
    group_by(init_strategy) %>%
    get_avg_silhouette(as.character(0:31), 18)
}) %>%
  bind_rows() %>%
  mutate(type='Damage')


eta_sil <- lapply(1:50, function(x){
  eta_stab %>% 
    group_by(init_strategy) %>%
    get_avg_silhouette(as.character(0:5), 6)
}) %>%
  bind_rows() %>%
  mutate(type='Misrepair')


get_silhouette <- function(df, sel, k){
  df %>%
    do({
      data_subset <- select(., sel)
      dist_matrix <- (proxy::dist(data_subset, method = "cosine"))
      clust <- cutree(hclust(dist_matrix), k = k)
      silhouette_scores <- cluster::silhouette(clust, dist_matrix)
      data.frame(sil_width=(silhouette_scores[, "sil_width"]))
    })
}

phi_sil <- phi_stab %>% 
    group_by(init_strategy) %>%
    get_silhouette(as.character(0:31), 18) %>%
  mutate(type='Damage')


eta_sil <- eta_stab %>% 
    group_by(init_strategy) %>%
    get_silhouette(as.character(0:5), 6) %>%
  mutate(type='Misrepair')

stab <- bind_rows(phi_sil,eta_sil) %>%
  mutate(init_strategy = str_to_title(init_strategy)) %>% 
  ggplot(aes(x = type, y = sil_width, colour = init_strategy)) + 
  geom_boxplot(outliers = F) +
  geom_point(size = 1, position = position_jitterdodge(jitter.width = 0.2)) + 
  theme_ch() + 
  labs(x = 'Signature type', y = 'Average stability', colour = 'Initialization strategy') + 
  scale_colour_manual(values = c('orange', 'purple')) + 
  theme(legend.position = 'none')

# ELBO

elbo <- read_csv('results/figure_data/wandb_export_2024-03-28T10 38 00.741+00 00.csv') %>% 
  pivot_longer(-c(Step)) %>%
  separate(name, c('name', 'measure'), ' - ') %>%
  filter(measure == 'ELBO') %>%
  mutate(init_strategy = case_when(
    name == 'dandy-sweep-10' ~ 'Uniform',
    name == 'balmy-sweep-9' ~ 'Uniform',
    name == 'solar-sweep-7' ~ 'Uniform',
    name == 'radiant-sweep-8' ~ 'Uniform',
    name == 'distinctive-sweep-6' ~ 'Uniform',
    name == 'hardy-sweep-5' ~ 'Kmeans',
    name == 'likely-sweep-3' ~ 'Kmeans',
    name == 'pleasant-sweep-1' ~ 'Kmeans',
    name == 'divine-sweep-2' ~ 'Kmeans',
    name == 'earthy-sweep-4' ~ 'Kmeans'
  )) 


elbo_line <- elbo %>%
  mutate(rolling_diff = value - lag(value, default = first(value))) %>%
  ggplot(aes(x=Step, y = value, colour = init_strategy, group =name)) + 
  geom_line() + 
  theme_ch() + 
  labs(y = 'ELBO', colour = 'Initialization strategy') + 
  scale_colour_manual(values = c('orange', 'purple')) + 
  theme(legend.position = c(0.8, 0.8))

elbo_final <- elbo %>%
  group_by(init_strategy, name) %>%
  summarize(value = max(value)) %>%
  ggplot(aes(x = init_strategy, y = value, colour = init_strategy)) + 
  geom_boxplot( outliers = F) +
  geom_point(size = 1, position = position_jitterdodge()) + 
  theme_ch() + 
  labs(x= 'Initialization strategy', y = 'Final step ELBO (lower is better)', colour = 'Initialization strategy') + 
  scale_colour_manual(values = c('orange', 'purple')) + 
  theme(legend.position = 'none')


pdf('figures/stability_init.pdf', width=7, height=3)
plot_grid(stab, elbo_line, elbo_final, ncol=3, 
          align='v', axis = 'lr', labels = c('a','b', 'c'), label_size = 8)
dev.off()


phis <- read_csv('results/figure_data/c7rl70ns_phis.csv')
etas <- read_csv('results/figure_data/c7rl70ns_etas.csv')


phis %>% 
  filter(type_col=='organ', alpha_bias==0.01, beta_bias == 0.01) %>%
  group_by(psi_bias) %>%
  do({
    dist_matrix <- (proxy::dist(.[,2:33], method = "cosine"))
    clust <- cutree(hclust(dist_matrix), k = 18)
    silhouette_scores <- cluster::silhouette(clust, dist_matrix)
    data.frame(mean_sil_width=mean(silhouette_scores[, "sil_width"]))
})

etas %>% 
  filter(type_col=='organ') %>%
  group_by(psi_bias, alpha_bias, beta_bias, posterior_sample) %>%
  do({
    dist_matrix <- (proxy::dist(.[,2:7], method = "cosine"))
    clust <- cutree(hclust(dist_matrix), k = 6)
    silhouette_scores <- cluster::silhouette(clust, dist_matrix)
    data.frame(mean_sil_width=mean(silhouette_scores[, "sil_width"]))
  })

bind_rows(
etas %>% 
  group_by(posterior_sample, alpha_bias, type_col, beta_bias, psi_bias) %>%
  get_silhouette(names(.)[2:7], 6) %>%
  summarise(mean_sil_width=mean(sil_width)) %>%
  mutate(type='Misrepair')
) %>%
  ungroup() %>%
  mutate(across(c(type_col, alpha_bias, beta_bias, psi_bias), ~factor(.x))) %>%
  #pivot_longer(c(alpha_bias, psi_bias, beta_bias, type_col)) %>%
  ggplot(aes(x = psi_bias, y = mean_sil_width, colour = type)) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  #facet_grid(beta_bias~alpha_bias, scales = 'free') + 
  scale_colour_manual(values = c('Damage' = 'maroon3', 'Misrepair'='aquamarine3')) + 
  theme_ch()




phis <- read_csv('results/figure_data/hol1va3d_phis.csv')
etas <- read_csv('results/figure_data/hol1va3d_etas.csv')



get_mean_silhouette <- function(df, sel, k){
  df %>%
    do({
      data_subset <- select(., sel)
      dist_matrix <- (proxy::dist(data_subset, method = "cosine"))
      clust <- cutree(hclust(dist_matrix), k = k)
      silhouette_scores <- cluster::silhouette(clust, dist_matrix)
      data.frame(mean_sil_width=mean(silhouette_scores[, "sil_width"]))
    })
}
  

df = head(etas, 1000)
Heatmap(as.matrix(df[,2:7]), col=scale_col, name = 'Signature\nweight',
        #row_split = cutree(hclust(proxy::dist(as.matrix(df[,2:7]), method='cosine')), k=6), 
        cluster_row_slices = F, cluster_columns = F,
        clustering_distance_rows = function(x){proxy::dist(x, method='cosine')}) + 
  rowAnnotation(D=(df$n_damage_sigs) )



f <- function(m, k){
  print(m, k)
  mean(cluster::silhouette(cutree(hclust(proxy::dist(m, method='cosine')), k=k)))
}

phis %>%
  group_by(posterior_sample, n_damage_sigs) %>%
  summarize(n())
  do({
    dist_matrix <- (proxy::dist(.[, 2:33], method = "cosine"))
    clust <- cutree(hclust(dist_matrix), k =first(.$n_damage_sigs))
    print(clust)
    silhouette_scores <- cluster::silhouette(clust, dist_matrix)
    print(silhouette_scores)
    data.frame(mean_sil_width=mean(silhouette_scores[, "sil_width"]))
  })


  summarise(d = hclust(proxy::dist( method='cosine')))
  summarise(MeanSilhouette = f(select(.,ACA:TTT), 2)) %>%
  ungroup()


bind_rows(
  phis %>% 
    filter(n_damage_sigs > 1) %>%
    group_by(posterior_sample, n_damage_sigs, n_misrepair_sigs) %>%
    summarise()
  
    get_mean_silhouette(names(.)[2:32]) %>%
    summarise(mean_sil_width=mean(sil_width)) %>%
    mutate(type='Damage'),
  
  etas %>% 
    group_by(alpha_bias, type_col, beta_bias, psi_bias) %>%
    get_silhouette(names(.)[2:6], 6) %>%
    summarise(mean_sil_width=mean(sil_width)) %>%
    mutate(type='Misrepair')
) %>%
  ungroup() %>%
  mutate(across(c(type_col, alpha_bias, beta_bias, psi_bias), ~factor(.x))) %>%
  #pivot_longer(c(alpha_bias, psi_bias, beta_bias, type_col)) %>%
  ggplot(aes(x = psi_bias, y = mean_sil_width, colour = type)) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  #facet_grid(beta_bias~alpha_bias, scales = 'free') + 
  scale_colour_manual(values = c('Damage' = 'maroon3', 'Misrepair'='aquamarine3')) + 
  theme_ch()




sparsity <- read_csv('results/figure_data/sparsity.csv') %>% 
  mutate(across(c(type_col, alpha_bias, beta_bias, psi_bias), ~factor(.x)))

sparsity %>%
  filter(type_col == 'organ') %>%
  ggplot(aes(x = psi_bias, y = value, colour = type)) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  facet_wrap(~measure, scales = 'free') + 
  scale_colour_manual(values = c(d = 'maroon3', m='aquamarine3')) + 
  theme_ch()

sparsity %>%
  filter(type_col == 'organ') %>%
  ggplot(aes(x = factor(alpha_bias), y = value, colour = type)) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  facet_wrap(~measure, scales = 'free') + 
  scale_colour_manual(values = c(d = 'maroon3', m='aquamarine3')) + 
  theme_ch()

sparsity %>%
  filter(type_col == 'organ') %>%
  ggplot(aes(x = factor(beta_bias), y = value, colour = type)) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  facet_wrap(~measure, scales = 'free') + 
  scale_colour_manual(values = c(d = 'maroon3', m='aquamarine3')) + 
  theme_ch()

sparsity %>%
  ggplot(aes(x = factor(type_col), y = value, colour = factor(type) )) + 
  geom_boxplot(outliers = F) + 
  geom_point(position = position_jitterdodge()) + 
  facet_wrap(~measure, scales = 'free') + 
  scale_colour_manual(values = c(d = 'maroon3', m='aquamarine3')) + 
  theme_ch()


scale_col = circlize::colorRamp2(c(0, 0.4, 0.8), c("white", adjustcolor("orangered3", alpha.f = 0.5), "orangered3"))
factor_col = c(`0.01`='#F8766D', `0.1` = '#7CAE00', `1` = '#00BFC4', `2` = '#C77CFF')

df = etas
Heatmap(as.matrix(df[,2:7]), col=scale_col, name = 'Signature\nweight',
        row_split = cutree(hclust(proxy::dist(as.matrix(df[,2:7]), method='cosine')), k=6), 
        cluster_row_slices = F, cluster_columns = F,
        clustering_distance_rows = function(x){proxy::dist(x, method='cosine')}) + 
  rowAnnotation(Psi=factor(df$psi_bias),
                Alpha=factor(df$alpha_bias),
                Beta=factor(df$beta_bias),
                show_legend = c(Psi = F,
                                Alpha = F,
                                Beta = T), 
                col = list(Psi = factor_col,
                           Alpha = factor_col, 
                           Beta = factor_col)) 


df = phis
Heatmap(as.matrix(df[,2:33]), col=scale_col, name = 'Signature\nweight',
        row_split = cutree(hclust(proxy::dist(as.matrix(df[,2:33]), method='cosine')), k=18), 
        cluster_row_slices = F, cluster_columns = F,
        clustering_distance_rows = function(x){proxy::dist(x, method='cosine')}) + 
  rowAnnotation(Psi=factor(df$psi_bias),
                Alpha=factor(df$alpha_bias),
                Beta=factor(df$beta_bias),
                show_legend = c(Psi = F,
                                Alpha = F,
                                Beta = T), 
                col = list(Psi = factor_col,
                           Alpha = factor_col, 
                           Beta = factor_col)) 
