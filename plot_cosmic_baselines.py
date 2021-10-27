import pandas as pd
import numpy as np
import damuta as da
from damuta.plotting import *
import seaborn as sns
import matplotlib.pyplot as pyplot
import os

sns.set_style("white")
sns.despine()

cosmic = load_sigs("data/COSMIC_v3.2_SBS_GRCh37.txt")

fp = f'plots/cosmic' 
os.makedirs(fp, exist_ok=True)

# plot distribution of signature similarities
pyplot.clf()
fig = sns.histplot(cosine_similarity(cosmic.to_numpy())[np.tril_indices(cosmic.shape[0],-1)], bins = 10)
fig.set(title = 'COSMIC Signature Similarities', xlabel='Cosine Similarity')
fig.figure.savefig(f'{fp}/cosmic_simdist.svg')

# marginalized phi
pyplot.clf()
fig = sns.histplot(cosine_similarity(get_phi(cosmic.to_numpy()))[np.tril_indices(cosmic.shape[0],-1)], bins = 10)
fig.set(title = 'COSMIC Marginalized Damage Signature Similarities',
        xlabel='Cosine Similarity')
fig.figure.savefig(f'{fp}/cosmic_phi_simdist.svg')

# marginalized eta
pyplot.clf()
fig = sns.histplot(cosine_similarity(get_eta(cosmic.to_numpy()).reshape(-1,6))[np.tril_indices(78,-1)], bins = 10)
fig.set(title = 'COSMIC Marginalized Repair Signature Similarities', xlabel='Cosine Similarity')
fig.figure.savefig(f'{fp}/cosmic_eta_simdist.svg')


# Marginalization examples

for sig_name in ['SBS2', 'SBS13', 'SBS11', 'SBS6', 'SBS9']:
    fig = plot_phi(get_phi(cosmic[cosmic.index == sig_name].to_numpy())) 
    fig.write_image(f'{fp}/{sig_name}_marginal_phi.svg')
    fig = plot_eta(get_eta(cosmic[cosmic.index == sig_name].to_numpy()).reshape(-1,6))
    fig.write_image(f'{fp}/{sig_name}_marginal_eta.svg')

