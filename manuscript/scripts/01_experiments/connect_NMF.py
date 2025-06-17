import os
import numpy as np
import pandas as pd
import damuta as da
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

fp = 'results/figure_data/'
os.makedirs(fp, exist_ok=True)

ann = pd.read_csv('data/phg_clinical_ann.csv', index_col=0, low_memory = False)
W = pd.read_csv('results/figure_data/h_W.csv', index_col = 0).loc[ann.index]
phi = pd.read_csv('results/figure_data/h_phi.csv', index_col = 0)
eta = pd.read_csv('results/figure_data/h_eta.csv', index_col = 0)
ds = pd.read_csv('results/figure_data/phg_all_deconstructsigs_activities.csv', index_col=0, sep='\t').loc[ann.index]
cosmic = pd.read_csv('data/COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col = 0).T[da.utils.mut96]

# calc tau
tau = np.einsum('jpc,kpm->jkpmc', phi.values.reshape(18,2,16), eta.values.reshape(6,2,3)).reshape((-1,96))


# find good N -> iterate for silhouette score, reconstruction error
# this is slow
n_compare = pd.DataFrame()
for rank in np.unique((np.logspace(0, 2, 100)).astype(int)) +1:
    nmf = [NMF(n_components=rank, random_state=42+i, max_iter=2000, l1_ratio=0.5, alpha_W=1e-6, alpha_H=5e-6, init='nndsvd', solver='cd') for i in range(3)]
    nmf = [m.fit(W) for m in nmf]
    components = np.vstack([m.components_ for m in nmf])
    kmeans = KMeans(n_clusters=rank, random_state=42).fit(components)
    # stability
    sil = silhouette_score(components, kmeans.labels_, metric='cosine')
    # reconstruction error
    reconstructed_data = [np.dot(m.transform(W), m.components_) for m in nmf]
    mses = [mean_squared_error(W, r) for r in reconstructed_data]
    frobs = [np.linalg.norm(W - r, ord='fro') for r in reconstructed_data]
    # count # number of cosmic signatures mapped to with > 0.8 cossim
    # renormalize factors -> just pick one nmf
    connect_sigs = (nmf[0].components_ / nmf[0].components_.sum(1)[:,None])
    n_cosmic_covered = (((cosine_similarity(cosmic.values, connect_sigs@tau) >0.8).sum(1)) >0).sum()
    n_connect_covered = (((cosine_similarity(cosmic.values, connect_sigs@tau) >0.8).sum(0)) >0).sum()
    n_compare = pd.concat([n_compare, pd.DataFrame({'rank': rank, 'sil': sil, 'mse': mses, 'frob': frobs, 'n_cosmic_covered': n_cosmic_covered, 'n_connect_covered':n_connect_covered})])

n_compare.to_csv('results/figure_data/n_compare.csv')

# NMF - connectivity factors
best_rank = 30

# check cosine similarity between connectivity signatures and COSMIC
# with best rank, renormalize factors

nmf = NMF(n_components=best_rank, random_state=42, max_iter=2000, l1_ratio=0.5, alpha_W=1e-6, alpha_H=5e-6, init='nndsvd', solver='cd').fit(W)
F2 = nmf.transform(W)
H2 = nmf.components_
# any samples with 0 activity?
np.isclose(F2.sum(1),0).any()
np.median(F2.sum(1))
nmf.reconstruction_err_
# percent of activities close to 0
((np.isclose(F2,0).sum(0) / F2.shape[0])).round(2)
# percent of connect_sig entries close to 0
(np.isclose(H2,0).sum(1) /108).round(2)

connect_sigs = (nmf.components_ / nmf.components_.sum(1)[:,None])
connect_activities = F2 / (F2.sum(1)[:,None])

pd.DataFrame(connect_sigs, index=[f'C{i+1}' for i in range(best_rank)], columns=W.columns).to_csv('results/figure_data/connect_sigs.csv')
pd.DataFrame(connect_activities, columns=[f'C{i+1}' for i in range(best_rank)], index=ann.index).to_csv('results/figure_data/connect_acts.csv')

# use hungarian algorithm to assign closest
sims = cosine_similarity(cosmic.values, connect_sigs@tau)
hungarian = linear_sum_assignment(sims, maximize=True)

# for each connect, note closest cosmic
map1 = pd.DataFrame({'connect': ['C' + str(x+1) for x in hungarian[1]], 'hungarian_cosmic': sims.index[hungarian[0]], 'closest_cosmic': [sims.index[sims['C' + str(x+1)].argmax()] for x in hungarian[1]]})
map1['hungarian_sim'] = np.diag((sims.loc[map1['hungarian_cosmic']])[map1['connect']])
map1['closest_sim'] = np.diag((sims.loc[map1['closest_cosmic']])[map1['connect']])
map1['all >= 0.8'] = [sims.index[sims[col] > 0.8].tolist() for col in map1['connect']]
map1.to_csv('results/figure_data/connect_sig_ann.csv')

# for each cosmic, note closest connect
map2 = pd.DataFrame({'cosmic': sims.index, 'closest_connect': [sims.columns[sims.loc[x].argmax()] for x in sims.index]})
map2['closest_sim'] = [sims.loc[x].max() for x in sims.index]
map2['all >= 0.8'] = [sims.columns[sims.loc[row] > 0.8].tolist() for row in sims.index]
map2.to_csv('results/figure_data/cosmic_ann.csv')

pd.DataFrame(connect_sigs@tau, index=[f'C{i+1}' for i in range(best_rank)], columns=cosmic.columns).to_csv('results/figure_data/connect_sigs_tau.csv')
