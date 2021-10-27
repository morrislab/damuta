import pandas as pd
import numpy as np
import damuta as da
from damuta.plotting import *
import theano
import seaborn as sns
import matplotlib.pyplot as pyplot
import umap
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
from matplotlib.pyplot import gcf
import os

sns.set_style("white")
sns.despine()

model, trace, dataset_args, model_args, pymc3_arg, run_id = da.load_checkpoint('ckpt/1yzfxe31')
counts, annotation = load_datasets(dataset_args)
trn, val, tst1, tst2 = da.split_data(counts, rng=np.random.default_rng(dataset_args['data_seed']))

fp = f'plots/{run_id}' 
os.makedirs(fp, exist_ok=True)

J = model_args['J']
K = model_args['K']
S = trn.shape[0]


fig = go.Figure(data = go.Scatter(y=trace.hist), 
                layout_title_text="",
                layout = {'yaxis': {'title': 'ELBO'},
                          'xaxis': {'title': 'Iteration'},
                          'font': {'size':20}})
fig.write_image(f'{fp}/trace_hist.svg')

cosmic = load_sigs("data/COSMIC_v3.2_SBS_GRCh37.txt")
hat = trace.approx.sample(10)

A = np.moveaxis(hat.A,1,2)
W = (hat.theta[:,:,:,None] * A).mean(0)
A = A.mean(0)
theta = hat.theta.mean(0)
phi = hat.phi.mean(0)
gamma = hat.gamma.mean(0)
eta = hat.eta.mean(0).reshape(-1,6)

fig = plot_sigs(trn.to_numpy()[0:5], row_title = "Sample") 
fig = fig.update_layout(title = "Mutational Catalogues", height = 800)
fig.write_image(f'{fp}/raw_sigs.svg')

## plot phi 
fig = plot_phi(phi).update_layout(height = 2000)
fig = fig.update_xaxes(tickfont_size=15)
fig.write_image(f'{fp}/phis.svg')

## plot eta 
fig = plot_eta(eta).update_layout(height = 1500)
fig = fig.update_xaxes(tickfont_size=15)
fig.write_image(f'{fp}/etas.svg')

# plot distribution of signature similarities
pyplot.clf()
fig = sns.histplot(cosine_similarity(phi)[np.tril_indices(J,-1)], bins = 10, color = phi_col[0])
fig.set(title = 'Inferred Damage Signature Similarities',
        xlabel='Cosine Similarity')
fig.figure.savefig(f'{fp}/phi_simdist.svg')

# plot distribution of signature similarities
pyplot.clf()
fig = sns.histplot(cosine_similarity(eta)[np.tril_indices(K,-1)], bins = 10, color = eta_col[0])
fig.set(title = 'Inferred Repair Signature Similarities',
        xlabel='Cosine Similarity')
fig.figure.savefig(f'{fp}/eta_simdist.svg')

# profile inferred signatures 
pyplot.clf()
profile_sigs(phi, da.get_phi(cosmic.to_numpy()), refidx=cosmic.index).to_csv(f'{fp}/phi_profile.csv')
profile_sigs(eta, da.get_eta(cosmic.to_numpy()).reshape(-1,6), refidx=cosmic.index, thresh = 0.95).to_csv(f'{fp}/eta_profile.csv')
profile_sigs(get_tau(phi, eta.reshape(-1,2,3)), cosmic.to_numpy(), refidx=cosmic.index).to_csv(f'{fp}/tau_profile.csv')

# mean/median W/A
pyplot.clf()
sns.heatmap(W.mean(0)).set_title('mean W')
fig.figure.savefig(f'{fp}/mean_W.svg') 

pyplot.clf()
sns.heatmap(A.mean(0)).set_title('mean A')
fig.figure.savefig(f'{fp}/mean_A.svg')

pyplot.clf()
sns.heatmap(np.median(W, axis=0)).set_title('median W')
fig.figure.savefig(f'{fp}/median_W.svg')

pyplot.clf()
sns.heatmap(np.median(A, axis=0)).set_title('median A')
fig.figure.savefig(f'{fp}/median_A.svg')

##################

# reduce annotation to training set
ann_trn = annotation.copy()
ann_trn = ann_trn.loc[trn.index]

# update annotation with data source
ann_trn['pcawg_class'].value_counts()
ann_trn['data_source'] = ann_trn.index.str.contains("sample")
ann_trn['data_source'].map({False: "PCAWG", True: "Hartwig"})

# update annotation with cluster membership
Z_theta = linkage(pdist(theta, 'cosine'), "ward")
ann_trn['theta_cluster'] = fcluster(Z_theta, t=pick_cutoff(theta), criterion='distance')
Z_gamma = linkage(pdist(gamma, 'euclidean'), "ward")
ann_trn['gamma_cluster'] = fcluster(Z_gamma, t=pick_cutoff(gamma, metric = 'euclidean'), criterion='distance')
Z_W = linkage(pdist(W.reshape(S,-1), 'cosine'), "ward")

# plot cluster maps of theta, gamma, and W
# colour clustering with datasource, pcawg_class, and cluster membership

# find theta and gamma clusters 
fig = plot_fclust_scree(theta).update_layout(title = f'Scree of Theta. Cuttoff selected: {pick_cutoff(theta)}')
fig.write_image(f'{fp}/theta_scree.svg')
fig = plot_fclust_scree(gamma, 'euclidean')
fig = fig.update_layout(title = f"Scree of Gamma. Cuttoff selected: {pick_cutoff(gamma, metric = 'euclidean')}")
fig.write_image(f'{fp}/gamma_scree.svg')

col_ann, luts = map_to_palette(ann_trn[['pcawg_class', 'data_source', 'theta_cluster', 'gamma_cluster']])

def plot_activity_clustermap(df, colour_annotation, lut, Z):
    # https://stackoverflow.com/a/53217838
    # only the colour values should be provided in colour_annotation
    # get these from map_to_palette
    # lut feeds only the legend
    
    colour_annotation = colour_annotation.loc[df.index]
    fig=sns.clustermap(df, row_linkage = Z, col_cluster = False, linewidth=0,
                       row_colors = colour_annotation, yticklabels=False)
    
    for label in list(lut.keys()):
        fig.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
    ll=fig.ax_col_dendrogram.legend(title='tissue type',loc="center", ncol=2, bbox_to_anchor=(0.5, 0.9), bbox_transform=gcf().transFigure)

    #for label in list(lut2.keys()):
    ##    g.ax_row_dendrogram.bar(0, 0, color=lut2[label], label=label, linewidth=0)
    ##l2=g.ax_row_dendrogram.legend(title='gamma_cluster', loc="center", ncol=2,bbox_to_anchor=(0.3, 0.87), bbox_transform=gcf().transFigure)
    #
    
    return fig

pyplot.clf()
fig = plot_activity_clustermap(pd.DataFrame(W.reshape(S,-1), index =trn.index), col_ann, luts[0], Z_W)
pyplot.savefig(f'{fp}/W_cluster.svg')

pyplot.clf()
fig = plot_activity_clustermap(pd.DataFrame(theta, index =trn.index), col_ann, luts[0], Z_W)
pyplot.savefig(f'{fp}/theta_cluster.svg')

pyplot.clf()
fig = plot_activity_clustermap(pd.DataFrame(gamma, index =trn.index), col_ann, luts[0], Z_W)
pyplot.savefig(f'{fp}/gamma_cluster.svg')





#ann = pd.DataFrame({'type': annotation.loc[trn.index]['hw']}, index = trn.index)
#d = pdist(hat.theta.mean(0).reshape(S, -1), 'cosine')
#Z = linkage(d, "ward")
#ann['theta_cluster'] = fcluster(Z, t=2, criterion='distance')
#d = pdist(hat.gamma.mean(0).reshape(S, -1), 'euclidean')
#Z = linkage(d, "ward")
#ann['gamma_cluster'] = fcluster(Z, t=6, criterion='distance')
#
#mat=W.reshape(S,-1)
#df = pd.DataFrame(mat, index = trn.index)
#ann = ann.sort_values('type')
#df = df.loc[ann.index]
#pal1 = list(sns.color_palette("Paired",7))[::-1]
#lut1 = dict(zip(ann['type'].unique(), pal1))
#ann['type'] = ann['type'].map(lut1)
#
#pal2 = sns.color_palette("Set1")
#lut2 = dict(zip(ann['gamma_cluster'].unique(), pal2))
#ann['gamma_cluster'] = ann['gamma_cluster'].map(lut2)
#
#pal3 = sns.color_palette("Set2")
#lut3 = dict(zip(ann['theta_cluster'].unique(), pal3))
#ann['theta_cluster'] = ann['theta_cluster'].map(lut3)
#
#wdf = pd.DataFrame(W.reshape(S,-1)).set_index(trn.index)
#wdf = wdf.loc[df.index]
#
#d = pdist(wdf, 'cosine')
#Z = linkage(d, "complete")
#
#
#pyplot.clf()
#g=sns.clustermap(df, row_linkage = Z, vmax=0.1, col_cluster = False, row_colors = ann, yticklabels=False)
#
#for label in list(lut1.keys()):
#    g.ax_col_dendrogram.bar(0, 0, color=lut1[label], label=label, linewidth=0)
#l1=g.ax_col_dendrogram.legend(title='tissue type',loc="center", ncol=2, bbox_to_anchor=(0.5, 1), bbox_transform=gcf().transFigure)
#
##for label in list(lut2.keys()):
##    g.ax_row_dendrogram.bar(0, 0, color=lut2[label], label=label, linewidth=0)
##l2=g.ax_row_dendrogram.legend(title='gamma_cluster', loc="center", ncol=2,bbox_to_anchor=(0.3, 0.87), bbox_transform=gcf().transFigure)
#
#pyplot.show()
#
#
## In[ ]:
#
#
#
#
#
## In[53]:
#
#
#mat=W.reshape(S,-1)
#df = pd.DataFrame(mat)
#df['type'] = annotation.loc[trn.index].reset_index()['pcawg_class']
#df=df.set_index(trn.index)
#df=df.sort_values("type")
#t=df.pop('type')
#pal = list(sns.color_palette("Paired",7))[::-1]
#lut = dict(zip(t.unique(), pal))
#row_colors = t.map(lut)
#
#for label in t.unique():
#    df2=df.loc[t == label,:]
#    row_colors2 = row_colors.loc[df2.index]
#    
#    df2=df2.reset_index(drop=True)
#    row_colors2 = row_colors2.reset_index(drop=True)
#    
#    pyplot.clf()
#    g=sns.clustermap(df2, col_cluster = False, row_colors = row_colors2)
#    g.ax_col_dendrogram.bar(0, 0, color=lut[label], label=label, linewidth=0)
#    g.ax_col_dendrogram.legend(title='tissue type')
#    pyplot.show()
#
#
## In[ ]:
#
#
#
#


## Do we need A? theta*gamma to test
#tg = np.einsum("sj,sk->sjk", hat.theta.mean(0), hat.gamma.mean(0))
#tg.shape
#
#normalize(tg.reshape(S,-1),axis=1)
#
#d = pdist(W.reshape(S,-1), 'cosine')
#Z = linkage(d, "complete")
#
#pyplot.clf()
#sns.clustermap(normalize(tg.reshape(S,-1),axis=1), row_linkage =Z, col_cluster = False)
#pyplot.show()
#
#
#
#pyplot.clf()
#sns.clustermap(normalize(W.reshape(S,-1), axis=1), row_linkage =Z, col_cluster = False)
#pyplot.show()
#