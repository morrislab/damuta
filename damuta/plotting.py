from .utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage,  fcluster
from scipy.spatial.distance import pdist
sns.set_style("white")
sns.despine()


tau_col = np.repeat(['cyan', 'black', 'red', 'grey', 'lightgreen', 'pink'], 16)

phi_col = np.repeat(['#EE30A7', '#8B1C62'], 16)
eta_col = np.repeat(['#76EEC6', '#458B74'], 3)


cosmic_palette = sns.color_palette(tau_col)
damage_palette = sns.color_palette(phi_col)
misrepair_palette = sns.color_palette(eta_col)

def plot_signatures(sigs, pal=cosmic_palette, aspect=5):
    df = sigs.reset_index()
    df = df.melt('index', var_name = 'Type', value_name = 'Value')
    g = sns.FacetGrid(df, row="index", sharey=False, aspect=aspect)
    g.map_dataframe(sns.barplot, x='Type', y = 'Value', palette = pal)
    plt.xticks(rotation=90)
    g.set_titles(row_template = '{row_name}')
    return g

def plot_cosmic_signatures(sigs, pal=cosmic_palette, aspect=5):
    return plot_signatures(sigs, pal, aspect)

def plot_damage_signatures(sigs, pal=damage_palette, aspect=3):
    return plot_signatures(sigs, pal, aspect)

def plot_misrepair_signatures(sigs, pal=misrepair_palette, aspect=1):
    return plot_signatures(sigs, pal, aspect)


    
def plot_phi_posterior(phi_approx, cols = phi_col):
    assert len(phi_approx.shape) == 3
    T, J, C = phi_approx.shape
    if cols is None: cols = [None]*32
    fig, axes = plt.subplots(J, 1, figsize=(8, 2*J), sharex=True)
    if J == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        for c in range(C):
            if "mut32" in globals():
                label = '{}'.format(mut32[c])
            else:
                label = str(c)
            ax.hist(phi_approx[:, j, c], bins=30, alpha=0.5, color=cols[c % len(cols)], label=label)
        ax.set_title('Phi {}'.format(j))
        ax.legend()
    plt.tight_layout()
    return fig


def plot_eta_posterior(eta_approx, cols = eta_col):
    assert len(eta_approx.shape) == 4
    T, K, C, M  = eta_approx.shape
    assert C==2
    if cols is None: cols = [None]*6
    fig, axes = plt.subplots(1, K, figsize=(4*K, 4), sharey=True)
    if K == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        for c in range(C):
            for m in range(M):
                idx = m if c == 0 else (m+3)
                if "mut6" in globals():
                    label = '{}'.format(mut6[idx])
                else:
                    label = str(idx)
                ax.hist(eta_approx[:, k, c, m], bins=30, alpha=0.5, color=cols[idx % len(cols)], label=label)
        ax.set_title('Eta {}'.format(k))
        ax.legend()
    plt.tight_layout()
    return fig

def plot_mean_std(array):
    assert len(array.shape) == 3 or len(array.shape) == 4
    if len(array.shape) == 3:
        array = array[None, :, :, :]
    mean = array.mean((0, 1))
    std = array.std((0, 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(mean.round(2), ax=axes[0], cmap='viridis')
    axes[0].set_title('mean')
    sns.heatmap(std.round(2), ax=axes[1], cmap='viridis')
    axes[1].set_title('std')
    plt.tight_layout()
    return fig

def plot_cossim(tau_gt, tau_hat):
    from sklearn.metrics.pairwise import cosine_similarity
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    im1 = axes[0, 1].imshow(cosine_similarity(tau_gt, tau_gt).round(2), cmap='viridis')
    axes[0, 1].set_title('tau gt vs tau gt')
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 1].imshow(cosine_similarity(tau_hat, tau_hat).round(2), cmap='viridis')
    axes[1, 1].set_title('tau hat vs tau hat')
    plt.colorbar(im2, ax=axes[1, 1])
    cross = cosine_similarity(tau_hat, tau_gt)
    if cross.shape[0] < cross.shape[1]:
        cross = cross.T
    im3 = axes[0, 0].imshow(cross.round(2), cmap='viridis')
    axes[0, 0].set_title('tau hat vs tau gt')
    plt.colorbar(im3, ax=axes[0, 0])
    axes[1, 0].axis('off')
    fig.suptitle('cosine distance of estimated signatures and ground truth')
    plt.tight_layout()
    return fig
    
    
def save_gv(model):
    # render doen't work well. use `dot -Tpng model_graph > foo.png` instead
    gv = pm.model_graph.model_to_graphviz(model)
    gv.render(format = 'png')


def plot_bipartite(w, main = '', ah=0, thresh = 0.1, node_space=20,
                   node_cols=['#a64d79', '#45818e']):
    import networkx as nx
    assert len(w.shape) == 2
    J,K = w.shape
        
    if np.any(w > 1):
        warnings.warn("Some w's >1. Edge width may be misleading.")
    
    if np.any(w < 0):
        warnings.warn("Some w's <0. Edge width may be misleading")
    
    
    y0s = np.flip(np.arange(0,J*node_space,step=node_space))
    y0s = y0s - y0s.mean()
    y1s = np.flip(np.arange(0,K*node_space,step=node_space))
    y1s = y1s - y1s.mean()
    node_y = np.array([[y0, y1, None] for y0 in y0s for y1 in y1s]).flatten()
    node_x = np.array([[0, 1, None] for y0 in y0s for y1 in y1s]).flatten()
    
    w = w.flatten()
    edges = w #/ np.max(w) * 10

    edge_cols = [f'rgba(0,0,0, {0.5*(ew+1)})' for ew in (w / np.max(w))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    G = nx.DiGraph()
    for j in range(J):
        G.add_node('J{}'.format(j), bipartite=0, pos=(0, J-j))
    for k in range(K):
        G.add_node('K{}'.format(k), bipartite=1, pos=(1, K-k))
    for j in range(J):
        for k in range(K):
            if w[j, k] > thresh:
                G.add_edge('J{}'.format(j), 'K{}'.format(k), weight=w[j, k])
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=[node_cols[0]]*J + [node_cols[1]]*K, node_size=500)
    edges = G.edges(data=True)
    for (u, v, d) in edges:
        ax.annotate('', xy=pos[v], xytext=pos[u], arrowprops=dict(arrowstyle='->', lw=2*d['weight'], color='k', alpha=0.5))
    ax.set_title(main)
    ax.axis('off')
    
    return fig

def plot_bipartite_K(weights):
    return plot_bipartite((weights/weights.sum(0)).round(2), main='K repairs J', ah=5)

def plot_bipartite_J(weights):
    return plot_bipartite((weights.T/weights.sum(1)).T.round(2), main='J repaired by K', ah=5)
    
def plot_nmut(nmut_dict):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [nmut_dict[dset] for dset in nmut_dict.keys()]
    ax.boxplot(data, labels=list(nmut_dict.keys()))
    ax.set_title('Number of mutations per sample in datasplit')
    return fig


def plot_pca(X, mcol=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    if mcol is not None:
        if hasattr(mcol, 'dtype') and mcol.dtype != float:
            col = mcol.astype('category').cat.codes
        else:
            col = mcol
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=col, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=ax, label=getattr(mcol, 'name', ''))
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if hasattr(mcol, 'name'):
        ax.set_title('coloured by {}'.format(mcol.name))
    return fig

def plot_elbow_pca(X, n_comp=10, mcol=None):
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    r = pca.explained_variance_ratio_
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(len(r)), r, marker='o')
    axes[0].set_title('% variance explained')
    axes[0].set_xlabel('PC')
    axes[0].set_ylabel('Variance ratio')
    if mcol is not None:
        sc = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=mcol, cmap='viridis', alpha=0.7)
        plt.colorbar(sc, ax=axes[1])
    else:
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    axes[1].set_title('PC1/2')
    return fig

def plot_fclust_scree(mat, metric = 'cosine', max_t = 10):
    d = pdist(mat, metric)
    Z = linkage(d, "ward")
    # from fcluster docs
    # flat clusters so that the original observations in 
    # each flat cluster have no greater a cophenetic distance than t.
    n_clust = [fcluster(Z, t=t, criterion='distance').max() for t in np.arange(1,max_t)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, max_t), n_clust, marker='o')
    ax.set_xlabel('dendrogram cutoff')
    ax.set_ylabel('number of clusters')
    return fig

def pick_cutoff(a, metric='cosine', thresh=5):
    d = pdist(a, metric)
    Z = linkage(d, "ward")
    n_clust = np.array([fcluster(Z, t=t, criterion='distance').max() for t in np.arange(0,100)])
    return np.where(n_clust < thresh)[0].min(0)


def map_to_palette(annotation, pal_list = ['Spectral','Dark2','Set1','Set2','Set3']):
    # map all columns to a pallet entry
    # make sure to subset columns of annotation appropriately
    # ie. only categorical
    i=0
    luts = []
    for col in annotation.columns:
        lut = dict(zip(annotation[col].unique(), sns.color_palette(pal_list[i], len(annotation[col].unique()) )))   
        annotation[col] = annotation[col].map(lut)
        luts.append(lut)
        i+=1
        i%=4
    
    return annotation, luts


def plot_activity_clustermap(df, colour_annotation, lut, Z,
                             titles = ['Tissue Type', 'Dataset'], 
                             bboxes = [(0.05, 0.9), (0.25, 0.9)], **kwargs):
    # https://stackoverflow.com/a/53217838
    # only the colour values should be provided in colour_annotation
    # get these from map_to_palette
    # lut feeds only the legend
    
    colour_annotation = colour_annotation.loc[df.index]
    fig=sns.clustermap(df, row_linkage = Z, col_cluster = False, 
                       linewidths=0.0, rasterized=True,
                       row_colors = colour_annotation, yticklabels=False,
                       **kwargs)
    #plt.subplots_adjust(top = 0.8)
    if lut is not None:
        for label in list(lut[0].keys()):
            fig.ax_col_dendrogram.bar(0, 0, color=lut[0][label], label=label, linewidth=0)
        ll=fig.ax_col_dendrogram.legend(title=titles[0], loc="upper left", 
                                        ncol= 2 if len(lut[0].keys()) < 10 else 4,
                                        bbox_to_anchor=bboxes[0], bbox_transform=gcf().transFigure)
        if len(lut) > 1:
            for label in list(lut[1].keys()):
                fig.ax_row_dendrogram.bar(0, 0, color=lut[1][label], label=label, linewidth=0)
            l2=fig.ax_row_dendrogram.legend(title=titles[1], loc="upper left", ncol=1, bbox_to_anchor=bboxes[1], bbox_transform=gcf().transFigure)
    
    return fig


def plot_activity_matrix(mat, tissue_types, x = 'Signature', thresh = 0.1):
    #assert isinstance(df, pd.DataFrame)
    assert isinstance(tissue_types, pd.Categorical)
    df = pd.DataFrame(mat)
    tots = tissue_types.value_counts()

    circ_size = df.div(df.sum(axis=1), axis=0)>thresh
    circ_size['Tissue Type'] =  tissue_types
    circ_size = circ_size.groupby('Tissue Type').sum()
    circ_size = circ_size.div(tots, axis=0)
    circ_size = circ_size.melt(ignore_index = False).reset_index()

    df['Tissue Type'] = tissue_types
    med_activities = df.groupby('Tissue Type').median()
    med_activities = med_activities.melt(var_name = x, ignore_index = False).reset_index()

    med_activities['n'] = circ_size['value']

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(med_activities[x], med_activities['Tissue Type'], c=med_activities['value'], s=med_activities['n']*300, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Median Activity')
    ax.set_xlabel(x)
    ax.set_ylabel('Tissue Type')
    return fig
        