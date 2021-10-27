from .utils import *
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import plotly as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import n_colors
import seaborn as sns

plt.io.templates.default = "none"
sns.set_style("white")
sns.despine()


tau_col = np.repeat(['cyan', 'black', 'red', 'grey', 'lightgreen', 'pink'], 16)
#phi_col = np.repeat(['green', 'blue'], 16)
phi_col = np.repeat(['#a64d79'], 32)
#eta_col = np.repeat(['orange', 'lightblue'], 3)
#eta_col = np.array(['cyan', 'black', 'red', 'grey', 'lightgreen', 'pink'])
eta_col = np.repeat(['#45818e'], 6)

def plot_sigs(sigs, xlab=None, cols=None, row_title='Sig'):
    assert len(sigs.shape) == 2
    if xlab is None: xlab = np.arange(sigs.shape[1])
    if cols is None: cols = np.repeat('grey', sigs.shape[1])
    
    fig = plt.subplots.make_subplots(rows=sigs.shape[0], cols=1, shared_xaxes=True,
                                     row_titles=([f'{row_title} {l}' for l in range(sigs.shape[0])]) )
    
    for s in range(sigs.shape[0]):
        fig.add_trace(go.Bar(x=xlab, y=sigs[s], hoverinfo='y', showlegend = False,
                             textposition='auto', marker_color=cols,
                        
                      ), row = s+1, col = 1 )

    fig.update_xaxes(tickangle=-45, matches = 'x')
    return fig

def plot_tau(tau):
    if len(tau.shape) == 1: 
        tau = tau.reshape(1,-1)
    return plot_sigs(tau, mut96, tau_col, 'Tau')


def plot_phi(phi):
    if len(phi.shape) == 1: 
        phi = phi.reshape(1,-1)
    return plot_sigs(phi, xlab=mut32, cols=phi_col, row_title='Phi')

def plot_eta(eta, cols = eta_col):
    # eta should be Kx2x3 or Kx6
    eta=eta.reshape(-1,6)
    return plot_sigs(eta, xlab = mut6, cols = eta_col, row_title = 'Eta')

    
def plot_phi_posterior(phi_approx, cols = phi_col):
    # TxJxC dimension df yields J subplots, C traces 
    assert len(phi_approx.shape) == 3
    T, J, C = phi_approx.shape
    if cols is None: cols = [None]*32
    
    fig = plt.subplots.make_subplots(rows=J, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                                     row_titles=([f'Phi {l}' for l in range(J)]))

    for j in range(J):
        for d, col, l in zip(phi_approx[:,j,:].T, cols, mut32):
            fig.add_trace(go.Histogram(x=d, histnorm='probability', marker_color=col,
                                       legendgroup = l, showlegend = j==0,
                                       name = l, hoverinfo='name'), row = j+1, col = 1)

    fig.update_yaxes(showticklabels = False)
    fig.update_layout(showlegend=True, barmode='overlay')
    fig.update_annotations(textangle = 0, x = -0.1)
    fig.update_traces(opacity=0.7)
    
    return fig


def plot_eta_posterior(eta_approx, cols = eta_col):
    # TxKxCxM dimension df yields CxK subplots, M traces 
    assert len(eta_approx.shape) == 4
    T, K, C, M  = eta_approx.shape
    assert C==2
    if cols is None: cols = [None]*6
    
    fig = plt.subplots.make_subplots(rows=1, cols=K, shared_xaxes=True, 
                                     column_titles=([f'Eta {l}' for l in range(K)]))

    for c in range(C):
        for m in range(M):
            for k in range(K):
                d = eta_approx[:,k,c,m]
                col = cols[(m if c==0 else (m+3))]
                fig.add_trace(go.Histogram(x=d, histnorm='probability', marker_color = col, 
                                           legendgroup = mut6[(m if c==0 else (m+3))], 
                                           showlegend = k==0, hoverinfo='name', 
                                           name = mut6[(m if c < 1 else (m+3))]), row = 1, col = k+1)
                

    fig.update_yaxes(showticklabels = False)
    fig.update_layout(barmode='overlay')
    fig.update_traces(opacity=0.7)
    
    return fig

def plot_mean_std(array):
    assert len(array.shape) == 3 or len(array.shape) == 4
    if len(array.shape) == 3: array = array[None,:,:,:]

    fig = plt.subplots.make_subplots(rows=1, cols=2, subplot_titles = ['mean', 'std'])
    fig.add_trace(go.Heatmap(z=array.mean((0, 1)).round(2), coloraxis='coloraxis'), row=1, col =1)
    fig.add_trace(go.Heatmap(z=array.std((0, 1)).round(2), coloraxis='coloraxis'), row=1, col =2)
    
    fig.update_layout(coloraxis=dict(colorscale = 'viridis'))
    return fig

def plot_cossim(tau_gt, tau_hat):
    # heatmap of cosine similarities
    fig = plt.subplots.make_subplots(
            rows=2, cols=2,
            column_widths=[0.5, 0.5], row_heights=[0.5, 0.5],
            specs=[[{"type": "heatmap", "rowspan": 2},  {"type": "heatmap"}   ],
                   [          None                   ,  {"type": "heatmap"}   ]]
          )
    
    fig.add_trace(go.Heatmap(z=cosine_similarity(tau_gt,tau_gt).round(2), coloraxis='coloraxis'), row=1, col =2)
    fig.update_xaxes(title_text="tau gt", row=1, col=2)
    fig.update_yaxes(title_text="tau gt", row=1, col=2)
    
    fig.add_trace(go.Heatmap(z=cosine_similarity(tau_hat,tau_hat).round(2), coloraxis='coloraxis'), row=2, col =2)
    fig.update_xaxes(title_text="tau hat", row=2, col=2)
    fig.update_yaxes(title_text="tau hat", row=2, col=2)
    
    cross = cosine_similarity(tau_hat, tau_gt)
    fig.update_xaxes(title_text="tau gt", row=1, col=1)
    fig.update_yaxes(title_text="tau hat", row=1, col=1)
    
    if cross.shape[0] < cross.shape[1]:
        cross = cross.T
        fig.update_xaxes(title_text="tau hat", row=1, col=1)
        fig.update_yaxes(title_text="tau gt", row=1, col=1)
    
    fig.add_trace(go.Heatmap(z=cross.round(2), coloraxis='coloraxis'), row=1, col =1)
    
    fig.update_layout(coloraxis=dict(colorscale = 'viridis'), showlegend=False, 
                      title = 'cosine distance of estimated signatures and ground truth')
    
    return fig
    
    
def save_gv(model):
    # render doen't work well. use `dot -Tpng model_graph > foo.png` instead
    gv = pm.model_graph.model_to_graphviz(model)
    gv.render(format = 'png')


def plot_bipartite(w, rescale = 10, main = '', ah=0, thresh = 0.01,
                   edge_cols = '#000000', node_cols=['#a64d79', '#45818e']):
    # create fully connected, directional bipartite graph 
    # input is JxK matrix st w[j,k] gives the (possibly 0) weight 
    # of edge spannig nodes j to k. 
    assert len(w.shape) == 2
    J,K = w.shape
    
    if isinstance(edge_cols, str):
        edge_cols = [edge_cols] * J*K
        
    else: 
        assert len(edge_cols) == J*K
        
    if np.any(w > 1):
        warnings.warn("Some w's >1. Edge width may be misleading.")
    
    if np.any(w < 0):
        warnings.warn("Some w's <0. Edge width may be misleading")
    
    
    y0s = np.arange(J) - np.arange(J).mean()
    y1s = np.arange(K) - np.arange(K).mean()
    node_y = np.array([[y0, y1, None] for y0 in y0s for y1 in y1s]).flatten()
    node_x = np.array([[0, 1, None] for y0 in y0s for y1 in y1s]).flatten()
    
    w = w.flatten()
    edges = w / np.max(w) * 10
    
    fig = go.Figure()
    
    # plot each edge with its weight
    
    i=-1
    for y0 in y0s:
        for y1 in y1s:
            i += 1
            source = (0.01, y0)
            target = (1-0.01, y1)
                
            if edges[i] > (thresh*10):
                fig.add_annotation(x=source[0], y = source[1],
                                   xref="x", yref="y",
                                   text="",
                                   showarrow=True,
                                   axref = "x", ayref='y',
                                   ax= target[0],
                                   ay= target[1],
                                   arrowhead = ah,
                                   arrowwidth= edges[i],
                                   arrowcolor=edge_cols[i]
                               )
    
            # add tiny jitter to show all overlapping points.
            fig.add_trace(go.Scatter(x=[0.5], y=[(y0+y1)/2] + pm.Normal.dist(0,0.001).random(),
                            marker=dict(size = 0.001),
                            hovertemplate=f'{w.flatten()[i].round(2)}',
                            mode='markers'))
            
    # plot nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                             marker=dict(size = 20, color=([node_cols[0], node_cols[1], node_cols[1]] * 3*J*K)),
                             hoverinfo='none',
                             mode='markers'))
    
    fig.update_layout(
                title=main,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
                   

    return fig

def plot_bipartite_K(weights):
    # normalize bipartite graph in J->K direction
    fig = plot_bipartite((weights/weights.sum(0)).round(2), main = 'K repairs J', ah=5)
    return fig

def plot_bipartite_J(weights):
    # normalize bipartite graph in K->J direction
    fig = plot_bipartite((weights.T/weights.sum(1)).T.round(2), direction = "back", main = 'J repaired by K', ah=5)
    return fig
    
def plot_nmut(nmut_dict):
    # dsets is a dict of nmut per sample per dataset
    # ex. {'train': train.sum(1)}
    
    fig = go.Figure()

    for dset in nmut_dict.keys():
        fig.add_trace(go.Box(y=nmut_dict[dset], name = dset))
    
    fig.update_layout(title_text="Number of mutations per sample in datasplit")
    return fig


def plot_pca(X, mcol=None):
    # X is row-data
    # mcol is (optional) colouring factor
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # pull out labels for categorical mcol
    if mcol.dtype!=float:
        col = mcol.astype('category').cat.codes
    else: 
        col = mcol
        mcol = mcol.round(2)
    
    # plot PC 1 & 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                             marker=dict(color=col, #set color equal to a variable
                                         colorscale='Agsunset',
                                         showscale=(mcol is not None),
                                         colorbar=dict(len=0.5)
                                        ),
                             customdata=mcol,
                             hovertemplate = '%{customdata}<extra></extra>',
                             mode='markers', name='pc1/2'))
    
    if hasattr(mcol, 'name'):
        fig.update_layout(title=f"coloured by {mcol.name}")
    fig.update_layout(xaxis_title="PC1",
                      yaxis_title="PC2")
    
    return fig

def plot_elbow_pca(X, n_comp=10, mcol=None):
    fig = make_subplots(rows=1, cols=2)
    # X is row-data
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    # plot PC elbow
    r = pca.explained_variance_ratio_
    fig.add_trace(go.Scatter(x=np.arange(len(r)), y=r, mode='lines+markers',
                             name = '% variance explained'), row=1, col=1)
    
    # plot PC 1 & 2
    fig.add_trace(go.Scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                             marker=dict(color=mcol, #set color equal to a variable
                                         colorscale='Agsunset',
                                         showscale=(mcol is not None),
                                         colorbar=dict(len=0.5)
                                        ),
                             mode='markers', name = "PC1/2"), row = 1, col = 2)
    
    
    return fig

def plot_fclust_scree(mat, metric = 'cosine', max_t = 10):
    d = pdist(mat, metric)
    Z = linkage(d, "ward")
    # from fcluster docs
    # flat clusters so that the original observations in 
    # each flat cluster have no greater a cophenetic distance than t.
    n_clust = [fcluster(Z, t=t, criterion='distance').max() for t in np.arange(1,max_t)]
    fig = go.Figure(go.Scatter(y = n_clust, x = np.arange(1,max_t)))
    fig.update_layout(yaxis_title = "number of clusters", xaxis_title="dendrogram cutoff")
    return fig

def pick_cutoff(a, metric='cosine', thresh=5):
    d = pdist(a, metric)
    Z = linkage(d, "ward")
    n_clust = np.array([fcluster(Z, t=t, criterion='distance').max() for t in np.arange(1,20)])
    return np.argmax(n_clust < thresh)


def map_to_palette(annotation, pal_list = ['Dark2','Set1','Set2','Set3']):
    # map all columns to a pallet entry
    # make sure to subset columns of annotation appropriately
    # ie. only categorical
    i=0
    luts = []
    for col in annotation.columns:
        lut = dict(zip(annotation[col].unique(), sns.color_palette(pal_list[i])))   
        annotation[col] = annotation[col].map(lut)
        luts.append(lut)
        i+=1
        i%=4
    
    return annotation, luts
