from utils import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import n_colors
plt.io.templates.default = "none"

idx96 = pd.MultiIndex.from_tuples([
            ('C>A', 'ACA'), ('C>A', 'ACC'), ('C>A', 'ACG'), ('C>A', 'ACT'), 
            ('C>A', 'CCA'), ('C>A', 'CCC'), ('C>A', 'CCG'), ('C>A', 'CCT'), 
            ('C>A', 'GCA'), ('C>A', 'GCC'), ('C>A', 'GCG'), ('C>A', 'GCT'), 
            ('C>A', 'TCA'), ('C>A', 'TCC'), ('C>A', 'TCG'), ('C>A', 'TCT'), 
            ('C>G', 'ACA'), ('C>G', 'ACC'), ('C>G', 'ACG'), ('C>G', 'ACT'), 
            ('C>G', 'CCA'), ('C>G', 'CCC'), ('C>G', 'CCG'), ('C>G', 'CCT'), 
            ('C>G', 'GCA'), ('C>G', 'GCC'), ('C>G', 'GCG'), ('C>G', 'GCT'), 
            ('C>G', 'TCA'), ('C>G', 'TCC'), ('C>G', 'TCG'), ('C>G', 'TCT'), 
            ('C>T', 'ACA'), ('C>T', 'ACC'), ('C>T', 'ACG'), ('C>T', 'ACT'), 
            ('C>T', 'CCA'), ('C>T', 'CCC'), ('C>T', 'CCG'), ('C>T', 'CCT'), 
            ('C>T', 'GCA'), ('C>T', 'GCC'), ('C>T', 'GCG'), ('C>T', 'GCT'), 
            ('C>T', 'TCA'), ('C>T', 'TCC'), ('C>T', 'TCG'), ('C>T', 'TCT'), 
            ('T>A', 'ATA'), ('T>A', 'ATC'), ('T>A', 'ATG'), ('T>A', 'ATT'), 
            ('T>A', 'CTA'), ('T>A', 'CTC'), ('T>A', 'CTG'), ('T>A', 'CTT'), 
            ('T>A', 'GTA'), ('T>A', 'GTC'), ('T>A', 'GTG'), ('T>A', 'GTT'), 
            ('T>A', 'TTA'), ('T>A', 'TTC'), ('T>A', 'TTG'), ('T>A', 'TTT'), 
            ('T>C', 'ATA'), ('T>C', 'ATC'), ('T>C', 'ATG'), ('T>C', 'ATT'), 
            ('T>C', 'CTA'), ('T>C', 'CTC'), ('T>C', 'CTG'), ('T>C', 'CTT'), 
            ('T>C', 'GTA'), ('T>C', 'GTC'), ('T>C', 'GTG'), ('T>C', 'GTT'), 
            ('T>C', 'TTA'), ('T>C', 'TTC'), ('T>C', 'TTG'), ('T>C', 'TTT'), 
            ('T>G', 'ATA'), ('T>G', 'ATC'), ('T>G', 'ATG'), ('T>G', 'ATT'), 
            ('T>G', 'CTA'), ('T>G', 'CTC'), ('T>G', 'CTG'), ('T>G', 'CTT'), 
            ('T>G', 'GTA'), ('T>G', 'GTC'), ('T>G', 'GTG'), ('T>G', 'GTT'), 
            ('T>G', 'TTA'), ('T>G', 'TTC'), ('T>G', 'TTG'), ('T>G', 'TTT')],
            names=['Type', 'Subtype'])
    
mut96 = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T', 'C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T', 
         'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T', 
         'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T', 'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T', 
         'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T', 'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T', 
         'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T', 
         'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T', 
         'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T', 
         'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 
         'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T', 'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T', 
         'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T', 'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T', 
         'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T', 
         'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']

mut32 = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT', 
         'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT', 
         'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT', 
         'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']

mut16 = ['A_A', 'A_C', 'A_G', 'A_T', 'C_A', 'C_C', 'C_G', 'C_T', 
         'G_A', 'G_C', 'G_G', 'G_T', 'T_A', 'T_C', 'T_G', 'T_T']

mut6 = ['C>A','C>G','C>T','T>A','T>C','T>G']

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
    # TxCxKxM dimension df yields CxK subplots, M traces 
    assert len(eta_approx.shape) == 4
    T, C, K, M  = eta_approx.shape
    assert C==2
    if cols is None: cols = [None]*6
    
    fig = plt.subplots.make_subplots(rows=1, cols=K, shared_xaxes=True, 
                                     column_titles=([f'Eta {l}' for l in range(K)]))

    for c in range(C):
        for m in range(M):
            for k in range(K):
                d = eta_approx[:,c,k,m]
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
    
    
def save_gv(model, fn = 'model_graph'):
    gv = pm.model_graph.model_to_graphviz(model)
    gv.format = 'png'
    return gv.render(filename=fn)


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
            source = (0.03, y0)
            target = (1-0.03, y1)
                
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
                             marker=dict(size = 40, color=([node_cols[0], node_cols[1], node_cols[1]] * 3*J*K)),
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