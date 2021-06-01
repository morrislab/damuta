#import matplotlib.pyplot as mplt
#from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pymc3 as pm
import arviz as az
from config import *
from utils import *
from sklearn.metrics.pairwise import cosine_similarity
import plotly as plt
import plotly.graph_objects as go
from plotly.colors import n_colors
plt.io.templates.default = "none"

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
phi_col = np.repeat(['green', 'blue'], 16)
eta_col = np.repeat(['orange', 'lightblue'], 3)

def plot_sigs(sigs, xlab, cols):
    assert len(sigs.shape) == 2
    
    fig = plt.subplots.make_subplots(rows=sigs.shape[0], cols=1, shared_xaxes=True)
    
    for s in range(sigs.shape[0]):
        fig.add_trace(go.Bar(x=xlab, y=sigs[s], hoverinfo='y', showlegend = False,
                             textposition='auto', marker_color=cols,
                        
                      ), row = s+1, col = 1 )

    fig.update_xaxes(tickangle=-45, matches = 'x')
    return fig

def plot_tau(tau):
    if len(tau.shape) == 1: 
        tau = tau.reshape(1,-1)
    return plot_sigs(tau, mut96, tau_col)

def plot_phi(phi):
    if len(phi.shape) == 1: 
        phi = phi.reshape(1,-1)
    return plot_sigs(phi, mut32, phi_col)

def plot_eta(eta, cols = eta_col):
    assert len(eta.shape) == 3
    e = np.hstack([eta[0], eta[1]])
    K = e.shape[0]
    fig = plt.subplots.make_subplots(rows=K, cols=1, shared_xaxes=True,
                                     row_titles=([f'Eta {l}' for l in range(K)]))
    for k in range(K):
        fig.add_trace(go.Bar(x=mut6, y=e[k], hoverinfo='y', showlegend = False,
                             textposition='auto', marker_color=cols), row = k+1,col = 1)
    return fig

    
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

def plot_tau_cos(tau_gt, tau_hat):

    
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
    
    fig.update_layout(coloraxis=dict(colorscale = 'viridis'), showlegend=False, height=1000,
                      title = 'cosine distance of estimated signatures and ground truth')
    
    return fig
    
    
def save_gv(model, fn = 'model_graph'):
    gv = pm.model_graph.model_to_graphviz(model)
    gv.format = 'png'
    return gv.render(filename=fn)


def plot_bipartite(w, rescale = 10, direction = 'forward'):
    # create fully connected, directional bipartite graph 
    # input is JxK matrix st w[j,k] gives the (possibly 0) weight 
    # of edge spannig nodes j to k. 
    assert len(w.shape) == 2
    J,K = w.shape
    
    y0s = -np.arange(J) + np.arange(J).mean()
    y1s = -np.arange(K) + np.arange(K).mean()
    node_y = np.array([[y0, y1, None] for y0 in y0s for y1 in y1s]).flatten()
    node_x = np.array([[0, 1, None] for y0 in y0s for y1 in y1s]).flatten()
    
    fig = go.Figure()
    
    # plot each edge with its weight
    i=-1
    for y0 in y0s:
        for y1 in y1s:
            i+=1
            
            if direction == 'forward': 
                source = (0.003*rescale, y0)
                target = (1-(0.003*rescale), y1)
            else:
                target = (0.003*rescale, y0)
                source = (1-(0.003*rescale), y1)
                
            if w.flatten()[i] != 0:
                fig.add_annotation(x=source[0], y = source[1],
                                   xref="x", yref="y",
                                   text="",
                                   showarrow=True,
                                   axref = "x", ayref='y',
                                   ax= target[0],
                                   ay= target[1],
                                   arrowhead = 5,
                                   arrowwidth=w.flatten()[i] * rescale,
                                   arrowcolor='rgb(6, 144, 143)'
                               )
    
                # add tiny jitter to show all overlapping points.
                fig.add_trace(go.Scatter(x=[0.5], y=[(y0+y1)/2] + pm.Normal.dist(0,0.001).random(),
                                marker=dict(size = 0.001),
                                hovertemplate=f'{w.flatten()[i]}',
                                mode='markers'))
                
    # plot nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            marker=dict(size = rescale * 4, color='#e8ec67'),
                            hoverinfo='none',
                            mode='markers'))
    
    fig.update_layout(
                title='<br>Network graph made with Python',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
                   

    return fig

def plot_bipartite_K(weights):
    # normalize bipartite graph in J->K direction
    fig = plot_bipartite(weights/weights.sum(0))
    return fig

def plot_bipartite_J(weights):
    # normalize bipartite graph in K->J direction
    fig = plot_bipartite((weights.T/weights.sum(1)).T, direction = "back")
    return fig
    

#def plot_diagnostics(trace, J, K, fn = 'model_diagnostics.pdf'):
#    
#    with PdfPages(fn) as pdf:
#        # attach metadata (as pdf note) to page
#        #pdf.attach_note("sticky note!")  
#    
#        # ELBO 
#        plt.plot(trace.hist)
#        plt.tight_layout()
#        pdf.savefig()
#        plt.close()
#        
#        # phi hat
#        for j in range(J):
#            plt.subplot(J,1,j+1)
#            for c in range(32):
#                sns.distplot(trace.approx.sample(1000)['phi'][:, j, c], kde=False, hist=True)
#            plt.title(f'Phi density estimates for topic {j}')
#        plt.tight_layout()
#        pdf.savefig()
#        plt.close()
#        
#        # eta hat
#        for k in range(K):
#            plt.subplot(K,1,k+1)
#            for m in range(3):
#                    sns.distplot(trace.approx.sample(1000)['eta'][:, 31, k, m], kde=False, hist=True)
#            plt.title(f'Eta density estimates for topic {k}')
#        plt.tight_layout()
#        pdf.savefig()
#        plt.close()
#                
#        # A summary
#        a = trace.approx.sample(1000)['A']
#        ax = sns.heatmap(np.round(a.mean((0, 1)), 2), annot=True, cmap="YlGnBu")
#        plt.tight_layout()
#        pdf.savefig()
#        plt.close()
#        
#        ax = sns.heatmap(np.round(a.std((0, 1)), 2), annot=True, cmap="YlGnBu")
#        plt.tight_layout()
#        pdf.savefig()
#        plt.close()
#        


