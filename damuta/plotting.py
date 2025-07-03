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

def plot_cosmic_signatures(sigs, pal=None, aspect=5):
    if pal is None:
        pal = cosmic_palette
    return plot_signatures(sigs, pal, aspect)

def plot_damage_signatures(sigs, pal=None, aspect=3):
    if pal is None:
        pal = damage_palette
    return plot_signatures(sigs, pal, aspect)

def plot_misrepair_signatures(sigs, pals=None, aspect=1):
    if pal is None:
        pal = misrepair_palette
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

