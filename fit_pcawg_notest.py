# fit_pcawg.py
from config import *
from utils import *
from plotting import *
from models import collapsed_model_factory



@extyaml
def fit_collapsed_model(train: np.ndarray, J: int, K: int,
                        alpha_bias: float, psi_bias: float, 
                        gamma_bias: float, beta_bias: float,
                        callbacks: list, n_steps: int, init_strategy: str, 
                        seed: int) -> (pm.model.Model, pm.variational.inference.ADVI):
    
    S = train.shape[0]
    N = train.sum(1)
    
    logging.debug(f"number of samples in corpus: {S}")
    logging.debug(f"mean number of mutations per sample: {N.mean()}")
    
    ## train loop
    with collapsed_model_factory(train, J, K, alpha_bias, psi_bias, 
                                 gamma_bias, beta_bias, init_strategy = init_strategy) as model:
        
        #wandb.log({'graphical model': wandb.Image(save_gv(model))})
        trace = pm.fit(n_steps, method = 'advi', callbacks = callbacks)
   
    return model, trace


@extyaml  
def cbs(*args, train=None, val=None, log_every=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            hat = approx.sample(100)
            
            wandb.log({         
                       "train alp": np.array([alp_B(train,B) for B in hat.B]),
                       "val alp": np.array([alp_B(val,B) for B in hat.B]),
                       "train lap": lap_B(train, hat.B),
                       "val lap": lap_B(val, hat.B)
                      })
    
    return [wandb_calls]


@extyaml
def load_counts(counts_fn, types_fn, type_subset=None):
    
    # probably don't want str input
    if isinstance(type_subset, str):
        warnings.warn("For str type_subset, each letter is matched. Are you sure you don't want list?", UserWarning)
        
    counts = pd.read_csv(counts_fn, index_col = [0], header = [0])[mut96]
    cancer_types = pd.read_csv(types_fn)
    cancer_types.columns=['type', 'guid']
    
    # subset guid by matching cancer type
    if type_subset is not None:
        
        # partial matches allowed
        sel = np.fromiter((map(any, zip(*[cancer_types.type.str.contains(x) for x in type_subset] ))), dtype = bool)
        
        # type should appear in the type column of the lookup 
        assert sel.any(), 'Cancer type subsetting yielded no selection. Check keywords?'
        
        counts = counts.loc[cancer_types.guid[sel]]

    return counts
    
@extyaml
def load_ref_taus(cosmic_fn, alex_local_fn, degas_local_fn):
    return [load_sigs(cosmic_fn, 'cosmic', sep = '\t'),
            load_sigs(alex_local_fn),
            load_sigs(degas_local_fn, 'cosmic')]
    

def main():
    wandb.init(project = 'pcawg-notest', #group = config['fit_pcawg_notest.py']['load_counts']['type_subset'],
               tags= config['fit_pcawg_notest.py']['load_counts']['type_subset'] )
    wandb.config.update(config)
    
    counts = load_counts()
    
    logging.debug(f'dim counts on read {counts.shape}')
    f = go.Figure().add_trace(go.Box(y=counts.sum(1)))
    f.update_layout(title = 'number of mutations per sample')
    wandb.log({'nmut': f})
    
    train, val  = split_by_count(counts.to_numpy())
    model, trace = fit_collapsed_model(train = train, callbacks = cbs(train = train, val = val))
    
    # unpack reference signature sets
    cosmic, pcawg_local, degas_local = load_ref_taus()

    # get signature names
    cosmic_names = np.array(cosmic.columns)
    pcawg_names = np.array(pcawg_local.columns)
    degas_names = np.array(degas_local.columns)
    
    # turn signatures to numpy for sanity
    cosmic = cosmic.to_numpy().T
    pcawg_local = pcawg_local.to_numpy().T
    degas_local = degas_local.to_numpy().T

    
    # get inferred estimates
    hat = trace.sample(1000)
    tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))
    W = np.einsum('tsj,tsjk -> tsjk', hat.theta, hat.A)
    
    # log all the plots
    wandb.log({'cosin similarity to Alexandrov et al. local signatures': plot_cossim(pcawg_local, tau_hat),
               'cosin similarity to Degasperi et al. local signatures': plot_cossim(degas_local, tau_hat),
               'cosin similarity to COSMIC signatures': plot_cossim(cosmic, tau_hat),
               
               'COSMIC >0.9 similarity': wandb.Table(cosmic_names[np.any(cosine_similarity(tau_hat,cosmic) > 0.9, axis = 0)]),
               'COSMIC >0.7 similarity': wandb.Table(cosmic_names[np.any(cosine_similarity(tau_hat,cosmic) > 0.7, axis = 0)]),
               'COSMIC phi >0.9 similarity': wandb.Table(cosmic_names[np.any(cosine_similarity(hat.phi.mean(0), get_phis(cosmic) ) > 0.9, axis = 0)]),
               'COSMIC phi >0.7 similarity': wandb.Table(cosmic_names[np.any(cosine_similarity(hat.phi.mean(0), get_phis(cosmic) ) > 0.7, axis = 0)]),

               'inferred signatures': plot_tau(tau_hat),
               'inferred phi': plot_phi(hat.phi.mean(0)),
               'inferred eta': plot_eta(hat.eta.mean(0)),

             })

    # bipartites
    n_comp = 5
    for i in range(n_comp):
        edges = pca.components_.reshape(-1, J, K)[i]
        edge_colours = np.tile('#f6b26b', J*K)
        edge_colours[edges.flatten() < 0] = '#6fa8dc'
        f = plot_bipartite(abs(edges), edge_cols = edge_colours, thresh = 0.1)
        f.update_layout(title=f"A component {i}")          

    
    # PCA elbow
    f = plot_elbow_pca(hat.A.mean(0)).update_layout('Variance explained for PC decomposition of A')
    wandb.log({f'PCA A elbow': f})
    f = plot_elbow_pca(W.mean(0)).update_layout('Variance explained for PC decomposition of W')
    wandb.log({f'PCA W elbow': f})

    # colour PCA with mcols
    for i in range(len(mcols.columns)):
        wandb.log({f'pca_ai {i}': plot_pca(hat.A.mean(0).reshape(-1, J*K), mcol = mcols[mcols.columns[i]]),
                   f'pca_tai {i}': plot_pca(W.mean(0).reshape(-1, J*K), mcol = mcols[mcols.columns[i]])
                  })

    # colour PCA with activities
    # color by theta for each context signature
    for j in range(J):
        fa = plot_pca(hat.A.mean(0).reshape(-1, J*K), mcol = hat.theta.mean(0)[:,j])
        fa.update_layout(title=f"PCA of A, coloured by activity of context signature {j}")
        fta = plot_pca(W.mean(0).reshape(-1, J*K), mcol = hat.theta.mean(0)[:,j])
        fta.update_layout(title=f"PCA of W, coloured by activity of context signature {j}")
        wandb.log({f'pca_aj {j}': fa, 'pca_taj {j}': fta})
        
    
    # r^2 of components, etc
    
    pca = PCA(n_components=n_comp)
    A_pca = pca.fit_transform(hat.A.mean(0).reshape(-1, J*K))
    W_pca = pca.fit_transform(W.mean(0).reshape(-1, J*K))

    c = pd.concat([mcols,
                   pd.DataFrame(A_pca, index = mcols.index, columns = [f'compunent_{i}' for i in range(n_comp)]),
                   pd.DataFrame(hat.theta.mean(0), index = mcols.index, columns = [f'theta_{j}' for j in range(J)])],
                   axis=1).corr()
    f = go.Figure().add_trace(go.Heatmap(z=c**2, colorscale = 'agsunset', 
                                         x = c.columns, y=c.columns))
    f.update_layout(title=f"r^2 for variables, PCA A")
    wandb.log({f'PCA A corr': f})

    c = pd.concat([mcols,
                   pd.DataFrame(W_pca, index = mcols.index, columns = [f'compunent_{i}' for i in range(n_comp)]),
                   pd.DataFrame(hat.theta.mean(0), index = mcols.index, columns = [f'theta_{j}' for j in range(J)])],
                   axis=1).corr()
    f = go.Figure().add_trace(go.Heatmap(z=c**2, colorscale = 'agsunset', 
                                         x = c.columns, y=c.columns))
    f.update_layout(title=f"r^2 for variables, PCA A")
    wandb.log({f'PCA W corr': f})

    
    
if __name__ == '__main__':
    main()