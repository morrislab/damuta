# fit_pcawg.py
from config import *
from utils import *
from plotting import *
from models import collapsed_model_factory



@extyaml
def fit_collapsed_model(train: np.ndarray, test: np.ndarray, J: int, K: int,
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
       
    
    ## refit just theta and A to test data shape 
    
    ts_p, ts_pp = split_by_count(test)
    
    with collapsed_model_factory(ts_p, J, K, alpha_bias, psi_bias, 
                                 gamma_bias, beta_bias, init_strategy = init_strategy,
                                 sig_obs = {'phi': trace.sample_node(model.phi).eval(), 
                                            'eta': trace.sample_node(model.eta).eval().swapaxes(0,1)}) as m_ts:

        trace_ts = pm.fit(n_steps//2, method = 'advi', callbacks = [test_cbs])
    
    Bs = trace_ts.sample(1000).B
    
    wandb.log({"ts' lap": lap_B(ts_p, Bs),
               "ts'' lap": lap_B(ts_pp, Bs),
               'test lap': lap_B(test, Bs)
              })
   
    return model, trace


        
def test_cbs(approx, losses, i):
    wandb.log({'test refit ELBO': losses[-1]})

@extyaml  
def cbs(*args, train=None, val=None, ref_taus, log_every=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            # unpack reference signature sets
            tau_cosmic, tau_alex, tau_degas = ref_taus
            
            hat = approx.sample(1000)
            tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))
            act_hat = np.einsum('sj,sjk->sjk', hat.theta.mean(0), hat.A.mean(0))
            
            wandb.log({         
                       "tr' alp": alp_B(train, hat.B.mean(0)),
                       "tr'' alp": alp_B(val, hat.B.mean(0)),
                       "tr' lap": lap_B(train, hat.B),
                       "tr'' lap": lap_B(val, hat.B),
                       'phi posterior': plot_phi_posterior(hat.phi, cols=None),
                       'eta posterior': plot_eta_posterior(hat.eta, cols=None),
                
                       'cosin similarity to Alexandrov et al. local signatures': plot_cossim(tau_alex, tau_hat),
                       'cosin similarity to Degasperi et al. local signatures': plot_cossim(tau_degas, tau_hat),
                       'cosin similarity to COSMIC signatures': plot_cossim(tau_cosmic, tau_hat),
                       'phi similarities': go.Figure(go.Heatmap(z=cosine_similarity(hat.phi.mean(0), 
                                                                                  hat.phi.mean(0)).round(2), 
                                                              colorscale = 'viridis')),
                       'eta similarities': go.Figure(go.Heatmap(z=cosine_similarity(flatten_eta(hat.eta.mean(0)),
                                                                                  flatten_eta(hat.eta.mean(0))).round(2), 
                                                              colorscale = 'viridis')),
                        
                       'inferred signatures': plot_tau(tau_hat),
                       'inferred phi': plot_phi(hat.phi.mean(0)),
                       'inferred eta': plot_eta(hat.eta.mean(0)),

                       'estimated recruitment': plot_mean_std(act_hat),
                       'mean recruitment': plot_bipartite(act_hat.mean(0).round(2)),
                       'median recruitment': plot_bipartite(np.median(act_hat,0).round(2)),
                
                      })
    
    return [wandb_calls]


@extyaml
def load_counts(counts_fn, types_fn, type_subset=None):

    counts = pd.read_csv(counts_fn, index_col = [0], header = [0])[mut96]
    cancer_types = pd.read_csv(types_fn)
    cancer_types.columns=['type', 'guid']
    
    # subset guid by matching cancer type
    if type_subset is not None:
        
        # partial matches allowed
        sel = pd.read_csv('pcawg_cancer_types.csv').type.str.contains(type_subset)
        
        # type should appear in the type column of the lookup 
        assert sel.any(), 'Cancer type subsetting yielded no selection. Check keywords?'
        
        counts = counts.loc[cancer_types.guid[sel]]

    return counts.to_numpy()
    
@extyaml
def load_ref_taus(cosmic_fn, alex_local_fn, degas_local_fn):
    return [load_sigs(cosmic_fn, 'cosmic', sep = '\t').to_numpy().T,
            load_sigs(alex_local_fn).to_numpy().T,
            load_sigs(degas_local_fn, 'cosmic').to_numpy().T]
    

def main():
    wandb.init(project = 'da-pcawg', group = config['fit_pcawg.py']['load_counts']['type_subset'],
               tags= [config['fit_pcawg.py']['load_counts']['type_subset']] )
    wandb.config.update(config)
    
    counts = load_counts()
    logging.debug(f'dim counts on read {counts.shape}')
    train, test = split_by_S(counts)
    
    wandb.log({'nmut per sample': plot_nmut({'train': train.sum(1), 'test': test.sum(1)})})
    
    tr_p, tr_pp  = split_by_count(train)
    
    wandb.log({'n samples train': train.shape[0],
               'n samples test': test.shape[0]
              })

    model, trace = fit_collapsed_model(train = train, test = test,
                                       callbacks = cbs(train = tr_p, val = tr_pp, 
                                                       ref_taus = load_ref_taus() 
                                       ))
    
    
if __name__ == '__main__':
    main()