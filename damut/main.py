# main.py
from .sim import *
from .model_factory import * 

def infer(train, cbs=None, model_args={}, pymc3_args={}):
    print('hello!')
    wandb.log({'uwu': 'a message!', 
               'fake_metric' : model_args['J']})

def load_dataset(dataset_sel, counts_fp=None, annotation_fp=None, annotation_subset=None, sim_rng=None,
                 sig_defs_fp=None, sim_seed=None, sim_S=None, sim_N=None, sim_I=None, sim_tau_hyperprior=None,
                 sim_J=None, sim_K=None, sim_alpha_bias=None, sim_psi_bias=None, sim_gamma_bias=None, sim_beta_bias=None):
    # load counts, or simulated data - as specified by dataset_sel
    if dataset_sel == 'load_counts':
        dataset = load_counts(counts_fp)
        annotation = pd.read_csv(annotation_fp, index_col = 0, header = 0)
        dataset = subset_samples(dataset, annotation, annotation_subset)
    
    elif dataset_sel == 'sim_from_sigs':
        sig_defs = load_sigs(sig_defs_fp)
        dataset, sim_params = sim_from_sigs(sig_defs, sim_tau_hyperprior, sim_S, sim_N, sim_I, sim_rng)
    
    elif dataset_sel == 'sim_parametric':
        dataset, sim_params = sim_parametric(sim_J,sim_K,sim_S,sim_N,sim_alpha_bias,sim_psi_bias,sim_gamma_bias,sim_beta_bias,sim_rng)
        
    else:
        assert False, 'dataset selection not recognized'
    
    return dataset


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
        
    wandb.log({'test refit ELBO': losses[-1]})

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



def main():
    wandb.init(project = 'da-pcawg', #group = config['fit_pcawg.py']['load_counts']['type_subset'],
               tags= config['fit_pcawg.py']['load_counts']['type_subset'] )
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