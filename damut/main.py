# main.py
from .sim import *
from .model_factory import *
from .plotting import * 

def infer(train, model_args={}, pymc3_args={}, cbs=None):
    
    models = {'tandem_lda': tandem_lda,
              'tandtiss_lda': tandtiss_lda
             }
    
    assert model_args['model_sel'] in models.keys(), \
        f"Unrecognized model selection. model_sel should be one of [{models.keys()}]"
        
    np.random.seed(model_args['model_seed']) 
    pm.set_tt_rng(model_args['model_seed'])  
    pymc3_args['random_seed'] = pymc3_args.pop('pymc3_seed')
    model_args.pop('model_seed') # TODO: remove this seeding hack with next pymc3 release
           
    model = models[model_args.pop('model_sel')](train = train, **model_args)
    with model: 
        trace = pm.fit(**pymc3_args, callbacks = cbs)
        
    return model, trace
    
def load_dataset(dataset_sel, counts_fp=None, annotation_fp=None, annotation_subset=None, data_rng=None,
                 data_seed = None, sig_defs_fp=None, sim_S=None, sim_N=None, sim_I=None, sim_tau_hyperprior=None,
                 sim_J=None, sim_K=None, sim_alpha_bias=None, sim_psi_bias=None, sim_gamma_bias=None, sim_beta_bias=None):
    # load counts, or simulated data - as specified by dataset_sel
    
    # seed -> rng as per https://albertcthomas.github.io/good-practices-random-number-generators/
    data_rng = np.random.default_rng(data_seed)
    
    if dataset_sel == 'load_counts':
        dataset = load_counts(counts_fp)
        annotation = pd.read_csv(annotation_fp, index_col = 0, header = 0)
        dataset = subset_samples(dataset, annotation, annotation_subset)
    
    elif dataset_sel == 'sim_from_sigs':
        sig_defs = load_sigs(sig_defs_fp)
        dataset, sim_params = sim_from_sigs(sig_defs, sim_tau_hyperprior, sim_S, sim_N, sim_I, data_rng)
    
    elif dataset_sel == 'sim_parametric':
        dataset, sim_params = sim_parametric(sim_J,sim_K,sim_S,sim_N,sim_alpha_bias,sim_psi_bias,sim_gamma_bias,sim_beta_bias,data_rng)
        
    else:
        assert False, 'dataset selection not recognized'
    
    return dataset

def cbs(*args, train=None, val=None, cosmic=None, log_every=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            hat = approx.sample(1000)
            tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))
            W_hat = np.einsum('sj,sjk->sjk', hat.theta.mean(0), hat.A.mean(0))
            
            wandb.log({         
                       "trn alp": alp_B(train, hat.B.mean(0)),
                       "val alp": alp_B(val, hat.B.mean(0)),
                       "trn lap": lap_B(train, hat.B),
                       "val lap": lap_B(val, hat.B),
                       'phi posterior': plot_phi_posterior(hat.phi, cols=None),
                       'eta posterior': plot_eta_posterior(hat.eta, cols=None),
                
                       #'cosin similarity to Alexandrov et al. local signatures': plot_cossim(tau_alex, tau_hat),
                       #'cosin similarity to Degasperi et al. local signatures': plot_cossim(tau_degas, tau_hat),
                       'cosin similarity to COSMIC signatures': plot_cossim(cosmic, tau_hat),
                       'phi similarities': go.Figure(go.Heatmap(z=cosine_similarity(hat.phi.mean(0), hat.phi.mean(0)).round(2), 
                                                              colorscale = 'viridis')),
                       'eta similarities': go.Figure(go.Heatmap(z=cosine_similarity(flatten_eta(hat.eta.mean(0)),
                                                                                  flatten_eta(hat.eta.mean(0))).round(2), 
                                                              colorscale = 'viridis')),
                        
                       'inferred signatures': plot_tau(tau_hat),
                       'inferred phi': plot_phi(hat.phi.mean(0)),
                       'inferred eta': plot_eta(hat.eta.mean(0)),

                       #'estimated recruitment': plot_mean_std(W_hat),
                       #'mean recruitment': plot_bipartite(W_hat.mean(0).round(2)),
                       #'median recruitment': plot_bipartite(np.median(W_hat,0).round(2)),
                
                      })
    
    return [wandb_calls]

