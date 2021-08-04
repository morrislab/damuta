from utils import *
from plotting import * 
from sim_data import sim_parametric
from models import collapsed_model_factory



@extyaml
def fit_collapsed_model(corpus_obs: np.ndarray, J: int, K: int,
                        alpha_bias: float, psi_bias: float, 
                        gamma_bias: float, beta_bias: float,
                        callbacks: list, n_steps: int, seed: int) -> (pm.model.Model, pm.variational.inference.ADVI):
  
    
    S = corpus_obs.shape[0]
    N = corpus_obs.sum(1)
    
    logging.debug(f"number of samples in corpus: {S}")
    logging.debug(f"mean number of mutations per sample: {N.mean()}")
    
    with collapsed_model_factory(corpus_obs, J, K, alpha_bias, 
                                 psi_bias, gamma_bias, beta_bias) as model:
        
        #wandb.log({'graphical model': wandb.Image(save_gv(model))})

        trace = pm.ADVI()
        trace.fit(n_steps, callbacks = callbacks)
   
    return model, trace


@extyaml  
def cbs(*args, train=None, val=None, params_gt=None, tau_gt=None, log_every=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            hat = approx.sample(1000)
            tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))   
            act_hat = np.einsum('sj,sjk->sjk', hat.theta.mean(0), hat.A.mean(0))
            
            wandb.log({         
                       'train alp': alp_B(train, hat.B.mean(0)),
                       'val alp': alp_B(val, hat.B.mean(0)),
                       'model alp': alp_B(train + val, params_gt['B']),
                       'phi posterior': plot_phi_posterior(hat.phi, cols=None),
                       'eta posterior': plot_eta_posterior(hat.eta, cols=None),
                
                       'tau similarities': plot_cossim(tau_gt, tau_hat),
                       'phi similarities': plot_cossim(params_gt['phi'], hat.phi.mean(0)),
                       'eta similarities': plot_cossim(flatten_eta(params_gt['eta']), flatten_eta(hat.eta.mean(0))),
                        
                       'inferred tau': plot_tau(tau_hat),
                       'inferred phi': plot_phi(hat.phi.mean(0)),
                       'inferred eta': plot_eta(hat.eta.mean(0)),

                       'estimated activities': plot_mean_std(act_hat),
                        
                       'average availability': plot_bipartite_J(act_hat.mean(0)),
                       'average recruitment': plot_bipartite_K(act_hat.mean(0)),
                       'availability in sample 0': plot_bipartite_J(act_hat[0]),
                       'recruitment in sample 0': plot_bipartite_K(act_hat[0]),

                      })
    
    return [wandb_calls]


def main():
    wandb.init()
    wandb.config.update(config)
    
    corpus, params = sim_parametric()
    train, val  = split_by_count(corpus)
    tau = get_tau(params['phi'], params['eta'])
    
    wandb.log({'gt tau': plot_tau(tau),
               'gt phi': plot_phi(params['phi']),
               'gt eta': plot_eta(params['eta']),
               'gt activities': plot_mean_std(np.einsum('sj,sjk->sjk', params['theta'], params['A'])),
              })
   

    model, trace = fit_collapsed_model(corpus_obs = train, 
                                       callbacks = cbs(train = train, val = val, 
                                                       params_gt = params, tau_gt = tau))
    
    
if __name__ == '__main__':
    main()
