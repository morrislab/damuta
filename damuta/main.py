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
    
    # TODO: fix pymc3 seeding 
    np.random.seed(pymc3_args['random_seed']) 
    pm.set_tt_rng(pymc3_args['random_seed'])  
    
           
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

def cbs(*args):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
    
    return [wandb_calls]

