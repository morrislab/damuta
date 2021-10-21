# inference.py
from .sim import *
from .model_factory import *
from .plotting import * 

models = {'tandem_lda': tandem_lda,
          'tandtiss_lda': tandtiss_lda,
          'vanilla_lda': vanilla_lda
         }

def infer(train, model_args, pymc3_args, cbs=None):
    
    margs = model_args.copy()
    pargs = pymc3_args.copy()
    
    assert margs['model_sel'] in models.keys(), \
        f"Unrecognized model selection. model_sel should be one of [{models.keys()}]"
    assert pargs['method'] in ['advi', 'fullrank_advi'], \
        f"Unrecognized approximation method selection. method should be one of ['advi', 'fullrank_advi']"
    
    # TODO: fix pymc3 seeding 
    np.random.seed(pymc3_args['random_seed']) 
    pm.set_tt_rng(pymc3_args['random_seed'])  
    
    model = models[margs.pop('model_sel')](train = train, **margs)
    
    with model: 
        if pargs['method'] == 'advi':
            opt = pm.ADVI(random_seed = pargs.pop('random_seed'))
        elif pargs['method'] == 'fullrank_advi':
            opt = pm.FullRankADVI(random_seed = pargs.pop('random_seed'))
        
        pargs.pop('method')
        trace = opt.fit(**pargs, callbacks = cbs)
        
    return model, opt
    

def cbs(*args):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
    
    return [wandb_calls]

