import pandas as pd
import numpy as np
import damuta as da
from damuta.plotting import *
from damuta.utils import alp_B, load_config
from damuta import ch_dirichlet
import theano
import theano.tensor as tt
import wandb
import os

if __name__ == "__main__":

    dataset_args, model_args, pymc3_args = load_config("config/foo.yaml")
    
    # init run
    run = wandb.init()
    
    # configure run with defaults and sweep args
    run.config.setdefaults({**dataset_args, **model_args, **pymc3_args})
    
    # override config with any arguments set up in wandb run
    # (useful for managing wandb sweeps) Adamo Young 2021
    run_config_d = dict(run.config)
    for d in [dataset_args, model_args, pymc3_args]:
        for k,v in d.items():
            d[k] = run_config_d[k]
            
    np.random.seed(model_args['model_seed'])
    pm.set_tt_rng(model_args['model_seed']) 

    # load data 
    counts = da.load_dataset(**dataset_args)
    trn, val, tst1, tst2 = da.split_data(counts, rng=np.random.default_rng(dataset_args['data_seed']))
    
    summary_table = wandb.Table(columns=['dataset', 'n samples', 'mean nmut', 'median nmut', 'min nmut', 'max nmut'],
                                data = [['train', trn.shape[0], trn.sum(1).mean(), np.median(trn.sum(1)), trn.sum(1).min(), trn.sum(1).max()],
                                        ['val', val.shape[0], val.sum(1).mean(), np.median(val.sum(1)), val.sum(1).min(), val.sum(1).max()],
                                        ['test1', tst1.shape[0], tst1.sum(1).mean(), np.median(tst1.sum(1)), tst1.sum(1).min(), tst1.sum(1).max()],
                                        ['test2', tst2.shape[0], tst2.sum(1).mean(), np.median(tst2.sum(1)), tst2.sum(1).min(), tst2.sum(1).max()] 
                                       ])
    wandb.log({'dataset summary': summary_table})
    
    # prep data
    model_args.pop('model_sel')
    model_args.pop('model_seed')
    
    cosmic = load_sigs("data/COSMIC_v3.2_SBS_GRCh37.txt")
    annotation = pd.read_csv(dataset_args['annotation_fp'], index_col = 0, header = 0).loc[trn.index]
    model_args['type_codes'] = pd.Categorical(annotation.type).codes
    
    def log_elbo(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
    def log_loss(*args, trn = trn, val = val, log_every = 1000):
        approx, losses, i = args
        
        if i % log_every == 0:
            
            B = approx.sample(100).B.mean(0)
        
            wandb.log({'trn_alp': alp_B(trn.to_numpy(), B),
                       'val_alp': alp_B(val.to_numpy(), B)
                      })
        

    model = da.tandtiss_lda(train =trn.to_numpy(), **model_args)
    
    with model:
        trace = pm.fit(25000, method = 'advi', random_seed = 10, callbacks = [log_elbo, log_loss])

    fp = "ckpt/" 
    os.makedirs(fp, exist_ok=True)
    save_checkpoint(fp + wandb.run.id, model=model,trace=trace, dataset_args=dataset_args, 
                    model_args=model_args, pymc3_args=pymc3_args)

    hat = trace.sample(100)
    
    wandb.log({
        # plot inferred signatures
        'phis': plot_phi(hat.phi.mean(0)),
        'map phi to cosmic': profile_sigs(hat.phi.mean(0), da.get_phi(cosmic.to_numpy()), refidx=cosmic.index),
        'etas': plot_eta(hat.eta.mean(0)),
        'map eta to cosmic': profile_sigs(hat.eta.mean(0).reshape(-1,6), da.get_eta(cosmic.to_numpy()).reshape(-1,6), refidx=cosmic.index),

        # distribution of signature separation
        # phi separation
        # eta separation
       
        # plot clustering
        ## A
        
        ## gamma
    
        
        ## M
        ## theta
        ## W        
    
    })
    
    
    
    
    # fit model for train samples
    #model, trace = da.infer(trn.to_numpy(), model_args, pymc3_args)