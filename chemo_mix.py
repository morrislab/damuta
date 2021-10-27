import pandas as pd
import numpy as np
import damuta as da
from damuta.plotting import *
import wandb
import os


if __name__ == "__main__":

    dataset_args, model_args, pymc3_args = da.load_config("config/chemo-mix.yaml")

    # configure run with defaults and sweep args
    run = wandb.init()
    run.config.setdefaults({**dataset_args, **model_args, **pymc3_args})
    
    # override config with any arguments set up in wandb run
    # (useful for managing wandb sweeps) Adamo Young 2021
    run_config_d = dict(run.config)
    for d in [dataset_args, model_args, pymc3_args]:
        for k,v in d.items():
            d[k] = run_config_d[k]
    
    # load data 
    # counts can be combined for free, annotation is ugly when columns don't match
    counts, annotation = load_datasets(dataset_args)
    
    trn, val, tst1, tst2 = da.split_data(counts, rng=np.random.default_rng(dataset_args['data_seed']))
    da.log_data_summary(trn, val, tst1, tst2)
    
    # transform type to categorical (only necessary for hirearchical model)
    if model_args['model_sel'] == 'tandtiss_lda':
        model_args['type_codes'] = pd.Categorical(annotation.loc[trn.index]['pcawg_class']).codes

    # perform inference
    # TODO: implement cbs that avoid pickle problem
    model, trace = da.infer(trn.to_numpy(), model_args, pymc3_args, cbs = [da.log_elbo])

    # save final model and trace
    os.makedirs("ckpt/", exist_ok=True)
    da.save_checkpoint("ckpt/" + wandb.run.id, model=model,trace=trace, dataset_args=dataset_args, 
                    model_args=model_args, pymc3_args=pymc3_args, run_id=wandb.run.id)

    # plots & evaluation
    cosmic = da.load_sigs('data/COSMIC_v3.2_SBS_GRCh37.txt')
    hat = trace.approx.sample(100)
    
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