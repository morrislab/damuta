import pandas as pd
import numpy as np
import damut as da
from damut.plotting import *
from damut import ch_dirichlet
import theano
import theano.tensor as tt
import wandb


if __name__ == "__main__":
    

    run = wandb.init(project = 'tandtiss')
    
    dataset_args, model_args, pymc3_args = da.load_config("config/tandtiss-defaults.yaml")
    run.config.update({**dataset_args, **model_args, **pymc3_args, 'device': theano.config.device})
    
    counts = da.load_dataset(**dataset_args)
    trn, val, tst1, tst2 = da.split_data(counts, rng=dataset_args['data_rng'])

    model_args.pop('model_sel')
    model_args.pop('model_seed')
    
    cosmic = load_sigs("data/COSMIC_v3.2_SBS_GRCh37.txt")
    annotation = pd.read_csv(dataset_args['annotation_fp'], index_col = 0, header = 0).loc[trn.index]
    model_args['type_codes'] = pd.Categorical(annotation.type).codes
    
    def log_elbo(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
    
    model = da.tandtiss_lda(train =trn.to_numpy(), **model_args)
    
    with model:
        trace = pm.fit(15000, method = 'advi', random_seed = 10, callbacks = [log_elbo])
    
    