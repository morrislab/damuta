import wandb
import damut as da
from damut.utils import load_config

if __name__ == "__main__":

    dataset_args, model_args, pymc3_args = load_config("config/config-defaults.yaml")
    
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

    # load data 
    counts = da.load_dataset(**dataset_args)
    trn, val, tst1, tst2 = da.split_data(counts)
    
    summary_table = wandb.Table(columns=['dataset', 'n samples', 'mean nmut', 'median nmut', 'min nmut', 'max nmut'],
                                data = [['train', trn.shape[0], trn.sum(1).mean(), np.median(trn.sum(1)), trn.sum(1).min(), trn.sum(1).max()],
                                        ['val', val.shape[0], val.sum(1).mean(), np.median(val.sum(1)), val.sum(1).min(), val.sum(1).max()],
                                        ['test1', tst1.shape[0], tst1.sum(1).mean(), np.median(tst1.sum(1)), tst1.sum(1).min(), tst1.sum(1).max()],
                                        ['test2', tst2.shape[0], tst2.sum(1).mean(), np.median(tst2.sum(1)), tst2.sum(1).min(), tst2.sum(1).max()] 
                                       ])
    wandb.log({'dataset summary': summary_table})
    
    # fit model for train samples
    model, trace = da.infer(trn.to_numpy(), model_args, pymc3_args)