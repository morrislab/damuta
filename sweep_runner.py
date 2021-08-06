import wandb
import damut as da
from damut.utils import load_config

if __name__ == "__main__":

    dataset_args, model_args, pymc3_args = load_config("config/config-defaults.yaml")
    

    # init run
    run = wandb.init()
    wandb.log({'destargs': dataset_args})
    # configure run with defaults and sweep args
    run.config.setdefaults({**dataset_args, **model_args, **pymc3_args})
    
    # override config with any arguments set up in wandb run
    # (useful for managing wandb sweeps) Adamo Young 2021
    run_config_d = dict(run.config)
    for d in [dataset_args, model_args, pymc3_args]:
        for k,v in d.items():
            d[k] = run_config_d[k]

    # fit model
    da.infer(10, model_args, pymc3_args)