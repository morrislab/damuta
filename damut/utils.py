import numpy as np
import pymc3 as pm
import pandas as pd
import wandb
import yaml

def load_config(config_fp):

    # Load the yaml file 
    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded configuration file {config_fp}")

    return config

def load_sweep_config(run, defaults_fp):
    # override config with any arguments set up in a wandb run
    # (useful for managing wandb sweeps)
    # Adamo Young 2021

    # load all defaults
    sim_args, dataset_args, model_args, pymc3_args = load_config(defaults_fp)
    run.config.setdefaults({**sim_args, **dataset_args, **model_args, **pymc3_args})

    # check for sweep-overwritten params
	run_config_d = dict(run.config)
	for d in [sim_args, dataset_args, model_args, pymc3_args]:
		for k,v in d.items():
			d[k] = run_config_d[k]

    return {sim_args, dataset_args, model_args, pymc3_args}

def save_checkpoint(fn, model, trace):
    with open(f'{fn}.pickle', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace, 'config': config}, buff)
        logging.debug(f"Model checkpoint pickled to: {fn}")
        
def load_checkpoint(fn):
    with open(fn, 'rb') as buff:
        data = pickle.load(buff)
        logging.debug(f"Model checkpoint loaded from: {fn}")
    return data



    