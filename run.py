import wandb
import damut as da
from damut.utils import load_config, split_data
import argparse

if __name__ == "__main__":

    # pick up config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='yaml config file', default = "config/config-defaults.yaml")
    parser.add_argument('-p', help='wandb project name to assign run to', default = None)
    args = parser.parse_args()

    # load config
    dataset_args, model_args, pymc3_args = load_config(args.f)

    # init run
    run = wandb.init()
    run.config.update({**dataset_args, **model_args, **pymc3_args})

    # load data 
    counts = da.load_dataset(**dataset_args)
    trn, val, tst1, tst2 = split_data(counts)
    
    # fit model for train samples
    model, trace = da.infer(trn.to_numpy(), val.to_numpy(), model_args, pymc3_args)

    # evaluate val

    # fit model for test samples

    # evaluate tst2