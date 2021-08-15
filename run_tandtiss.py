import wandb
import damut as da
import numpy as np
import pandas as pd
import pymc3 as pm
import argparse

if __name__ == "__main__":

    # pick up config file path
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='yaml config file', default = "config/tandtiss-defaults.yaml")
    parser.add_argument('-p', help='wandb project name to assign run to', default = None)
    args = parser.parse_args()

    # load config
    dataset_args, model_args, pymc3_args = da.load_config(args.f)

    # init run
    run = wandb.init(project = 'tandtiss')
    run.config.update({**dataset_args, **model_args, **pymc3_args})
    
    #np.random.seed(model_args['model_seed']) 
    #pm.set_tt_rng(model_args['model_seed'])  
    #model_args.pop('model_seed') # TODO: remove this seeding hack with next pymc3 release

    # load data 
    cosmic = da.load_sigs("data/COSMIC_v3.2_SBS_GRCh37.txt")
    counts = da.load_dataset(**dataset_args)
    trn, val, tst1, tst2 = da.split_data(counts, rng=dataset_args['data_rng'])
    
    summary_table = wandb.Table(columns=['dataset', 'n samples', 'mean nmut/sample', 'median nmut/sample', 'min nmut/sample', 'max nmut/sample'],
                                data = [['train', trn.shape[0], trn.sum(1).mean(), np.median(trn.sum(1)), trn.sum(1).min(), trn.sum(1).max()],
                                        ['val', val.shape[0], val.sum(1).mean(), np.median(val.sum(1)), val.sum(1).min(), val.sum(1).max()],
                                        ['test1', tst1.shape[0], tst1.sum(1).mean(), np.median(tst1.sum(1)), tst1.sum(1).min(), tst1.sum(1).max()],
                                        ['test2', tst2.shape[0], tst2.sum(1).mean(), np.median(tst2.sum(1)), tst2.sum(1).min(), tst2.sum(1).max()] 
                                       ])
    wandb.log({'dataset summary': summary_table})
    
    # turn tissue type annotations categorical
    annotation = pd.read_csv(dataset_args['annotation_fp'], index_col = 0, header = 0).loc[trn.index]
    model_args['type_codes'] = pd.Categorical(annotation.type).codes
    
    # fit model for train samples
    model, trace = da.infer(trn.to_numpy(), model_args, pymc3_args, 
                            da.cbs(train=trn.to_numpy(), val=val.to_numpy(), cosmic=cosmic.to_numpy(), log_every= max(1, pymc3_args['n']/5))
                           )
    
    # evaluate 
    hat = trace.sample(100)
    wandb.log({'inferred tau mapping': wandb.Table(dataframe = da.profile_sigs(da.get_tau(hat.phi.mean(0), hat.eta.mean(0)), cosmic.to_numpy(), refidx = cosmic.index)),
               'inferred phi mapping': wandb.Table(dataframe = da.profile_sigs(hat.phi.mean(0), da.get_phi(cosmic.to_numpy()), refidx = cosmic.index)),
               'inferred eta mapping': wandb.Table(dataframe = da.profile_sigs(hat.eta.mean(0).reshape(-1,6), 
                                                                               da.get_eta(cosmic.to_numpy()).reshape(-1,6), 
                                                                               refidx = cosmic.index, thresh = 0.95)),
              })

    # fit model for test samples

    # evaluate tst2