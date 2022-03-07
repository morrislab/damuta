from .utils import *

def log_elbo(*args):
    approx, losses, i = args
    wandb.log({'ELBO': losses[-1]})
        
def log_loss(trn, val, call_every = 1000):
    def cb(*args):
        approx, losses, i = args
        if i % call_every != 0:
            return None
        
        B = approx.sample(100).B.mean(0)
    
        wandb.log({'trn_alp': alp_B(trn.to_numpy(), B),
                   'val_alp': alp_B(val.to_numpy(), B)
                  })
    return cb
        
def log_data_summary(trn, val, tst1, tst2):
    summary_table = wandb.Table(columns=['dataset', 'n samples', 'mean nmut', 'median nmut', 'min nmut', 'max nmut'],
                                data = [['train', trn.shape[0], trn.sum(1).mean(), np.median(trn.sum(1)), trn.sum(1).min(), trn.sum(1).max()],
                                        ['val', val.shape[0], val.sum(1).mean(), np.median(val.sum(1)), val.sum(1).min(), val.sum(1).max()],
                                        ['test1', tst1.shape[0], tst1.sum(1).mean(), np.median(tst1.sum(1)), tst1.sum(1).min(), tst1.sum(1).max()],
                                        ['test2', tst2.shape[0], tst2.sum(1).mean(), np.median(tst2.sum(1)), tst2.sum(1).min(), tst2.sum(1).max()] 
                                       ])
    wandb.log({'dataset summary': summary_table})


def ckpt(fp, model, trace, dataset_args, model_args, pymc3_args):
    def cb(*args):
        save_checkpoint(fp, model, trace, dataset_args, model_args, pymc3_args)
    return cb
    
    