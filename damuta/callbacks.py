import wandb


__all__ = ["Callback", "LogELBO", "LogWandB"]


class Callback:
    # Copyright 2020 The PyMC Developers
    # https://github.com/pymc-devs/pymc/blob/e987950b359edfe4a49c92ee67c57983fe0fc5d9/pymc/variational/callbacks.py
    
    def __call__(self, approx, loss, i):
        raise NotImplementedError

class LogELBO(Callback):
    """Log ELBO using `wandb.log()`. `wandb.init()` must be run first.
     
    Parameters
    ----------
    every: int
        Frequency at which wandb.log() is called
        
    Examples
    --------
    >>> with model:
    ...     approx = pm.fit(n=1000, callbacks=[LogELBO(every=50)])
    """

    def __init__(self, every=100):
        self.every = every

    def __call__(self, approx, loss, i):
        if i % self.every or i < self.every:
            return
        
        wandb.log({"ELBO": loss[i-1]})
