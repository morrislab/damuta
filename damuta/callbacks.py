import wandb


__all__ = ["Callback", "StopOnConvergence", "LogELBO"]


class Callback:
    # Copyright 2020 The PyMC Developers
    # https://github.com/pymc-devs/pymc/blob/e987950b359edfe4a49c92ee67c57983fe0fc5d9/pymc/variational/callbacks.py
    
    def __call__(self, approx, loss, i):
        raise NotImplementedError

class StopOnConvergence(Callback):
    """Stop iteration once loss no longer improves.

    Parameters
    ----------
    n_mutations: int
        Number of mutations in the dataset
    every: int
        Frequency at which convergence is checked, and the window size for
        computing the mean loss
    tol: float
        Per mutation loss tolerance (nats). Loss improvement below this
        value for consecutive windows indicates convergence and stops
        iteration. If this is reached, the converged_step attribute is set.

    Examples
    --------
    >>> stop = StopOnConvergence(n_mutations=model.dataset.counts.to_numpy().sum())
    >>> model.fit(n=50000, callbacks=[stop])
    >>> stop.converged_step
    """

    def __init__(self, n_mutations, every=1000, tol = 1e-5):
        self.n_mutations = n_mutations
        self.every = every
        self.tol = tol
        self.converged_step = None

    def __call__(self, approx, loss, i):
        if len(loss) < self.every * 3:
            return

        if len(loss) % self.every == 0:
            loss3 = loss[-(3 * self.every):-(2 * self.every)].mean()
            loss2 = loss[-(2 * self.every):-(self.every)].mean()
            loss1 = loss[-self.every:].mean()

            loss_ok1 = (loss2 - loss1) / self.n_mutations < self.tol
            loss_ok2 = (loss3 - loss2) / self.n_mutations < self.tol

            if loss_ok1 and loss_ok2:
                self.converged_step = len(loss)
                raise StopIteration(f'Converged at step {len(loss)}.')


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
