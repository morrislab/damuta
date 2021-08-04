from scipy.special import softmax, logsumexp, loggamma

def alp_B(data, B):
    return (data * np.log(B)).sum() / data.sum()

def mult_ll(x, p):
    # x and p should both be same dimensions; Sx96
    return loggamma(x.sum(1) + 1) - loggamma(x+1).sum(1) + (x * np.log(p)).sum(1)

def lap_B(data, Bs):
    # Bs should be shape DxSx96 where D is the number of posterior samples
    # use logsumexp for stability
    assert Bs.ndim == 3, 'expected multiple trials for B'
    return logsumexp(np.vstack([mult_ll(data, B) for B in Bs]).sum(1)) - np.log(Bs.shape[0])

def alr(x, e=1e-12):
    '''
    additive log ratio
    x is a NxD matrix
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    '''
    # add small value for stability in log
    x = x + e
    return (np.log(x) - np.log(x[...,-1]).reshape(-1,1))[:,0:-1]

def alr_inv(y):
    '''
    inverse alr transform
    y is a Nx(D-1) matrix
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr_inv(alr(x))
    array([ 0.1,  0.3,  0.4,  0.2])
    '''
    if y.ndim == 1: y = y.reshape(1,-1)
    return softmax(np.hstack([y, np.zeros((y.shape[0], 1)) ]), axis = 1)
