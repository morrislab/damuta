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

