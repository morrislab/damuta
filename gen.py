# gen.py 
# generative process for substitution mutations in cancer
import argparse
import numpy as np
from numpy.random import default_rng
rng = default_rng(1080)


# https://stackoverflow.com/a/37755413
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)

def mask_renorm(B):
    # remove mutations impossible in the context and normalize
    S, C, M = B.shape
    c_mask = np.tile([False, False, False, True, True, True], (S, C//2 ,1))
    t_mask = np.tile([True, True, True, False, False, False], (S, C//2 ,1))
    m = np.concatenate([c_mask, t_mask], axis = 1)
    B[m] = 0
    return B/B.sum(2)[:, :, np.newaxis]

def generate_samples(args):
    theta = rng.dirichlet(args.psi, args.S)
    A = rng.dirichlet(args.gamma, (args.S, args.J))
    phi = rng.dirichlet(args.alpha, args.J)
    eta = rng.dirichlet(args.beta, args.K)

    # for each sample get count of damage contexts drawn from each damage context signature
    Z = np.array([rng.multinomial(*x) for x in zip(args.N, np.dot(theta, phi), np.full((args.S,), 1))]).squeeze()
    # get transition probabilities
    B = mask_renorm(np.dot(phi.T, np.dot(A, eta)).swapaxes(0,1))
    # for each damage context, get count of misrepair
    V = np.array([rng.multinomial(*x) for x in zip(Z.reshape(args.S*args.C, 1), B.reshape(args.S*args.C, args.M))]).reshape(100,32,6)

    return V

def validate_args(parser):
    # validate passed arguments
    args = parser.parse_args()

    for va, vl, ka, kl in zip([args.N, args.psi, args.gamma, args.alpha, args.beta], \
                              [args.S, args.J,   args.K,     args.C,     args.M],    \
                              ["N", "psi", "gamma", "alpha", "beta"], ["S", "J", "K", "C", "M"]):

        assert len(va) == 1 or len(va) == vl, \
        f"Parameter {ka} misized; len({ka}) should be equal to either 1 or {kl}. recieved len({ka})={len(va)}, {kl}={vl}"

        if len(va) == 1:
            vars(args)[f'{ka}'] = np.full((vl,), va)

    return args


def main():

    parser = argparse.ArgumentParser(description='damut generative model')
    parser.add_argument("-S", type=int, default = 100, help = "number of samples")
    parser.add_argument('-N', type=int, default = (1000,), nargs='+', help = "number of mutations per sample")
    parser.add_argument('-C', type=int, default = 32, help = "number of damage context types")
    parser.add_argument('-M', type=int, default = 6, help = "number of misrepair types")
    parser.add_argument('-J', type=int, default = 10, help = "number of damage context signatures")
    parser.add_argument('-K', type=int, default = 10, help = "number of misrepair signatures")
    parser.add_argument('-psi', type=float, default = [1], nargs='+', action=Store_as_array,
                        help='psi parameter for selecting sample-wise damage context signature activities')
    parser.add_argument('-gamma', type=float, default = [0.1], nargs='+', action=Store_as_array,
                        help='gamma parameter for selecting sample-wise misrepair signature activities')
    parser.add_argument('-alpha', type=float, default = [0.1], nargs='+', action=Store_as_array,
                        help='alpha parameter for damage context signature')
    parser.add_argument('-beta', type=float, default = [0.1], nargs='+', action=Store_as_array,
                        help='beta parameter for misrepair signature')
    
    args = validate_args(parser)
    print(generate_samples(args)[0])


if __name__ == '__main__':
    main()