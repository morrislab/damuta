import damuta as da
import pandas as pd
import numpy as np
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', type=str, default='results/ckpt/subclones')
parser.add_argument('--outdir', type=str, default='results/figure_data')
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)


print('loading subclone model...')
with open(args.trained_model, 'rb') as file: data = pickle.load(file)

h_lda = data['Model']
h_hat = h_lda.approx.sample(1)
n = len(h_lda.dataset.annotation.index)

A = np.moveaxis(h_hat.A[0],0,1)
W = (h_hat.theta[0,:,:,None]*A)

W_df = pd.DataFrame(W.reshape(n, -1), index = h_lda.dataset.annotation.index)
W_df = W_df.rename(columns=lambda x: f'D{x//6 + 1}_M{x%6 + 1}')
W_df.to_csv(f'{args.outdir}/subclone_h_W.csv')

gamma = pd.DataFrame(W.sum(1), index = W_df.index)
gamma = gamma.rename(columns=lambda x: 'M' + str(x))
gamma.to_csv(f'{args.outdir}/subclone_h_gamma.csv')

theta = pd.DataFrame(W.sum(2), index = W_df.index)
theta = theta.rename(columns=lambda x: 'D' + str(x))
theta.to_csv(f'{args.outdir}/subclone_h_theta.csv')
