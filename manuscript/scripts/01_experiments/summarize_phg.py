import damuta as da
import pandas as pd
import numpy as np
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', type=str, default='results/ckpt/phg')
parser.add_argument('--cosmic_def', type=str, default='data/COSMIC_v3.2_SBS_GRCh37.txt')
parser.add_argument('--outdir', type=str, default='results/figure_data')
args = parser.parse_args()


os.makedirs(args.outdir, exist_ok=True)


print('loading signatures...')
cosmic = pd.read_csv(args.cosmic_def, sep='\t', index_col = 0).T[da.mut96]
cosmic = da.SignatureSet(cosmic)


###############################
# DAMUTA h_lda parameter estimates
###############################
print('loading trained model...')
with open(args.trained_model, 'rb') as file: data = pickle.load(file)
h_lda = data['Model']
h_hat = h_lda.approx.sample(1)
n = len(h_lda.dataset.annotation.index)

A = np.moveaxis(h_hat.A[0],0,1)
W = (h_hat.theta[0,:,:,None]*A)

print('writing parameter estimates...')

W_df = pd.DataFrame(W.reshape(n, -1), index = h_lda.dataset.annotation.index)
W_df = W_df.rename(columns=lambda x: f'D{x//6 + 1}_M{x%6 + 1}')
W_df.to_csv(f'{args.outdir}/h_W.csv')

gamma = pd.DataFrame(W.sum(1), index = W_df.index)
gamma = gamma.rename(columns=lambda x: 'M' + str(x+1))
gamma.to_csv(f'{args.outdir}/h_gamma.csv')

theta = pd.DataFrame(W.sum(2), index = W_df.index)
theta = theta.rename(columns=lambda x: 'D' + str(x+1))
theta.to_csv(f'{args.outdir}/h_theta.csv')

phi = pd.DataFrame(h_hat.phi[0], index=[f'D{i}' for i in range(1,h_lda.n_damage_sigs+1)], columns=da.mut32)
phi.to_csv(f'{args.outdir}/h_phi.csv', index_label='name')

eta = pd.DataFrame(h_hat.eta[0].reshape(-1,6), index=[f'M{i}' for i in range(1,h_lda.n_misrepair_sigs+1)], columns=da.mut6)
eta.to_csv(f'{args.outdir}/h_eta.csv', index_label='name')
