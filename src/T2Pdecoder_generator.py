# An auto-enconder by pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
import argparse
import torch.nn.functional as F
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from model import VAE


torch.manual_seed(0)
n_seed = 30

random.seed(n_seed)
np.random.seed(n_seed)


def gen_data_from_rdz(mat_pth, num_samples=100):
    with torch.no_grad():
        z = torch.randn(num_samples, hidden_size)
        generated_data = vae.decode(z)
    generated_df = pd.DataFrame(generated_data.numpy())
    z_df = pd.DataFrame(z.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)


def gene_data_TTS(mat_pth, train_pid, test_pid, hidden_size=4):
    z = pd.read_csv(mat_pth+'/BN_VAE_best_mu.csv')
    z_train = z[z['PID'].isin(train_pid)]
    z_test = z[z['PID'].isin(test_pid)]
    num_samples = z_train.shape[0]
    z_train.drop(['PID'],axis=1,inplace=True)
    z_train = torch.tensor(z_train.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z_train)
    generated_df = pd.DataFrame(generated_data.numpy())
    z_df = pd.DataFrame(z_train.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)
    num_samples = z_test.shape[0]
    z_test.drop(['PID'],axis=1,inplace=True)
    z_test = torch.tensor(z_test.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z_test)
    generated_df = pd.DataFrame(generated_data.numpy())
    z_df = pd.DataFrame(z_test.numpy())
    generated_df.to_csv(mat_pth+'/generated_data_'+str(num_samples)+'.csv',index=False)


def gen_data_new_cohort(mat_pth,hidden_size,z_path):
    z = pd.read_csv(z_path)
    num_samples = z.shape[0]
    z.drop(['PID'],axis=1,inplace=True)
    z = torch.tensor(z.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z)
    generated_df = pd.DataFrame(generated_data.numpy()) 
    z_df = pd.DataFrame(z.numpy())
    generated_df.to_csv(mat_pth+'/'+'generated_data_'+str(num_samples)+'.csv',index=False)


#params = [470, 462, 244, 184, 0.0005, 83] 
parser = argparse.ArgumentParser(description="BN-VAE application ")
parser.add_argument('--node_num', type=list, default=[470, 462, 244], help='the number of nodes in each layer')
parser.add_argument('--model_pth', type=str, default='./save_model/BN_VAE_best.pth', help='the pretrain model path')
parser.add_argument('--dim', type=int, default=12, help='embedding dimension')
parser.add_argument('--p_drop', type=float, default=0.1, help='the dropout rate')
parser.add_argument('--emb_dir', type=str, default='./data/rna_emb_df_cohort.csv', help='the embedding data file path')
parser.add_argument('--out_dir', type=str, default='./results/VAE', help='the output directory to save the model')
args = parser.parse_args()


input_size = 5738
hidden_size = args.dim
node_num = args.node_num
p_drop = args.p_drop
mat_pth = args.out_dir
if not os.path.exists(mat_pth):
    os.makedirs(mat_pth)

model_path = args.model_pth
vae = VAE(input_size=input_size, hidden_size=hidden_size,node_num=node_num,p_drop=p_drop)

vae.load_state_dict(torch.load(model_path))
vae.eval()
z_path = args.emb_dir

gen_data_new_cohort(mat_pth,hidden_size,z_path)
