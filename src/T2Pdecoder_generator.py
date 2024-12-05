# An auto-enconder by pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
import os
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


def gen_data_new_cohort(mat_pth,hidden_size,z_path,fs='RNA'):
    z = pd.read_csv(z_path)
    num_samples = z.shape[0]
    z.drop(['PID'],axis=1,inplace=True)
    z = torch.tensor(z.values, dtype=torch.float32)
    with torch.no_grad():
        generated_data = vae.decode(z)
    generated_df = pd.DataFrame(generated_data.numpy()) 
    z_df = pd.DataFrame(z.numpy())
    generated_df.to_csv(mat_pth+'/'+fs+'_generated_data_'+str(num_samples)+'.csv',index=False)


save_path = './save_model/T2Pdecoder/'
input_size = 5738
params = [470, 462, 244, 184, 0.0005, 83]
rd = params[5]
hidden_size = 12
num_epochs = 182
batch_size = params[3]
p_drop = 0.1
node_num = params[0:3]
save_interval=10000
KLD_weight = 1


new_dir = 'Layer_'+str(node_num[0])+'_'+str(node_num[1])+'_'+str(node_num[2])+'_Drop_'+str(p_drop)+'_BS_'+str(batch_size) +'_RD_'+str(rd)
if not os.path.exists(save_path+new_dir):
    os.makedirs(save_path+new_dir)
mat_pth = save_path+new_dir
model_path = mat_pth+'/BN_VAE_best.pth'
vae = VAE(input_size=input_size, hidden_size=hidden_size,node_num=node_num,p_drop=p_drop)

vae.load_state_dict(torch.load(model_path))

gen_data_from_rdz(mat_pth, num_samples=100)
gen_data_from_rdz(mat_pth, num_samples=1000)
train_df = pd.read_csv(mat_pth+'/train_pid.csv')
test_df = pd.read_csv(mat_pth+'/test_pid.csv')
train_pid = train_df.iloc[:,0].values.tolist()
test_pid = test_df.iloc[:,0].values.tolist()
gene_data_TTS(mat_pth, train_pid, test_pid, hidden_size=hidden_size)



## predict Protein based RNA
z_path = './save_model/embedding/glioma/rna_emb_df_cc2024.csv'
gen_data_new_cohort(mat_pth,hidden_size,z_path,'RNA')

# ## predicte Protein based Single Cell RNA
z_path = './save_model/embedding/glioma/rna_emb_df_NG2021_sc.csv'
gen_data_new_cohort(mat_pth,hidden_size,z_path,'RNA')


# predict Protein based PanGlioma RNA
z_path = './save_model/embedding/glioma/rna_emb_df_PanGlioma.csv'
gen_data_new_cohort(mat_pth,hidden_size,z_path,'RNA')


# ## predict Protein based Protein
z_path = './save_model/embedding/glioma/pro_emb_df_CPTAC.csv'
gen_data_new_cohort(mat_pth,hidden_size,z_path,'Pro')
z_path = './save_model/embedding/glioma/pro_emb_df_CGGA.csv'
gen_data_new_cohort(mat_pth,hidden_size,z_path,'Pro')