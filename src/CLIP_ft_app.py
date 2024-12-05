import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import EmbeddingModel

torch.manual_seed(0)

params = [90, 951, 184, 162, 68, 12, 0.0005, 38]
random.seed(params[0])
np.random.seed(params[0])


# cohort = '_NG2021_sc_all'
# rna_dir = './data/NG2021_scale_scRNA_18860.csv'


cohort = '_cc2024'
rna_dir = './data/cc2024_rna_18860.csv'
pro_dir = './data/cc2024_pro_5738.csv'

# cohort = '_merge'
# pro_dir = './data/merged_pro_gli_df.csv'

# cohort = '_PanGlioma'
# rna_dir = './data/panGlioma_rna_18860.csv'



node_num = [params[1],params[2],params[3]]
embedding_dim = params[5]
learning_rate = params[6]
batch_size = params[4]
num_epochs = 56
save_pth = './save_model/embedding/glioma/'
mat_pth = save_pth+'saved_model/'
save_emb_pth = save_pth




pro_df = pd.read_csv(pro_dir)
pid_df = pro_df[['PID']]
pro_df.drop(['PID'],axis=1,inplace=True)
pro_data = torch.tensor(pro_df.values, dtype=torch.float32)
sample_num = pro_data.shape[0]

pro_encoder = EmbeddingModel(pro_data.shape[1],node_num,embedding_dim)
pro_pretrain_pth = save_pth+f'/pro_encoder_best.pth'
pro_encoder.load_state_dict(torch.load(pro_pretrain_pth))
pro_emb = pro_encoder(pro_data)
pro_emb = pro_emb.detach().numpy()
pro_emb = StandardScaler().fit_transform(pro_emb.T).T
pro_emb_df = pd.DataFrame(pro_emb)
pro_emb_df.insert(0,'PID',pid_df['PID'])
pro_emb_df.to_csv(save_emb_pth+'/pro_emb_df'+cohort+'.csv',index=None)



rna_df = pd.read_csv(rna_dir)
pid_df = rna_df[['PID']]
rna_df.drop(['PID'],axis=1,inplace=True)
rna_data = torch.tensor(rna_df.values, dtype=torch.float32)

sample_num = rna_data.shape[0]

rna_encoder = EmbeddingModel(rna_data.shape[1],node_num,embedding_dim)
rna_pretrain_pth = save_pth+f'/rna_encoder_best.pth'
rna_encoder.load_state_dict(torch.load(rna_pretrain_pth))
rna_emb = rna_encoder(rna_data)
rna_emb = rna_emb.detach().numpy()
rna_emb = StandardScaler().fit_transform(rna_emb.T).T
rna_emb_df = pd.DataFrame(rna_emb)
rna_emb_df.insert(0,'PID',pid_df['PID'])
rna_emb_df.to_csv(save_emb_pth+'/rna_emb_df'+cohort+'.csv',index=None)
