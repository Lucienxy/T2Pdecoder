import pandas as pd
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import EmbeddingModel

torch.manual_seed(0)


random.seed(90)
np.random.seed(90)
# params = [90, 951, 184, 162, 68, 12, 0.0005, 38]
parser = argparse.ArgumentParser(description="Apply the CLIP model to get the embedding of RNA or/and protein ")
parser.add_argument('--dim', type=int, default= 12, help='embedding dimension')
parser.add_argument('--node_num', type=list, default=[951, 184, 162], help='the number of nodes in each layer')
parser.add_argument('--pro_dir', type=str, default='', help='the protein abundance data file path')
parser.add_argument('--rna_dir', type=str, default='', help='the RNA expression data file path')
parser.add_argument('--model_pth', type=str, default='./save_model/embedding/glioma/', help='the trained model path')
parser.add_argument('--out_dir', type=str, default='./results/embedding/fine_tune/', help='the output directory to save the model')
parser.add_argument('--save_name', type=str, default='cohort', help='the suffix name of the saved model')
args = parser.parse_args()


# Load the data
if args.pro_dir == '' and args.rna_dir == '':
    print('Please provide at least one data path to apply the model')
    exit(0)
if not os.path.exists(args.model_pth):
    print('The trained model path does not exist')
    exit(0)

cohort = args.save_name
rna_dir = args.rna_dir
pro_dir = args.pro_dir

node_num = args.node_num
embedding_dim = args.dim

save_emb_pth = args.out_dir
if not os.path.exists(save_emb_pth):
    os.makedirs(save_emb_pth)

if args.rna_dir != '':
    rna_df = pd.read_csv(rna_dir)
    pid_df = rna_df[['PID']]
    rna_df.drop(['PID'],axis=1,inplace=True)
    rna_data = torch.tensor(rna_df.values, dtype=torch.float32)
    rna_encoder = EmbeddingModel(18860,node_num,embedding_dim)
    rna_pretrain_pth = args.model_pth+f'/rna_encoder_best.pth'
    rna_encoder.load_state_dict(torch.load(rna_pretrain_pth))
    rna_encoder.eval()
    rna_emb = rna_encoder(rna_data)
    rna_emb = rna_emb.detach().numpy()
    rna_emb = StandardScaler().fit_transform(rna_emb.T).T
    rna_emb_df = pd.DataFrame(rna_emb)
    rna_emb_df.insert(0,'PID',pid_df['PID'])
    rna_emb_df.to_csv(save_emb_pth+'/rna_emb_df_'+cohort+'.csv',index=None)
    print('Has done the RNA embedding!')
if args.pro_dir != '':
    pro_df = pd.read_csv(pro_dir)
    pid_df = pro_df[['PID']]
    pro_df.drop(['PID'],axis=1,inplace=True)
    pro_data = torch.tensor(pro_df.values, dtype=torch.float32)
    pro_encoder = EmbeddingModel(5738,node_num,embedding_dim)
    pro_pretrain_pth = args.model_pth+f'/pro_encoder_best.pth'
    pro_encoder.load_state_dict(torch.load(pro_pretrain_pth))
    pro_encoder.eval()
    pro_emb = pro_encoder(pro_data)
    pro_emb = pro_emb.detach().numpy()
    pro_emb = StandardScaler().fit_transform(pro_emb.T).T
    pro_emb_df = pd.DataFrame(pro_emb)
    pro_emb_df.insert(0,'PID',pid_df['PID'])
    pro_emb_df.to_csv(save_emb_pth+'/pro_emb_df_'+cohort+'.csv',index=None)
    print('Has done the protein embedding!')

