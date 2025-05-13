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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import VAE


torch.manual_seed(0)
n_seed = 30

random.seed(n_seed)
np.random.seed(n_seed)

def cal_scc(recon_x,y,sel_marker):
    scc_sam_list = []
    scc_fea_list = []
    for i in range(recon_x.shape[0]):
        scc_sam_list.append(stats.spearmanr(recon_x[i,sel_marker].detach().numpy(),y[i,:])[0])
    for j in range(len(sel_marker)):
        scc_fea_list.append(stats.spearmanr(recon_x[:,sel_marker[j]].detach().numpy(),y[:,j])[0])
    return np.median(scc_sam_list),np.median(scc_fea_list)
def cal_pcc(recon_x,y,sel_marker):
    scc_sam_list = []
    scc_fea_list = []
    for i in range(recon_x.shape[0]):
        scc_sam_list.append(stats.pearsonr(recon_x[i,sel_marker].detach().numpy(),y[i,:])[0])
    for j in range(len(sel_marker)):
        scc_fea_list.append(stats.pearsonr(recon_x[:,sel_marker[j]].detach().numpy(),y[:,j])[0])
    return np.median(scc_sam_list),np.median(scc_fea_list)

# only use the selected marker to calculate the loss
def loss_function(recon_x, y, sel_marker):
    MSE = F.mse_loss(recon_x[:,sel_marker], y, reduction='mean')
    return MSE
#params = [470, 462, 244, 184, 0.0005, 83] # 70
parser = argparse.ArgumentParser(description="BN-VAE fine-tuning in CITE-seq ")
parser.add_argument('--rd', type=int, default=0, help='random seed')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch', type=int, default=1024, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--node_num', type=list, default=[470, 462, 244], help='the number of nodes in each layer')
parser.add_argument('--test_ratio', type=float, default=0.2, help='the ratio of test set')
parser.add_argument('--p_drop', type=float, default=0.1, help='the dropout rate')
parser.add_argument('--dim', type=int, default=12, help='embedding dimension')
parser.add_argument('--pro_dir', type=str, default='./data/cite_seq_gbm_pro.csv', help='the protein abundance data file path')
parser.add_argument('--emb_dir', type=str, default='./data/rna_emb_df_cite.csv', help='the embedding data file path')
parser.add_argument('--model_pth', type=str, default='./saved_model/T2Pdecoder/glioma/BN_VAE_best.pth', help='the pretrain model path')
parser.add_argument('--pro_idx_dir', type=str, default='./data/cite_ft_pro_idx.csv', help='the selected marker file path')
parser.add_argument('--out_dir', type=str, default='./results/VAE/cite_ft/', help='the output directory to save the model')
args = parser.parse_args()

input_size = 5738
hidden_size = args.dim
node_num = args.node_num
p_drop = args.p_drop
mat_pth = args.out_dir
if not os.path.exists(mat_pth):
    os.makedirs(mat_pth)
# ft_pth = mat_pth + '/cite_ft_2_do/'
# if not os.path.exists(ft_pth):
#     os.makedirs(ft_pth)
# model_path = mat_pth+'/vae_assign_'+str(hidden_size)+'_'+str(num_epochs)+'.pth'
model_path = args.model_pth
vae = VAE(input_size=input_size, hidden_size=hidden_size,node_num=node_num,p_drop=p_drop)

vae.load_state_dict(torch.load(model_path))

pro_data = pd.read_csv(args.pro_dir)
idx_df = pd.read_csv(args.pro_idx_dir)
idx_list = idx_df['idx'].tolist()
pid_df = pro_data[['PID']]
rna_emb = pd.read_csv(args.emb_dir)
rna_emb.drop(['PID'],axis=1,inplace=True)
rna_emb_values = rna_emb.values
pro_data.drop(['PID'],axis=1,inplace=True)
pro_list = pro_data.columns.tolist()
pro_data_values = pro_data.values

num_epochs = args.num_epochs
batch_size = args.batch
lr = args.lr
rd = args.rd

x_train, x_test, y_train, y_test = train_test_split(rna_emb_values, pro_data_values, test_size=args.test_ratio, random_state=rd)

train_indices = np.where(np.all(rna_emb_values == x_train[:, None], axis=2))[1]
train_pid_list = pid_df.iloc[train_indices,0].values.tolist()
test_indices = np.where(np.all(rna_emb_values == x_test[:, None], axis=2))[1]
test_pid_list = pid_df.iloc[test_indices,0].values.tolist()

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

data_tensor = torch.tensor(rna_emb_values, dtype=torch.float32)
pro_tensor = torch.tensor(pro_data_values, dtype=torch.float32)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)  

loss_pth_train = mat_pth+'/train_loss.csv'
loss_pth_test = mat_pth+'/test_loss.csv'
# save train and test pid as csv
train_pid_df = pd.DataFrame(train_pid_list)         
train_pid_df.to_csv(mat_pth+'/train_pid.csv',index=None)
test_pid_df = pd.DataFrame(test_pid_list)
test_pid_df.to_csv(mat_pth+'/test_pid.csv',index=None)
print('train_dim',x_train.shape)
print('test_dim',x_test.shape)


train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=1E-4)
# train
losses = []
r2_list = []
min_test_loss = 100
min_test_epoch = 0

for epoch in range(num_epochs):
    vae.train()
    for batch_x, batch_y in train_dataloader:
        optimizer.zero_grad()
        outputs = vae.decode(batch_x)
        loss = loss_function(outputs, batch_y, idx_list)
        loss.backward()
        optimizer.step()
    vae.eval()
    with torch.no_grad():
        outputs= vae.decode(x_train)
        loss = loss_function(outputs, y_train,idx_list)
        pcc_sam,pcc_fea = cal_pcc(outputs,y_train,idx_list)
        outputs_t = vae.decode(x_test)
        loss_t = loss_function(outputs_t, y_test,idx_list)
        pcc_sam_t,pcc_fea_t = cal_pcc(outputs_t,y_test,idx_list)
        losses.append([loss.item(),loss_t.item()])
        # print(f"Epoch: {epoch + 1}, train_loss: {loss.item():.4f}, test_loss: {loss_t.item():.4f}")
        print(f"Epoch: {epoch + 1}, train_loss: {loss.item():.4f}, test_loss: {loss_t.item():.4f},train_pcc_sam: {pcc_sam:.4f}, train_pcc_fea: {pcc_fea:.4f},test_pcc_sam: {pcc_sam_t:.4f}, test_pcc_fea: {pcc_fea_t:.4f}")
        if loss_t.item() < min_test_loss:
            min_test_loss = loss_t.item()
            min_test_epoch = epoch+1
            torch.save(vae.state_dict(), mat_pth+'/cite_ft_best.pth')
            outputs = vae.decode(x_test)
            sel_out = outputs[:,idx_list]
            sel_res_df = pd.DataFrame(sel_out.detach().numpy(),columns=pro_list)
            sel_res_df.insert(0,'PID',test_pid_list)
            sel_res_df.to_csv(mat_pth+'/sel_pro_generate_best_test.csv',index=False)
print('Best Epoch:',min_test_epoch)
print('Best Test Loss:',min_test_loss)

