import pandas as pd
import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from model import EmbeddingModel


def create_neg_indice_pair(num_samples):
    tmp_idx1 = torch.randperm(num_samples)
    tmp_idx2 = torch.randperm(num_samples)
    for i in range(num_samples-1):
        if tmp_idx1[i] == tmp_idx2[i]:
            tmp_r = tmp_idx2[i].clone()
            tmp_idx2[i] = tmp_idx2[i+1]
            tmp_idx2[i+1] = tmp_r
        
    if tmp_idx1[-1] == tmp_idx2[-1]:
        tmp_r = tmp_idx2[-1].clone()
        tmp_idx2[-1] = tmp_idx2[0]
        tmp_idx2[0] = tmp_r
    return tmp_idx1, tmp_idx2

def create_neg_indice_pair_all(num_samples):
    tmp_idx1 = []
    tmp_idx2 = []
    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                continue
            else:
                tmp_idx1.append(i)
                tmp_idx2.append(j)
    return torch.tensor(tmp_idx1), torch.tensor(tmp_idx2)

def create_neg_samples(embeddings, indices):
    negative_samples = embeddings[indices]
    return negative_samples

def compute_loss(batch_size, rna_embeddings, pro_embeddings):
    labels_pos = torch.ones(batch_size)
    # labels_neg = torch.zeros(batch_size)
    loss_pos = criterion(rna_embeddings, pro_embeddings, labels_pos)
    # neg_idx_rna, neg_idx_pro = create_neg_indice_pair(batch_size)
    neg_idx_rna, neg_idx_pro = create_neg_indice_pair_all(batch_size)
    neg_rna_sample = create_neg_samples(rna_embeddings, neg_idx_rna)
    neg_pro_sample = create_neg_samples(pro_embeddings, neg_idx_pro)
    labels_neg = torch.ones(neg_idx_rna.shape[0])*(-1)
    loss_neg = criterion(neg_rna_sample, neg_pro_sample, labels_neg)
    loss = (loss_pos + loss_neg)
    # print('Pos Loss: {:.8f}, Neg Loss: {:.8f}'.format(loss_pos.item(), loss_neg.item()))
    return loss_pos,loss_neg

''' 
hyper-parameters
0: global random seed
1-3: hidden layer nodel number
4: batch size
5: embedding dimension
6: learning rate
7: random seed for train-test split
'''


random.seed(90)
np.random.seed(90)
#params = [90, 951, 184, 162, 68, 12, 0.0005, 38]
parser = argparse.ArgumentParser(description="Pre-train CLIP model on Pan-Cancer data")
parser.add_argument('--rd', type=int, default=38, help='random seed')
parser.add_argument('--ep', type=int, default=100, help='the epochs to train the model')
parser.add_argument('--dim', type=int, default=12, help='embedding dimension')
parser.add_argument('--batch', type=int, default=68, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--node_num', type=list, default=[951, 184, 162], help='the number of nodes in each layer')
parser.add_argument('--pro_dir', type=str, default='./data/test_gbm_pro_df.csv', help='the protein abundance data file path')
parser.add_argument('--rna_dir', type=str, default='./data/test_gbm_rna_df.csv', help='the RNA expression data file path')
parser.add_argument('--out_dir', type=str, default='./results/embedding/', help='the output directory to save the model')
args = parser.parse_args()
'''
rna_dir: the pan-cancer RNA expression data file path
pro_dir: the pan-cancer protein abundance data file path
RNA data and protein data are ordered by the same patient ID list
'''
rna_dir = args.rna_dir
pro_dir = args.pro_dir


rna_df = pd.read_csv(rna_dir)
pro_df = pd.read_csv(pro_dir)
if pro_df[['PID']].equals(rna_df[['PID']]) == False:
    print('The PID list of RNA and protein data are not the same!')
    exit(0)
pid_df = rna_df[['PID']]
rna_df.drop(['PID'],axis=1,inplace=True)
pro_df.drop(['PID'],axis=1,inplace=True)
rna_data = torch.tensor(rna_df.values, dtype=torch.float32)
pro_data = torch.tensor(pro_df.values, dtype=torch.float32)
print('RNA data shape:', rna_data.shape)
print('Protein data shape:', pro_data.shape)

sample_num = rna_data.shape[0]

rd = args.rd
node_num = args.node_num
embedding_dim = args.dim
learning_rate = args.lr
batch_size = args.batch
num_epochs = args.ep
save_pth = args.out_dir

if not os.path.exists(save_pth):
    os.makedirs(save_pth)

torch.manual_seed(0)
rna_encoder = EmbeddingModel(rna_data.shape[1],node_num,embedding_dim)
pro_encoder = EmbeddingModel(pro_data.shape[1],node_num,embedding_dim)
criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(list(rna_encoder.parameters()) + list(pro_encoder.parameters()), lr=learning_rate, weight_decay=1E-5)


x_train, x_test, y_train, y_test = train_test_split(rna_df.values, pro_df.values, test_size=0.1, random_state=rd)


train_indices = np.where(np.all(rna_df.values == x_train[:, None], axis=2))[1]
train_pid_list = pid_df.iloc[train_indices,0].values.tolist()
test_indices = np.where(np.all(rna_df.values == x_test[:, None], axis=2))[1]
test_pid_list = pid_df.iloc[test_indices,0].values.tolist()
train_pid_df = pd.DataFrame(train_pid_list)
train_pid_df.to_csv(save_pth+'/train_pid.csv',index=None)
test_pid_df = pd.DataFrame(test_pid_list)
test_pid_df.to_csv(save_pth+'/test_pid.csv',index=None)

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

loss_list = []
min_test_loss = 100
min_test_epoch = 0
for epoch in range(num_epochs):
    rna_encoder.train()
    pro_encoder.train()
    for bat_rna, bat_pro in train_dataloader:
        optimizer.zero_grad()
        rna_embeddings = rna_encoder(bat_rna)
        pro_embeddings = pro_encoder(bat_pro)
        loss_pos,loss_neg = compute_loss(bat_rna.shape[0], rna_embeddings, pro_embeddings)
        loss = (loss_pos + loss_neg)
        loss.backward()
        optimizer.step()
    rna_encoder.eval()
    pro_encoder.eval()
    with torch.no_grad():
        rna_embeddings = rna_encoder(x_train)
        pro_embeddings = pro_encoder(y_train)
        loss_pos,loss_neg = compute_loss(x_train.shape[0], rna_embeddings, pro_embeddings)
        loss = (loss_pos + loss_neg)
        rna_embeddings_t = rna_encoder(x_test)
        pro_embeddings_t = pro_encoder(y_test)
        loss_pos_t,loss_neg_t = compute_loss(x_test.shape[0], rna_embeddings_t, pro_embeddings_t)
        loss_t = (loss_pos_t + loss_neg_t)
        loss_list.append([epoch+1,loss_pos.item(),loss_neg.item(),loss.item(),loss_pos_t.item(),loss_neg_t.item(),loss_t.item()])
        print(f'Epoch:{epoch+1},train:pos:{loss_pos.item():.4f},neg:{loss_neg.item():.4f},total:{loss.item():.4f},test:pos:{loss_pos_t.item():.4f},neg:{loss_neg_t.item():.4f},total:{loss_t.item():.4f}')
        if loss_t.item() < min_test_loss:
            min_test_loss = loss_t.item()
            min_test_epoch = epoch+1
            torch.save(rna_encoder.state_dict(), save_pth+f'/rna_encoder_best.pth')
            torch.save(pro_encoder.state_dict(), save_pth+f'/pro_encoder_best.pth')
loss_df = pd.DataFrame(loss_list,columns=['Epoch','train_pos','train_neg','train_loss','test_pos','test_neg' ,'test_loss'])
# save train and test loss as table
loss_df.to_csv(save_pth+'/loss.csv',index=None)
print('Min Test loss',min_test_loss,'Min epoch',min_test_epoch)

