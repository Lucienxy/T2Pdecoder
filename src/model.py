## This file contains the model definition for the CLIP model and VAE model

import pandas as pd
import numpy as np
import torch
import random
import os
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F


class EmbeddingModel(nn.Module):
    def __init__(self, input_dim, node_num, embedding_dim):
        super(EmbeddingModel, self).__init__()

        self.fc0 = nn.Linear(input_dim, node_num[0])
        self.bn0 = nn.BatchNorm1d(node_num[0])

        self.fc1 = nn.Linear(node_num[0], node_num[1])
        self.bn1 = nn.BatchNorm1d(node_num[1])

        self.fc2 = nn.Linear(node_num[1], node_num[2])
        self.bn2 = nn.BatchNorm1d(node_num[2])

        # self.fc3 = nn.Linear(node_num[2], node_num[3])
        # self.bn3 = nn.BatchNorm1d(node_num[3])

        # self.fc4 = nn.Linear(node_num[3], node_num[-1])
        # self.bn4 = nn.BatchNorm1d(node_num[-1])

        self.fc5 = nn.Linear(node_num[-1], embedding_dim)
        self.bn5 = nn.BatchNorm1d(embedding_dim)
        self._initialize_weights()

    def forward(self, x):
        h0 = self.bn0(F.relu(self.fc0(x)))
        h1 = self.bn1(F.relu(self.fc1(h0)))
        h2 = self.bn2(F.relu(self.fc2(h1)))
        # h3 = self.bn3(F.relu(self.fc3(h2)))
        # h4 = self.bn4(F.relu(self.fc4(h3)))
        # h5 = self.bn5(self.fc5(h4))
        h5 = self.fc5(h2)
        return h5
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Scaler(nn.Module):
    def __init__(self, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * torch.sqrt(scale)

class Sampling(nn.Module):
    def forward(self, inputs):
        z_mean, z_std = inputs
        noise = torch.randn_like(z_mean)
        return z_mean + z_std * noise


class VAE(nn.Module):
    def __init__(self, input_size=512, hidden_size=3,node_num = [256,128,64],p_drop = 0.1):
        super(VAE, self).__init__()
        self.scaler = Scaler()
        # node_num = [256, 128, 64]
        self.node_num = node_num
        self.p = p_drop
        self.fc1 = nn.Linear(input_size, node_num[0])
        # self.bn1 = nn.BatchNorm1d(node_num[0])
        self.fc2 = nn.Linear(node_num[0], node_num[1])
        # self.bn2 = nn.BatchNorm1d(node_num[1])
        self.fc3 = nn.Linear(node_num[1], node_num[2])
        # self.bn3 = nn.BatchNorm1d(node_num[2])
        self.fc_mean = nn.Linear(node_num[2], hidden_size)
        self.fc_logvar = nn.Linear(node_num[2], hidden_size)
        self.bn_z_mean = nn.BatchNorm1d(hidden_size, affine=False, eps=1e-8)
        self.bn_z_std = nn.BatchNorm1d(hidden_size, affine=False, eps=1e-8)
        self.sampling = Sampling()

        self.fc5 = nn.Linear(hidden_size, node_num[2])
        # self.bn5 = nn.BatchNorm1d(node_num[2])
        self.fc6 = nn.Linear(node_num[2], node_num[1])
        # self.bn6 = nn.BatchNorm1d(node_num[1])
        self.fc7 = nn.Linear(node_num[1], node_num[0])
        # self.bn7 = nn.BatchNorm1d(node_num[0])
        self.fc8 = nn.Linear(node_num[0], input_size)
        # self._initialize_weights()

    def encode(self, x):
        # add dropout layer
        p=self.p
        h1 = F.relu(self.fc1(x))
        h1 = F.dropout(h1, p=p,training=self.training)
        h2 = F.relu(self.fc2(h1))
        h2 = F.dropout(h2, p=p,training=self.training)
        h3 = F.relu(self.fc3(h2))
        mu, logvar = self.fc_mean(h3), self.fc_logvar(h3)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        z_mean = self.bn_z_mean(mu)
        z_mean = self.scaler(z_mean, mode='positive')
        z_std = self.bn_z_std(logvar)
        z_std = self.scaler(z_std, mode='negative')
        z = self.sampling((z_mean, z_std))
        return z, z_mean

    def decode(self, z):
        p=self.p
        h5 = F.relu(self.fc5(z))
        h5 = F.dropout(h5, p=p,training=self.training)
        h6 = F.relu(self.fc6(h5))
        h6 = F.dropout(h6, p=p,training=self.training)
        h7 = F.relu(self.fc7(h6))
        h7 = F.dropout(h7, p=p,training=self.training)
        h8 = self.fc8(h7)
        return h8

    def forward(self, x):
        mu, logvar = self.encode(x)
        z,z_mean = self.reparameterize(mu, logvar)
        decoded = self.decode(z)
        return decoded, mu, logvar, z,z_mean

    def loss_function(self, recon_x, x, mu, logvar, KLD_weight=1):
        MSE = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = MSE + KLD * KLD_weight
        return loss, MSE, KLD

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

