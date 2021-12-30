## Authors: Huichen Li, George Gunter (2020)
import numpy as np
import random
import torch
import torch.nn as nn

import os
import argparse
import datetime

from utils import SeqDataset
from utils import train_epoch, eval_data
from utils import plot_seqs


class AutoEncoder(nn.Module):
    def __init__(self, seq_len, n_feature, device):
        super(AutoEncoder, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.n_feature = n_feature

        self.n_hidden1 = 100
        self.n_latent = 10
        self.n_hidden2 = self.n_hidden1

        self.encoder = nn.Sequential(
            nn.Linear(self.n_feature*self.seq_len, self.n_hidden1),
            nn.ReLU(),
            nn.Linear(self.n_hidden1, self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden2),
            nn.ReLU(),
            nn.Linear(self.n_hidden2, self.n_feature*self.seq_len)
        )

        self.to(self.device)

    def forward(self, x):
        x_shape = x.shape
        x = x.to(self.device)
        _batch_size = x_shape[0]
        x = x.reshape((_batch_size, -1))
        x_e = self.encoder(x)
        x_d = self.decoder(x_e)
        x_d = x_d.reshape(x_shape)
        return x_d

    def loss(self, pred_x, true_x):
        true_x = true_x.to(self.device)
        return torch.norm(pred_x - true_x.float(), p=2, dim=1)


if __name__ == '__main__':
    n_features = 1
    batch_size = 16

    data_names = ['high_congestion_speed', 'medium_congestion_speed', 'low_congestion_speed', 'Train_X']
    for data_name in data_names:
        train_X = np.loadtxt(f'./data/{data_name}.csv')
        seq_len = train_X.shape[1]
        print(f"{data_name}, shape: {train_X.shape}")
        trainset = SeqDataset(train_X)  # 544, 100, 1 for Train_X.csv
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

        model = AutoEncoder(seq_len, n_features, 'cuda')

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        np.savez(f'./models/ae_{data_name}.npz', seq_len=seq_len)

        SAVE_PATH = f'./models/ae_{data_name}.pt'
        do_load = True
        if do_load and os.path.exists(SAVE_PATH):
            model.load_state_dict(torch.load(SAVE_PATH))

        n_epoch = 500
        if not do_load:
            best_loss = 999999
        else:
            train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
            best_loss = 1.0 * train_ls / train_tot

        for e in range(n_epoch):
            l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
            train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
            avg_loss = 1.0 * train_ls / train_tot
            if e % 10 == 0:
                print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
                      datetime.datetime.now())
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), SAVE_PATH)




