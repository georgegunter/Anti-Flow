## Authors: Huichen Li, George Gunter (2020)
# https://curiousily.com/posts/time-series-anomaly-detection-using-lstm-autoencoder-with-pytorch-in-python/

import torch
import torch.nn as nn
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os

from Detectors.Deep_Learning.AutoEncoders.utils import train_epoch, eval_data
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset
from Detectors.Deep_Learning.AutoEncoders.utils import plot_seqs


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        _batch_size = x.shape[0]
        x = x.reshape((_batch_size, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((_batch_size, 1, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        _batch_size = x.shape[0]
        x = x.repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((_batch_size, self.seq_len, self.hidden_dim))
        x = self.output_layer(x)
        return x


class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, device='cuda'):
        super(RecurrentAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(self.device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred_x, true_x):
        true_x = true_x.to(self.device)
        return torch.norm(pred_x - true_x.float(), p=2, dim=1)


if __name__ == '__main__':
    n_features = 1
    embedding_dim = 128
    batch_size = 16

    data_names = ['high_congestion_speed', 'medium_congestion_speed', 'low_congestion_speed', 'Train_X']
    for data_name in data_names:
        train_X = np.loadtxt(f'./data/{data_name}.csv')
        seq_len = train_X.shape[1]
        print(f"{data_name}, shape: {train_X.shape}")
        trainset = SeqDataset(train_X)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

        model = RecurrentAutoencoder(seq_len, n_features, embedding_dim)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        np.savez(f'./models/lstm_ae_{data_name}.npz', seq_len=seq_len, embedding_dim=embedding_dim)

        SAVE_PATH = f'./models/lstm_ae_{data_name}.pt'
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