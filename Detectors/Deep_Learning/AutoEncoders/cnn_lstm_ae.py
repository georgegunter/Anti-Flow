## Authors: Huichen Li, George Gunter (2020)

import torch
import numpy as np
import math
import os
import datetime

class Encoder(torch.nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim
        self.cnn_channels, self.kernel_size, self.stride = cnn_channels, kernel_size, stride

        self.cnn = torch.nn.Conv1d(in_channels=self.n_features, out_channels=self.cnn_channels,
                                    kernel_size=self.kernel_size, stride=self.stride)

        self.cnn_len = math.floor((self.seq_len - (self.kernel_size - 1) - 1) / stride + 1)
        self.cnn_features = cnn_channels

        self.rnn = torch.nn.LSTM(
            input_size=self.cnn_features,
            hidden_size=self.embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        _batch_size = x.shape[0]
        x = x.reshape((_batch_size, self.n_features, self.seq_len))
        x_cnn = self.cnn(x)
        x_cnn = x_cnn.reshape((_batch_size, self.cnn_len, self.cnn_features))
        x_rnn, (hidden_n, _) = self.rnn(x_cnn)
        return hidden_n.reshape((_batch_size, 1, self.embedding_dim))


class Decoder(torch.nn.Module):
    def __init__(self, seq_len, embedding_dim, n_features):
        super(Decoder, self).__init__()
        self.seq_len, self.embedding_dim, self.n_features = seq_len, embedding_dim, n_features
        self.hidden_dim = embedding_dim*2
        self.rnn = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        _batch_size = x.shape[0]
        x = x.repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn(x)
        x = x.reshape((_batch_size, self.seq_len, self.hidden_dim))
        x = self.output_layer(x)
        return x


class CNNRecurrentAutoencoder(torch.nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device):
        super(CNNRecurrentAutoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride).to(self.device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, pred_x, true_x):
        true_x = true_x.to(self.device)
        return torch.norm(pred_x - true_x.float(), p=2, dim=1)


# if __name__ == '__main__':
#     n_features = 1
#     embedding_dim = 32
#     cnn_channels = 8
#     kernel_size = 16
#     stride = 1
#     batch_size = 16
#     device = 'cuda'

#     data_names = ['high_congestion_speed', 'medium_congestion_speed', 'low_congestion_speed', 'Train_X']
#     for data_name in data_names:
#         #%%
#         train_X = np.loadtxt(f'./data/{data_name}.csv')
#         seq_len = train_X.shape[1]
#         print(f"{data_name}, shape: {train_X.shape}")
#         trainset = SeqDataset(train_X)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

#         model = CNNRecurrentAutoencoder(seq_len, n_features, embedding_dim, cnn_channels, kernel_size, stride, device)

#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

#         np.savez(f'./models/cnn_lstm_ae_{data_name}.npz', seq_len=seq_len, embedding_dim=embedding_dim,
#                  cnn_channels=cnn_channels, kernel_size=kernel_size, stride=stride)

#         SAVE_PATH = f'./models/cnn_lstm_ae_{data_name}.pt'
#         do_load = True
#         if do_load and os.path.exists(SAVE_PATH):
#             model.load_state_dict(torch.load(SAVE_PATH))

#         n_epoch = 500
#         if not do_load:
#             best_loss = 999999
#         else:
#             train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
#             best_loss = 1.0 * train_ls/train_tot

#         for e in range(n_epoch):
#             l = train_epoch(model=model, optimizer=optimizer, dataloader=trainloader)
#             train_ls, train_tot = eval_data(model=model, dataloader=trainloader)
#             avg_loss = 1.0 * train_ls / train_tot
#             if e % 10 == 0:
#                 print("Epoch %d, total loss %f, total predictions %d, avg loss %f" % (e, train_ls, train_tot, avg_loss),
#                       datetime.datetime.now())
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 torch.save(model.state_dict(), SAVE_PATH)



