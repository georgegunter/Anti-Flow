## Authors: Huichen Li, George Gunter (2020)

import numpy as np
# import sklearn
# from sklearn import metrics
import matplotlib.pyplot as plt
import os
import torch

#%%
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        Xs = np.expand_dims(self.X[idx], axis=1)
        ys = np.expand_dims(self.X[idx], axis=1)
        return Xs, ys

def train_epoch(model, optimizer, dataloader):
    model.train()

    cum_loss = 0.0
    tot_num = 0.0
    for X, y in dataloader:
        optimizer.zero_grad()
        B = X.shape[0]
        pred = model(X.float())

        if(pred.shape[2] != 1):
            flat_pred = flatten_preds(pred)
            flat_X = X.reshape(flat_pred.shape)
            loss = model.loss(flat_pred,flat_X)
        else:
            loss = model.loss(pred, X)

        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()

        # cum_loss += loss.item()
        cum_loss += loss.detach().cpu().numpy()
        # pred_c = pred.max(1)[1].cpu()
        tot_num = tot_num + B

    return cum_loss / tot_num

def eval_data(model, dataloader):
    model.eval()
    tot = 0
    ls = 0.0
    for X, y in dataloader:
        pred = model(X.float())
        if(pred.shape[2] != 1):
            flat_pred = flatten_preds(pred)
            flat_X = X.reshape(flat_pred.shape)
            loss = model.loss(flat_pred,flat_X)
        else:
            loss = model.loss(pred, X)
        loss_total = torch.sum(loss)

        ls += loss_total.detach().cpu().numpy()
        tot += X.size()[0]
    return ls, tot

def flatten_preds(tensor_batch):
    batch_num = tensor_batch.shape[0]
    seq_len = tensor_batch.shape[1]
    feature_num = tensor_batch.shape[2]

    tensor_flattened = torch.zeros(batch_num,seq_len*feature_num,1)

    for i in range(tensor_batch.shape[0]):
        x = tensor_batch[i]
        x_flat = x.T.flatten()
        x_flat = x_flat.reshape(seq_len*feature_num,1)
        tensor_flattened[i] = x_flat #The transpose is a little weird
    return tensor_flattened

def eval_inputs(model, data, batch_size = 16):
    # model.eval()
    dataSet = SeqDataset(data)
    dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False)
    
    preds = []
    losses = []
    
    for X, y in dataloader:
        pred = model(X.float())
        if(pred.shape[2] != 1):
            flat_pred = flatten_preds(pred)
            flat_X = X.reshape(flat_pred.shape)
            loss = model.loss(flat_pred,flat_X)
        else:
            loss = model.loss(pred, X)
      
        for i in range(X.shape[0]):
            preds.append(pred[i].detach().cpu().numpy().T)
            losses.append(loss[i].detach().cpu().numpy()) 
    return [np.array(preds),np.array(losses)]
        
def sliding_window(model,input_data,seq_len=100,want_centering=False,return_x=False):
    # This may need changing when there are multiple features...
    input_vals = []

    for i in range(len(input_data)-seq_len):
        if(want_centering):
            x = input_data[i:i+seq_len]
            x_centered = x - np.mean(x)
            input_vals.append(x_centered)
        else:
            input_vals.append(input_data[i:i+seq_len])
        
    input_vals = np.array(input_vals)
    reconstructions,losses = eval_inputs(model,input_vals)
    if return_x:
        return [reconstructions,losses,input_vals]
    else:
        return [reconstructions,losses]

def sliding_window_mult_feat(model,timeseries_list,n_features=4,seq_len=100):
    input_vals = []
    t_length = len(timeseries_list[0])
    num_samples = t_length-seq_len

    for i in range(num_samples):
        start = i
        end = i+seq_len

        data_sample = np.zeros([n_features*seq_len,1])

        for j in range(n_features):
            x = timeseries_list[j][start:end].reshape([seq_len,1])
            data_sample[j*seq_len:(j+1)*seq_len] = x

        input_vals.append(data_sample)
        
    input_vals = np.array(input_vals)

    reconstructions,losses = eval_inputs(model,input_vals)

    return [reconstructions,losses]

def get_loss_filter_indiv(time_samples,losses,loss_window_length=100):
    
    veh_losses_filtered = np.zeros_like(time_samples)
    loss_counts = np.zeros_like(time_samples)

    for i in range(len(losses)):
        l = losses[i]
        # for j in
        veh_losses_filtered[i:i+loss_window_length] = veh_losses_filtered[i:i+loss_window_length] + l
        loss_counts[i:i+loss_window_length] = loss_counts[i:i+loss_window_length] + 1
    i+=1
    veh_losses_filtered[i:i+loss_window_length] = veh_losses_filtered[i:i+loss_window_length] + l
    loss_counts[i:i+loss_window_length] = loss_counts[i:i+loss_window_length] + 1

    veh_losses_filtered = np.divide(veh_losses_filtered,loss_counts)

    return veh_losses_filtered

def get_losses(model, dataloader):
    losses = []

    for X, y in dataloader:
        pred = model(X.float())
        loss = model.loss(pred, y)
        losses += list(loss.detach().cpu().numpy().reshape(-1))

    return losses


# def plot_seqs(model, dataloader):
#     model.eval()
#     tot = 0
#     ls = 0.0
#     fig = plt.figure()
#     for X, y in dataloader:
#         pred = model(X.float())
#         loss = torch.sum(model.loss(pred, y))
#         ls += loss.detach().cpu().numpy()
#         tot += X.size()[0]
#         for i in range(X.shape[0]):
#             x_s = list(range(X.shape[1]))
#             # plt.plot(x_s, X[i].detach().cpu().numpy().reshape(-1), linestyle='-.')
#             plt.plot(x_s, pred[i].detach().cpu().numpy().reshape(-1))
#     plt.savefig('preds.png')
#     plt.close()
#     return ls, tot


# if __name__ == '__main__':
#     device = 'cuda'
#     seq_len = 100
#     n_features = 1

#     model = lstm_ae.RecurrentAutoencoder(seq_len, n_features, 128)
#     model = model.to(device)

#     batch_size = 1

#     benign_X = np.loadtxt('./data/Train_X.csv')  # TODO: change to benign test data file name
#     benign_labels = [0 for _ in range(benign_X.shape[0])]  # assign 0 as labels to benign data
#     benignset = lstm_ae.SeqDataset(benign_X)
#     benignloader = torch.utils.data.DataLoader(benignset, batch_size=batch_size, shuffle=False)

#     mal_X = np.loadtxt('./data/Train_X.csv')  # TODO: change to malicious test data file name
#     mal_labels = [1 for _ in range(mal_X.shape[0])]  # assign 1 as labels to malicious data
#     malset = lstm_ae.SeqDataset(mal_X)
#     malloader = torch.utils.data.DataLoader(malset, batch_size=batch_size, shuffle=False)

#     SAVE_PATH = './models/lstm_2_ae.pt'
#     model.load_state_dict(torch.load(SAVE_PATH))

#     # use reconstruction losses as the predicted labels for the two sets
#     benign_losses = get_losses(model, dataloader=benignloader)
#     mal_losses = get_losses(model, dataloader=malloader)

#     # concatenate the two parts together
#     both_labels = benign_labels + mal_labels
#     both_losses = benign_losses + mal_losses

#     # TODO: change dir_path and plot_name to what you want
#     eval_auc(ys=both_labels, preds=both_losses, pos_label=1, do_plot=True, dir_path='./plots', plot_name='test_random')


