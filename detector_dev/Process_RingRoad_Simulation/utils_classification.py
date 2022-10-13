from sklearn.cluster import KMeans

import matplotlib.pyplot as pt
import sys
import numpy as np
import time
from copy import deepcopy

import csv

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth



def k_means_cluster(max_losses,cluster_diff=0.1):

    min_l = np.min(max_losses)
    max_l = np.max(max_losses)
    normalize_losses = (max_losses-min_l)/(max_l-min_l)
    X = normalize_losses
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(-1,1))
    labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    positive_labels = []
    negative_labels = []

    for i in range(len(X)):
        l = X[i]
        label = labels[i]
        if(label==0):negative_labels.append(l)
        else:positive_labels.append(l) 

    if(np.min(positive_labels)-cluster_diff > np.max(negative_labels)):
        return labels,cluster_centroids
    else:
        return np.zeros_like(labels)




def get_F1_score(assigned_labels,true_labels):
    num_TPs = 0
    num_FPs = 0
    num_TNs = 0
    num_FNs = 0


    for i in range(len(assigned_labels)):
        if(assigned_labels[i] == 0 and true_labels[i] == 0):num_TNs += 1
        elif(assigned_labels[i] == 1 and true_labels[i] == 1):num_TPs += 1
        elif(assigned_labels[i] == 1 and true_labels[i] == 0):num_FPs += 1
        elif(assigned_labels[i] == 0 and true_labels[i] == 1):num_FNs += 1

    F1_score = num_TPs






