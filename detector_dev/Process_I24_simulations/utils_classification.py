from sklearn.cluster import KMeans

import matplotlib.pyplot as pt
import sys
import numpy as np
import time
from copy import deepcopy

import csv

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth



def k_means_cluster(rec_error_dict,cluster_diff=0.1):

    max_losses = []
    veh_ids = []
    for veh_id in rec_error_dict:
        losses = rec_error_dict[veh_id]
        max_losses.append(np.max(losses))
        veh_ids.append(veh_id)

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

    labels_dict = dict.fromkeys(rec_error_dict.keys())

    if(np.min(positive_labels)-cluster_diff > np.max(negative_labels)):

        for i in range(len(veh_ids)):
            veh_id = veh_ids[i]
            label = labels[i]
            labels_dict[veh_id] = label

    else:
        for i in range(len(veh_ids)):
            veh_id = veh_ids[i]
            label = labels[i]
            labels_dict[veh_id] = 0


    return labels_dict,cluster_centroids

def threshold_classifer(rec_error_dict,threshold):
    classifier_labels_dict = dict.fromkeys(rec_error_dict.keys())
    for veh_id in rec_error_dict:
        rec_errors = rec_error_dict[veh_id]
        max_rec_error = np.max(rec_errors)

        if(max_rec_error > threshold):
            classifier_labels_dict[veh_id] = 1
        else:
            classifier_labels_dict[veh_id] = 0

    return classifier_labels_dict


def f1_calculation(assigned_labels_list,vehicle_labels_list):
    tp = 0
    fp = 0
    fn = 0

    num_samples = len(assigned_labels_list)

    for i in range(num_samples):
        classifier_label = assigned_labels_list[i]
        true_label = vehicle_types_list[i]
        if(true_label == 1 and classifier_label == 1):
            tp += 1
        elif(true_label == 1 and classifier_label == 0):
            fp += 1
        elif(true_label == 0 and classifier_label == 1):
            fn += 1

    f1_score = tp/(tp + 1/2 (fp + fn))

    return f1






