import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import Detectors.Deep_Learning.AutoEncoders.utils
reload(Detectors.Deep_Learning.AutoEncoders.utils)
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model

# import flow.visualize.visualize_ring as visualize_ring
# reload(visualize_ring)
# from flow.visualize.visualize_ring import get_measured_leader,get_rel_dist_to_measured_leader,get_vel_of_measured_leader



import torch

# Anti-Flow specific functions for  detection:

from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder


from detector_dev.utils import Bando_OVM_FTL

import os

import Adversaries.controllers.car_following_adversarial
from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact
from Adversaries.controllers.car_following_adversarial import ACC_Benign


import time

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

import flow.visualize.visualize_ring as visualize_ring

from flow.visualize.visualize_ring import get_measured_leader,get_rel_dist_to_measured_leader,get_vel_of_measured_leader

from copy import deepcopy

import sys

from scipy.interpolate import interp1d




def load_ARED_data():

    ARED_data_path = '/Users/vanderbilt/Desktop/Research_2021/ARED-Model-Calibration/'

    T_vals = np.loadtxt(ARED_data_path+'vehT.csv').T
    V_vals = np.loadtxt(ARED_data_path+'vehV.csv').T
    S_vals = np.loadtxt(ARED_data_path+'vehS.csv').T
    VL_vals = np.loadtxt(ARED_data_path+'vehVL.csv').T
    X_vals = np.loadtxt(ARED_data_path+'vehX.csv').T


    print('ARED data loaded in.')

    return T_vals,V_vals,S_vals,VL_vals,X_vals


def get_ARED_data_resampled_for_detection(T,V,S,VL,X):

    V_resampled = []
    S_resampled = []
    T_resampled = []
    X_resampled = []
    VL_resampled = []

    # veh_ids = range(21) # number of vehicles

    veh_ids = range(T.shape[1])

    timeseries_dict = dict.fromkeys(veh_ids)

    for veh_id in veh_ids:
        times = T[:,veh_id]
        speeds = V[:,veh_id]
        spacings = S[:,veh_id]
        lead_speeds = VL[:,veh_id]
        rel_vel = lead_speeds - speeds
        
        # resample everything to be at 10 herz so it works with detection hyper parameters:
        
        resampled_times = np.arange(times[0],times[-1],0.1)
        T_resampled.append(resampled_times)
        #speed
        speed_interp = interp1d(times, speeds, kind='cubic')
        resampled_speeds = speed_interp(resampled_times)
        V_resampled.append(resampled_speeds)
        #spacing
        spacing_interp = interp1d(times, spacings, kind='cubic')
        resampled_spacings = spacing_interp(resampled_times)
        S_resampled.append(resampled_spacings)
        #relative speed:
        rel_vel_interp = interp1d(times, rel_vel, kind='cubic')
        resampled_rel_vel = rel_vel_interp(resampled_times)
        VL_resampled.append(resampled_rel_vel)
        #position:
        position_interp = interp1d(times, X[:,veh_id], kind='cubic')
        X_resampled.append(position_interp(resampled_times))
        
        
        data = np.array([resampled_times,resampled_speeds,resampled_spacings,resampled_rel_vel]).T
        
        timeseries_dict[veh_id] = data

    V_resampled = np.array(V_resampled).T
    S_resampled = np.array(S_resampled).T
    T_resampled = np.array(T_resampled).T
    X_resampled = np.array(X_resampled).T
    VL_resampled = np.array(VL_resampled).T

    return V_resampled,S_resampled,T_resampled,X_resampled,VL_resampled,timeseries_dict





def train_detector(timeseries_dict,model_name='ARED_detector',n_epoch=100,num_samples_per_veh=10):

    veh_ids = list(timeseries_dict.keys())

    timeseries_list = []

    for veh_id in veh_ids:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

    train_X = make_train_X(timeseries_list,num_samples_per_veh = num_samples_per_veh)

    model = get_cnn_lstm_ae_model(n_features=4)

    model_file_name = model_name

    print('Model: '+model_file_name)

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model



def get_rec_errors(timeseries_dict,model,want_timeseries_plot=False):
    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in veh_ids:
        timeseries_list = []
        
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

        timeseries_list = [speed,accel,head_way,rel_vel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

    
    if(want_timeseries_plot):
        plt.figure()
        
        for veh_id in veh_ids:
            smoothed_loss = smoothed_losses[veh_id]
            plt.plot(smoothed_loss)
        
    return smoothed_losses

from sklearn.cluster import KMeans

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
        return np.zeros_like(labels),cluster_centroids