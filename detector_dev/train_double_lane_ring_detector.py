import os
import numpy as np
import matplotlib.pyplot

import Detectors.Deep_Learning.AutoEncoders.utils

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat
from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

import torch

from Data_Processing.sim_processing_utils import get_trajectory_timeseries

import time

def train_double_lane_detection_model(timeseries_dict,n_epoch=600,model=None,seq_len=100):
    veh_ids = list(timeseries_dict.keys())
    timeseries_list = []

    for veh_id in veh_ids:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        num_samples = len(timeseries_dict[veh_id][:,0])
        if(num_samples > seq_len):
            speed = timeseries_dict[veh_id][:,1]
            accel = np.gradient(speed,.1)
            head_way = timeseries_dict[veh_id][:,2]
            rel_vel = timeseries_dict[veh_id][:,3]
         
            timeseries_list.append([speed,accel,head_way,rel_vel])
        
    train_X = make_train_X(timeseries_list,seq_len=seq_len)
    
    if(model is None):
        model = get_cnn_lstm_ae_model(n_features=4,seq_len=seq_len)
    
    model_file_name = 'double_lane_ring_length600'
    
    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch,seq_len=seq_len)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model



if __name__ == '__main__':

	sim_file_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/double_lane_ring_road_attack_parameter_sweep'

	sim_file_path = os.path.join(sim_file_repo_path,'ring_600m_double_lane_TAD_0.0_ADR_0.0_ver_1.csv')

	timeseries_dict = get_trajectory_timeseries(sim_file_path,warmup_period=0,want_print_finished_loading=True)

	train_double_lane_detection_model(timeseries_dict,n_epoch=600,seq_len=100)







