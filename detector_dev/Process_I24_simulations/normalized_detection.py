
import numpy as np

from importlib import reload

import Data_Processing.sim_processing_utils as sim_processing_utils

from Data_Processing.sim_processing_utils import get_trajectory_timeseries

from importlib import reload

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

from copy import deepcopy

import os

import torch

import detector_dev.Process_I24_simulations.i24_utils as i24_utils

from proccess_i24_losses import *

import ray

import time



def make_timeseries_list(trajectory_dict):
    timeseries_list = []
    for veh_id in trajectory_dict:
        trajectory_samples = []
        trajectory_data = trajectory_dict[veh_id]
        
        speed = trajectory_data[:,1]
        accel = np.gradient(speed,.1)
        head_way = trajectory_data[:,2]
        rel_vel = trajectory_data[:,3]
        
        trajectory_samples.append(speed)
        trajectory_samples.append(accel)
        trajectory_samples.append(head_way)
        trajectory_samples.append(rel_vel)
        
        timeseries_list.append(trajectory_samples)
    return timeseries_list

def make_timeseries_list_normalized(trajectory_dict,normalizing_factors=[30.0,28.41799494,500.0,23.09151303]):
    timeseries_list_non_normalized = make_timeseries_list(trajectory_dict)
    num_timeseries = len(timeseries_list_non_normalized[0])

    if(normalizing_factors is None):
        normalizing_factors = np.zeros(num_timeseries,)
        
        for t_list in timeseries_list_non_normalized:
            for i in range(num_timeseries):
                max_abs_val = np.max(np.abs(t_list[i]))
                if(max_abs_val > normalizing_factors[i]): normalizing_factors[i] = max_abs_val
    
    normalized_timeseries_list = []
    
    for t_list in timeseries_list_non_normalized:
        t_list_normed = deepcopy(t_list)
        for i in range(num_timeseries):
            t_list_normed[i] = t_list_normed[i]/normalizing_factors[i]
        normalized_timeseries_list.append(t_list_normed)
            
    return normalized_timeseries_list,normalizing_factors

def get_rec_errors_normalized(timeseries_dict,model,normalizing_factors=[30.0,28.41799494,500.0,23.09151303],seq_len=100,warmup_period=1200):
    
    begin_time = time.time()

    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0
    
    total_vehicles = len(veh_ids)

    testing_losses_dict = {}
        
    normalized_timeseries_list,normalizing_factors = make_timeseries_list_normalized(timeseries_dict,normalizing_factors)


    for veh_id in veh_ids:
        
        speed = timeseries_dict[veh_id][:,1]
        
        if(len(speed) > seq_len):
            
            timeseries_list = normalized_timeseries_list[num_veh_processed]

            _,loss = sliding_window_mult_feat(model,timeseries_list)

            testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        if(num_veh_processed % 50 == 0):
            total_compute_time = time.time()-begin_time
            
            sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'/'+str(total_vehicles)+', total compute time: '+str(total_compute_time)+'\r')
            
            
        
    print('\n')
    
    smoothed_losses = {}
    
    #Get smoothed loss values:
    
    for veh_id in testing_losses_dict:
        loss = testing_losses_dict[veh_id]
        if(loss is not None):
            vehicles_time = timeseries_dict[veh_id][:,0]
            smoothed_losses[veh_id] =  [vehicles_time,loss_smooth(vehicles_time,loss)]
    
    print('Total time to calculate loses: '+str(time.time()-begin_time))
    
    return smoothed_losses


def filter_timeseries_dict_for_length(timeseries_dict,seq_len):
    timeseries_dict_filtered = {}

    for veh_id in timeseries_dict:
        if(len(timeseries_dict[veh_id]) >= seq_len):
            timeseries_dict_filtered[veh_id] = timeseries_dict[veh_id]

    return timeseries_dict_filtered


def process_sim(emission_file_path,
    loss_emission_repo,
    model,
    warmup_period=1200,
    normalizing_factors=[30.0,28.41799494,500.0,23.09151303]):


    timeseries_dict = get_sim_timeseries(csv_path=emission_file_path,
        warmup_period=warmup_period)

    timeseries_dict = filter_timeseries_dict_for_length(timeseries_dict,seq_len=100)

    rec_errors_normalized = get_rec_errors_normalized(timeseries_dict,
        model,
        normalizing_factors=normalizing_factors,
        warmup_period=warmup_period)

    sim_name = get_sim_name(emission_file_path)
    file_path_to_write = os.path.join(loss_emission_repo,sim_name)
    write_losses_to_file(rec_errors_normalized,file_path_to_write)

    return file_path_to_write

@ray.remote
def process_sim_ray(emission_file_path,loss_emission_repo,model,warmup_period=1200,normalizing_factors=[30.0,28.41799494,500.0,23.09151303]):
    return process_sim(emission_file_path,loss_emission_repo,model,warmup_period)



if __name__ == '__main__':

    normalizing_factors= [30.0,28.41799494,500.0,23.09151303] #I know this is horrible practice...

    save_path = os.path.join(os.getcwd(),'models/cnn_lstm_ae_normalized_i24_detector.pt')
    model = get_cnn_lstm_ae_model(n_features=4)
    model.load_state_dict(torch.load(save_path))


    loss_emission_repo = '/Volumes/My Passport for Mac/i24_random_sample/ae_rec_error_results/1800_inflow_normalized'
    emission_file_repo = '/Volumes/My Passport for Mac/i24_random_sample/simulations/1800_inflow'


    existing_loss_files = []
    files_list = os.listdir(loss_emission_repo)
    for file_name in files_list:
        if(file_name[-3:] == 'csv'):existing_loss_files.append(file_name)

    files_list = os.listdir(emission_file_repo)

    all_emission_files = []

    for file_name in files_list:
        if(file_name[-3:] == 'csv' and file_name not in existing_loss_files):
            all_emission_files.append(file_name)

    loss_result_ids = []


    begin_processing_losses_time = time.time()

    warmup_period = 1200

    print('Loading losses. ')
    ray.init()
    for file_name in all_emission_files:
        emission_file_path = os.path.join(emission_file_repo,file_name)


        loss_result_ids.append(process_sim_ray.remote(emission_file_path,
            loss_emission_repo,
            model,
            warmup_period=warmup_period,
            normalizing_factors=normalizing_factors))

    file_results = ray.get(loss_result_ids)



    print('Finished calculating normalized reconstruction errors.')
    print('Total compute time: '+str(time.time() - begin_processing_losses_time))








