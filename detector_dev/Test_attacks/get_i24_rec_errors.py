import numpy as np
import csv
from copy import deepcopy
import ray
import sys
import time
import os

import torch

from importlib import reload

import Data_Processing.sim_processing_utils as sim_processing_utils

from Data_Processing.sim_processing_utils import get_trajectory_timeseries

from importlib import reload

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder



import detector_dev.Process_I24_simulations.i24_utils as i24_utils

from Data_Processing.sim_processing_utils import get_trajectory_timeseries




def get_sim_timeseries(csv_path,warmup_period=0.0):
    print('Loading: '+str(csv_path))
    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    begin_time = time.time()

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        id_index = 0
        time_index = 0
        speed_index = 0
        headway_index = 0
        relvel_index = 0
        edge_index = 0
        pos_index = 0

        x_index = 0
        y_index = 0


        edge_list = ['Eastbound_3',':202186118','Eastbound_4','Eastbound_5','Eastbound_6',':202186134','Eastbound_7']

        row1 = next(csvreader)
        num_entries = len(row1)
        while(row1[id_index]!='id' and id_index<num_entries):id_index +=1
        while(row1[edge_index]!='edge_id' and edge_index<num_entries):edge_index +=1
        while(row1[time_index]!='time' and time_index<num_entries):time_index +=1
        while(row1[speed_index]!='speed' and speed_index<num_entries):speed_index +=1
        while(row1[headway_index]!='headway' and headway_index<num_entries):headway_index +=1
        while(row1[relvel_index]!='leader_rel_speed' and relvel_index<num_entries):relvel_index +=1



        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[id_index]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>101):
                        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[id_index] # Set new veh id
                    #Allocate space for storing:
                    # sim_dict[curr_veh_id] = []

                curr_veh_id = row[id_index]
                sim_time = float(row[time_index])
                edge = row[edge_index]
                if(sim_time > warmup_period and edge in edge_list):
                    # data = [time,speed,headway,leader_rel_speed]

                    # Check what was filled in if missing a leader:
                    s = float(row[headway_index])
                    dv = float(row[relvel_index])
                    v = float(row[speed_index])
                    t = float(row[time_index])

                    if(s > 500):
                        s = 500.0
                        dv = 0.0

                    data = [t,v,s,dv]
                    curr_veh_data.append(data)
            row_num += 1

        #Add the very last vehicle's information:
        if(len(curr_veh_data)>101):
            sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
        end_time = time.time()
        print('Data loaded, total time: '+str(end_time-begin_time))
        

    return sim_dict



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

def get_rec_errors_normalized(timeseries_dict,model,normalizing_factors=[30.0,28.41799494,500.0,23.09151303],seq_len=100,warmup_period=500):
    
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


def write_rec_errors_to_file(rec_error_dict,file_name):
    veh_ids = list(rec_error_dict.keys())

    rec_errors_list = []

    for veh_id in veh_ids:
        RE_vals = rec_error_dict[veh_id][1]
        for i in range(len(RE_vals)):
            rec_errors_list.append([veh_id,RE_vals[i]])

    with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in rec_errors_list:
                writer.writerow(row)

    print('Written to file: '+str(file_name))

    return file_name

def process_sim(model,
    sim_file_path,
    write_file_path,
    warmup_period=500,
    normalizing_factors=[30.0,28.41799494,500.0,23.09151303]):


    # timeseries_dict =  get_trajectory_timeseries(emission_path=sim_file_path,
    #     warmup_period=warmup_period,
    #     want_print_finished_loading=False)

    timeseries_dict = get_sim_timeseries(csv_path=sim_file_path,
        warmup_period=warmup_period)

    timeseries_dict = filter_timeseries_dict_for_length(timeseries_dict,seq_len=100)

    rec_errors_normalized = get_rec_errors_normalized(timeseries_dict,
        model,
        normalizing_factors=normalizing_factors,
        warmup_period=warmup_period)

    write_rec_errors_to_file(rec_errors_normalized,write_file_path)

    return write_file_path

@ray.remote
def process_sim_ray(model,sim_file_path,write_file_path,warmup_period=500,normalizing_factors=[30.0,28.41799494,500.0,23.09151303]):
    return process_sim(model,sim_file_path,write_file_path,warmup_period)



def process_all_files(model,sim_files,write_files,warmup_period=500):

    results_ids = []

    for i in range(len(sim_files)):
        sim_file_path = sim_files[i]
        write_file_path = write_files[i]

        try:
            
            results_ids.append(process_sim_ray.remote(model,sim_file_path,write_file_path,warmup_period=warmup_period))
        except:
            print('Issue with data processing file: '+sim_file_path)

    results = ray.get(results_ids)

if __name__ == '__main__':



    normalizing_factors= [30.0,28.41799494,500.0,23.09151303] #I know this is horrible practice...

    save_path = '/Users/vanderbilt/Desktop/General_research_tools/Anti-Flow/detector_dev/Process_I24_simulations/models/cnn_lstm_ae_normalized_i24_detector.pt'
    model = get_cnn_lstm_ae_model(n_features=4)
    model.load_state_dict(torch.load(save_path))
    print('Detection models loaded.')

    ray.init(num_cpus = 4)


    want_get_max_velocity_rec_errors = False

    if(want_get_max_velocity_rec_errors):


        print('Analyzing max velocity attack.')

        sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample'

        rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample_rec_errors'

        print(sim_repo_path)

        all_files_in_sim_repo = os.listdir(sim_repo_path)

        sim_files = []

        write_files = []

        for file in all_files_in_sim_repo:
            if(('ver' in file) and ('csv' in file)):
                sim_files.append(os.path.join(sim_repo_path,file))
                write_files.append(os.path.join(rec_error_repo_path,file))

        process_all_files(model,sim_files,write_files)

        print('Found rec errors.')


    want_get_radar_warp_rec_errors = False

    if(want_get_radar_warp_rec_errors):

        print('Analyzing radar warp attack.')

        sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/i24_random_sample'

        rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/i24_random_sample_rec_errors'

        print(sim_repo_path)

        all_files_in_sim_repo = os.listdir(sim_repo_path)

        all_files_in_rec_error_repo = os.listdir(rec_error_repo_path)

        sim_files = []

        write_files = []

        for file in all_files_in_sim_repo:
            if(('ver' in file) and ('csv' in file)):
                if(file not in all_files_in_rec_error_repo):
                    sim_files.append(os.path.join(sim_repo_path,file))
                    write_files.append(os.path.join(rec_error_repo_path,file))

        process_all_files(model,sim_files,write_files)

        print('Found rec errors.')

    want_get_RDA_rec_errors = True

    if(want_get_RDA_rec_errors):

        print('Analyzing radar warp attack.')

        sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/i24_random_sample'

        rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/i24_random_sample_rec_errors'

        print(sim_repo_path)

        all_files_in_sim_repo = os.listdir(sim_repo_path)

        sim_files = []

        write_files = []

        for file in all_files_in_sim_repo:
            if(('ver' in file) and ('csv' in file)):
                sim_files.append(os.path.join(sim_repo_path,file))
                write_files.append(os.path.join(rec_error_repo_path,file))

        process_all_files(model,sim_files,write_files)

        print('Found rec errors.')

