import numpy as np

import torch

import time
from time import time as timer_start

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder


import flow.visualize.visualize_ring as visualize_ring

from flow.visualize.visualize_ring import get_measured_leader,get_rel_dist_to_measured_leader,get_vel_of_measured_leader

from copy import deepcopy

import sys

import utils

from utils import assess_relative_model_on_attack

import ray

import os

import csv

from sklearn.metrics import roc_curve,auc


def get_sim_timeseries(csv_path,warmup_period=0.0):
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
                    if(len(curr_veh_data)>0):
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
        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
        end_time = time.time()
        print('Data loaded, total time: '+str(end_time-begin_time))
        

    return sim_dict

def get_sim_timeseries_all_data(csv_path,warmup_period=0.0):
    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        id_index = 0
        time_index = 0
        speed_index = 0
        headway_index = 0
        relvel_index = 0
        edge_index = 0
        pos_index = 0


        edge_list = ['Eastbound_3',':202186118','Eastbound_4','Eastbound_5','Eastbound_6',':202186134','Eastbound_7']

        row1 = next(csvreader)
        num_entries = len(row1)
        while(row1[id_index]!='id' and id_index<num_entries):id_index +=1
        while(row1[edge_index]!='edge_id' and edge_index<num_entries):edge_index +=1
        while(row1[time_index]!='time' and time_index<num_entries):time_index +=1
        while(row1[speed_index]!='speed' and speed_index<num_entries):speed_index +=1
        while(row1[headway_index]!='headway' and headway_index<num_entries):headway_index +=1
        while(row1[relvel_index]!='leader_rel_speed' and relvel_index<num_entries):relvel_index +=1


        row_num += 1

        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[id_index]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>0):
                        sim_dict[curr_veh_id] = curr_veh_data
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[id_index] # Set new veh id
                    #Allocate space for storing:
                    sim_dict[curr_veh_id] = []

                curr_veh_id = row[id_index]
                sim_time = float(row[time_index])
                edge = row[edge_index]
                if(sim_time > warmup_period and edge in edge_list):
                    # data = [time,speed,headway,leader_rel_speed]
                    curr_veh_data.append(row)
            row_num += 1

        #Add the very last vehicle's information:
        sim_dict[curr_veh_id] = curr_veh_data
        print('Data loaded.')
    return sim_dict

def get_losses_complete_obs(emission_path,model,warmup_period=600):

    begin_time = time.time()
    
    timeseries_dict = get_sim_timeseries(emission_path,warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0
    
    total_vehicles = len(veh_ids)

    testing_losses_dict = {}

    for veh_id in veh_ids:
        
        speed = timeseries_dict[veh_id][:,1]
        
        if(len(speed) > 101):
            accel = np.gradient(speed,.1)
            head_way = timeseries_dict[veh_id][:,2]
            rel_vel = timeseries_dict[veh_id][:,3]

            timeseries_list = [speed,accel,head_way,rel_vel]

            _,loss = sliding_window_mult_feat(model,timeseries_list)

            testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        if(num_veh_processed % 100 == 0):
            total_compute_time = timer_start()-begin_time
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

def get_rec_errors(timeseries_dict,model,seq_len=100,warmup_period=1200):

    begin_time = time.time()

    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0
    
    total_vehicles = len(veh_ids)

    testing_losses_dict = {}

    for veh_id in veh_ids:
        
        speed = timeseries_dict[veh_id][:,1]
        
        if(len(speed) > seq_len):
            accel = np.gradient(speed,.1)
            head_way = timeseries_dict[veh_id][:,2]
            rel_vel = timeseries_dict[veh_id][:,3]

            timeseries_list = [speed,accel,head_way,rel_vel]

            _,loss = sliding_window_mult_feat(model,timeseries_list)

            testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        if(num_veh_processed % 100 == 0):
            total_compute_time = timer_start()-begin_time
            print('Vehicles processed: '+str(num_veh_processed)+'/'+str(total_vehicles)+', total compute time: '+str(total_compute_time))
        
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

def write_losses_to_file(smoothed_losses,file_name):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for veh_id in smoothed_losses:
            vehicle_times = smoothed_losses[veh_id][0]
            losses = smoothed_losses[veh_id][1]
            num_samples = len(losses)
            for i in range(num_samples):
                writer.writerow([veh_id,vehicle_times[i],losses[i]])
                
            
    print('Loss file written to csv.')

def get_sim_name(file_name):
    i = 0
    while(file_name[i:i+3] != 'Dur'):i+=1
    return file_name[i:]

def process_file_for_losses(emission_file_path,loss_emission_repo,model,warmup_period=1200):
    timeseries_dict = get_sim_timeseries(csv_path=emission_file_path,warmup_period=warmup_period)
    i24_losses_test = get_rec_errors(timeseries_dict,model,warmup_period=warmup_period)
    sim_name = get_sim_name(emission_file_path)
    file_path_to_write = os.path.join(loss_emission_repo,sim_name)
    write_losses_to_file(i24_losses_test,file_path_to_write)
    return file_path_to_write

@ray.remote
def process_file_for_losses_ray(emission_file_path,loss_emission_repo,model,warmup_period=1200):
    return process_file_for_losses(emission_file_path,loss_emission_repo,model,warmup_period)


def get_losses_csv(file_path):
    loss_dict = {}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            veh_id = row[0]
            time = float(row[1])
            loss = float(row[2])
            if(veh_id not in loss_dict.keys()):
                loss_dict[veh_id] = []
                loss_dict[veh_id].append([time,loss])
            else:
                loss_dict[veh_id].append([time,loss])
    for veh_id in loss_dict:
        loss_dict[veh_id] = np.array(loss_dict[veh_id])
    return loss_dict

