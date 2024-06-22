import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import torch

from detector_dev.utils import run_ring_sim_variable_cfm,Bando_OVM_FTL

from Adversaries.controllers.car_following_adversarial import ACC_Benign

import detector_dev.utils as utils

import Data_Processing.sim_processing_utils as sim_processing_utils

from Data_Processing.sim_processing_utils import get_trajectory_timeseries


from Detectors.Deep_Learning.AutoEncoders.utils import *

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from copy import deepcopy

import ray

import sys

import csv

print('Libraries loaded.')



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


def make_timeseries_list_normalized(trajectory_dict,normalizing_factors=None):
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


def get_rec_errors_normalized(timeseries_dict,model,normalizing_factors=[13.91463681, 10.29542389, 47.98582978,  7.21631525],seq_len=100,warmup_period=1200):
    
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

        if(num_veh_processed % 10 == 0):
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


def get_sim_name(emission_file_path):
	num_chars = len(emission_file_path)
	begin_file_name = 0
	j = 0
	while(j<num_chars):
		if(emission_file_path[j] == '/'):
			begin_file_name = j
		j += 1

	return emission_file_path[begin_file_name+1:]

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


def process_sim(emission_file_path,
    loss_emission_repo,
    model,
    warmup_period=100,
    normalizing_factors=[13.91463681, 10.29542389, 47.98582978,  7.21631525]):


    timeseries_dict = get_trajectory_timeseries(csv_path=emission_file_path,
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
def process_sim_ray(emission_file_path,loss_emission_repo,model,warmup_period=1200,normalizing_factors=[13.91463681, 10.29542389, 47.98582978,  7.21631525]):
    return process_sim(emission_file_path,loss_emission_repo,model,warmup_period)




if __name__ == '__main__':


	

	# simulation for which no attack has occurred:
	training_data_file_path = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep/ring_600m_double_lane_TAD_0.0_ADR_0.0_ver_5.csv'

	model = get_cnn_lstm_ae_model(n_features=4)

	model_is_trained = False

	if(model_is_trained):
		model_path = 'models/cnn_lstm_ae_normalized_double_lane_ring_detector.pt'
		# model_path = os.path.join(os.getcwd(),model_path)
		model.load_state_dict(torch.load(model_path))
		print('Trained model loaded.')


	# normalizing_factors = [13.91463681, 10.29542389, 47.98582978,  7.21631525]# found previously

	train_on_benign = True

	if(train_on_benign):

		trajectory_dict_training_data = get_trajectory_timeseries(training_data_file_path)

		normalized_timeseries_list,normalizing_factors = make_timeseries_list_normalized(trajectory_dict_training_data)


		# normalizing_factors works out to [13.91463681, 10.29542389, 47.98582978,  7.21631525]

		train_X = make_train_X(normalized_timeseries_list)
		print('Training data prepared.')

		model_file_name = 'normalized_double_lane_ring_detector'

		model = train_model(model,train_X,model_file_name)

	sim_repo_path = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep'

	rec_error_repo_path = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep/normalized_rec_errors'

	all_files_in_sim_repo = os.listdir(sim_repo_path)

	sim_files = []

	for file in all_files_in_sim_repo:
		if('ver_1.csv' in file):
			sim_files.append(os.path.join(sim_repo_path,file))

	ray.init(num_cpus=4)


	result_ids = []

	for file in sim_files:

		print('File: '+file)

		result_ids.append(
			process_sim_ray.remote(emission_file_path=file,
				loss_emission_repo=rec_error_repo_path,
				model=model,
				warmup_period=100,
				normalizing_factors=[13.91463681, 10.29542389, 47.98582978,  7.21631525]))


	results = ray.get(result_ids)











