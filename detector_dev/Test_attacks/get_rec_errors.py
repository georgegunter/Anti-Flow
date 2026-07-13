import numpy as np
import torch
import time
from copy import deepcopy
import sys

import ray
import os
import csv
from sklearn.metrics import roc_curve,auc


from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

from Data_Processing.sim_processing_utils import get_trajectory_timeseries


def get_rec_errors(emission_path,model,warmup_period=20):

    timeseries_dict =  get_trajectory_timeseries(emission_path,warmup_period=warmup_period,want_print_finished_loading=False)

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
    
    rec_error_dict = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        rec_error_dict[veh_id] =  loss_smooth(time,loss)
        
    return rec_error_dict

@ray.remote
def get_rec_error_ray_helper(emission_path,model,warmup_period=20):
    rec_error_dict = get_rec_errors(emission_path,model,warmup_period)
    return rec_error_dict

def write_rec_errors_to_file(rec_error_dict,file_name):
	veh_ids = list(rec_error_dict.keys())

	rec_errors_list = []

	for veh_id in veh_ids:
		RE_vals = rec_error_dict[veh_id]
		for i in range(len(RE_vals)):
			rec_errors_list.append([veh_id,RE_vals[i]])

	with open(file_name, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for row in rec_errors_list:
				writer.writerow(row)

	print('Written to file: '+str(file_name))

	return file_name

def process_file(model,sim_file_path,write_file_path):
	rec_error_dict = get_rec_errors(sim_file_path,model)
	return write_rec_errors_to_file(rec_error_dict,write_file_path)

@ray.remote
def process_file_ray_helper(model,sim_file_path,write_file_path):
	return process_file(model,sim_file_path,write_file_path)


# @ray.remote
# def get_relevant_data_ray(model,sim_file_path,write_file_path):
# 	return get_relevant_data(file_path)


def process_all_files(model,sim_files,write_files):

	results_ids = []

	for i in range(len(sim_files)):
		sim_file_path = sim_files[i]
		write_file_path = write_files[i]

		try:
			results_ids.append(process_file_ray_helper.remote(model,sim_file_path,write_file_path))
		except:
			print('Issue with data processing file: '+sim_file_path)

	results = ray.get(results_ids)

if __name__ == '__main__':

	####### Single lane: #######

	ray.init(num_cpus=4)

	model = get_cnn_lstm_ae_model(n_features=4)
	MODEL_PATH = '/Users/vanderbilt/Desktop/General_research_tools/Anti-Flow/detector_dev/models/cnn_lstm_ae_ringlength600_1lane__1.0percentGPS.pt'
	model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
	print('Detection models loaded.')


	want_process_single_lane_max_velocity = False
	if(want_process_single_lane_max_velocity):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_parameter_sweep'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_parameter_sweep_rec_errors'

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


	want_process_single_lane_radar_warp = False
	if(want_process_single_lane_radar_warp):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_parameter_sweep'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_parameter_sweep_rec_errors'

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


	####### Double lane: #######


	model = get_cnn_lstm_ae_model(n_features=4)
	MODEL_PATH = '/Users/vanderbilt/Desktop/General_research_tools/Anti-Flow/detector_dev/models/cnn_lstm_ae_double_lane_ring_length600.pt'
	model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))
	print('Detection models loaded.')



	want_process_double_lane_max_velocity = True
	if(want_process_double_lane_max_velocity):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/double_lane_ring_road_attack_parameter_sweep'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/double_lane_ring_road_attack_parameter_sweep_rec_errors'

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


	want_process_double_lane_radar_warp = True
	if(want_process_double_lane_radar_warp):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/double_lane_ring_road_attack_parameter_sweep'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/double_lane_ring_road_attack_parameter_sweep_rec_errors'

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




