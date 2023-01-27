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





def get_max_rec_errors(rec_errors_results_path,sim_results_path,warmup_period=1200):
	timeseries_dict = get_sim_timeseries(csv_path=sim_results_path,warmup_period=warmup_period)
	rec_error_results = get_losses_csv(file_path=rec_errors_results_path)
	veh_types = get_vehicle_types(sim_results_path)

	max_rec_errors_benign = []
	max_rec_errors_attack = []

	for veh_id in rec_error_results:
		rec_errors = rec_error_results[veh_id][:,1]
		if('attack' in veh_types[veh_id]): max_rec_errors_attack.append(np.max(rec_errors))
		else: max_rec_errors_benign.append(np.max(rec_errors))

	return max_rec_errors_benign,max_rec_errors_attack


def write_max_rec_errors_to_file(path_to_write_file,max_rec_errors_attack,max_rec_errors_benign):

	with open(path_to_write_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for max_rec_error in max_rec_errors_attack:
			writer.writerow([1,max_rec_error])
		for max_rec_error in max_rec_errors_benign:
			writer.writerow([0,max_rec_error])

	print('Max rec errors written to file.')


def process_file(rec_errors_results_path,
	sim_results_path,
	file_name,
	classifiction_results_repo):

	max_rec_errors_benign,max_rec_errors_attack = get_max_rec_errors(rec_errors_results_path,sim_results_path)


	max_rec_erroc_file_name = 'max_rec_errors_'+file_name 

	path_to_write_file = os.path.join(classifiction_results_repo,max_rec_erroc_file_name)


	write_max_rec_errors_to_file(path_to_write_file,
					max_rec_errors_attack,
					max_rec_errors_benign)


	return path_to_write_file


@ray.remote
def process_file_ray(rec_errors_results_path,
	sim_results_path,
	file_name,
	classifiction_results_repo):
    return process_file(rec_errors_results_path,
	sim_results_path,
	file_name,
	classifiction_results_repo)


if __name__ == '__main__':

	rec_error_results_repo = '/Volumes/My Passport for Mac/i24_random_sample/ae_rec_error_results/1800_inflow_normalized'
	sim_results_repo = '/Volumes/My Passport for Mac/i24_random_sample/simulations/1800_inflow'

	classifiction_results_repo = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/Process_I24_simulations/normalized_detection_classification_results'

	max_rec_error_repo = os.path.join(classifiction_results_repo,'max_rec_errors')


	attack_file_names = []
	files_list = os.listdir(rec_error_results_repo)
	for file_name in files_list:
		if(file_name[-3:] == 'csv'):
			attack_file_names.append(file_name)


	all_rec_error_results_files = []
	files_list = os.listdir(rec_error_results_repo)
	for file_name in files_list:
		if(file_name[-3:] == 'csv'):
			file_path = os.path.join(rec_error_results_repo,file_name)
			all_rec_error_results_files.append(file_path)


	all_sim_results_files = []
	files_list = os.listdir(sim_results_repo)
	for file_name in files_list:
		if(file_name[-3:] == 'csv'):
			file_path = os.path.join(sim_results_repo,file_name)
			all_sim_results_files.append(file_path)



	ray.init()

	max_rec_error_file_path_ids = []


	for file_name in attack_file_names:
		rec_error_results_file_path = None
		sim_results_file_path = None

		# make sure find the simulation result properly
		j = 0
		found_sim_result = False
		sim_result_not_in_repo = False

		while(not found_sim_result and not sim_result_not_in_repo):
			if(not file_name in all_sim_results_files[j]): j+=1
			else:found_sim_result = True
			if j>=50:sim_result_not_in_repo = True

		if(found_sim_result):
			sim_results_path = all_sim_results_files[j]

			k = 0 
			found_rec_error_result = False
			rec_error_result_not_in_repo = False

			while(not found_rec_error_result and not rec_error_result_not_in_repo):
				if(not file_name in all_rec_error_results_files[k]): k+=1
				else:found_rec_error_result = True
				if k>=50:rec_error_result_not_in_repo = True

			if(found_rec_error_result):
				rec_errors_results_path = all_rec_error_results_files[k]

				

				max_rec_error_file_path_ids.append(
					process_file_ray.remote(
						rec_errors_results_path,
						sim_results_path,
						file_name,
						classifiction_results_repo))

				print('Wrote max rec errors to file: '+file_name)

			else:print('Could not locate reconstruction error results file for file: '+str(file_name))


		else:print('Could not locate simulation results file for file: '+str(file_name))


	max_rec_error_ray_results = ray.get(max_rec_error_file_path_ids)

	print('Finished finding max rec errors.')






