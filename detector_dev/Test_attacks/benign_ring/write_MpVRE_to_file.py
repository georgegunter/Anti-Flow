import numpy as np
import matplotlib.pyplot as plt
import os

from detector_dev.Process_I24_simulations.utils_classification import *


FIGURE_SIZE = [35,10]
TITLE_FONTSIZE = 40
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 35
MARKERSIZE = 30

def get_all_csv_files_in_repo(repo_path):

	all_files_in_repo = os.listdir(repo_path)

	csv_files = []

	for file in all_files_in_repo:
		if('csv' in file):
			csv_files.append(os.path.join(repo_path,file))

	return csv_files


def get_sim_name(file_path):
	i = 0
	while(file_path[i:i+3] != 'ver'): i+=1

	return file_path[i:]

def write_MpVRE(file_path,write_repo):

	rec_error_list = np.loadtxt(file_path,delimiter=',',dtype=str)
	max_rec_errors = {}
	for i in range(len(rec_error_list)):
		veh_id = rec_error_list[i][0]
		rec_error_val = float(rec_error_list[i][1])
		if veh_id not in max_rec_errors.keys():
			max_rec_errors[veh_id] = rec_error_val
		else:
			if(rec_error_val > max_rec_errors[veh_id]):
				max_rec_errors[veh_id] = rec_error_val

	MpVRE_list = []

	for veh_id in max_rec_errors:
		MpVRE_list.append([veh_id,str(max_rec_errors[veh_id])])

	file_write_path = os.path.join(write_repo,get_sim_name(file_path))





def write_MpVRE_all_files_in_repo(repo_path,write_repo):
	all_file_paths = get_all_csv_files_in_repo(rec_error_repo_path)

	for file_path in all_file_paths:
		write_MpVRE(file_path,write_repo)

	print('Finished writing MpVREs in repository.')








if __name__ == '__main__':

	####################
	### SINGLE LANE: ###
	####################

	rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/benign_single_lane_ring/rec_errors'

	write_repo = os.path.join(rec_error_repo_path,'/max_per_vehicle_rec_errors')


