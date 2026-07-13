import numpy as np
import matplotlib.pyplot as plt
import os


def get_all_csv_paths_in_repo(repo_path):
	all_csv_file_paths = []

	for file in os.listdir(repo_path):
		if('csv' in file):
			file_path = os.path.join(repo_path,file)
			all_csv_file_paths.append(file_path)

	return all_csv_file_paths



def fix_single_to_double(file_path):
	# NOTE: I found a naming convention error where some double lane sims were falsely called single

	fixed_file_path = ''
	if('single' in file_path):
		k = file_path.find('single')
		fixed_file_path = file_path[:k]+'double'+file_path[k+6:]

		os.rename(file_path,fixed_file_path)
	else:
		print('File does not need to be fixed: '+file_path)


def fix_all_files_in_repo(repo_path):
	all_csv_file_paths = get_all_csv_paths_in_repo(repo_path)
	for file_path in all_csv_file_paths:
		fix_single_to_double(file_path)
	print('Renamed all files in '+repo_path)



def get_rec_error_dict_from_file(file_path):
	rec_errors_list = np.loadtxt(file_path,delimiter=',',dtype=str)
	rec_errors_dict = {}

	for i in range(len(rec_errors_list)):
		veh_id = rec_errors_list[i,0]
		RE = float(rec_errors_list[i,1])

		if(veh_id in rec_errors_dict.keys()):
			rec_errors_dict[veh_id].append(RE)
		else:
			rec_errors_dict[veh_id] = []
			rec_errors_dict[veh_id].append(RE)

	for veh_id in rec_errors_dict:
		rec_errors_dict[veh_id] = np.array(rec_errors_dict[veh_id])

	return rec_errors_dict


def write_max_per_vehicle_RE_to_file(rec_errors_dict,file_path):
	MpV_RE_values = []
	for veh_id in rec_errors_dict:
		MpV_RE_values.append([veh_id,str(np.max(rec_errors_dict[veh_id]))])

	np.savetxt(file_path,np.array(MpV_RE_values),delimiter=',',fmt="%s")
	print('File written: '+file_path)

def get_MpV_RE_to_file(rec_error_file_path,write_file_path):
	rec_errors_dict = get_rec_error_dict_from_file(rec_error_file_path)
	write_max_per_vehicle_RE_to_file(rec_errors_dict,write_file_path)


def get_MpVRE_for_all_files_in_repo(rec_error_repo_path):
	MpV_RE_repo_path = os.path.join(rec_error_repo_path,'max_per_vehcicle_rec_errors')


	# check if relevant repository already exists:
	all_files_in_repo = os.listdir(rec_error_repo_path)
	if(not 'max_per_vehcicle_rec_errors' in all_files_in_repo):
		os.mkdir(MpV_RE_repo_path)

	rec_error_file_paths = []
	write_file_paths = []

	all_files_in_repo = os.listdir(rec_error_repo_path)

	for file in all_files_in_repo:
		if('.csv' in file):
			rec_error_file_paths.append(os.path.join(rec_error_repo_path,file))
			write_file_paths.append(os.path.join(MpV_RE_repo_path,file))

	for i in range(len(rec_error_file_paths)):
		get_MpV_RE_to_file(rec_error_file_paths[i],write_file_paths[i])

	print('Finished writing MpV_RE values to file.')







if __name__ == '__main__':


	rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_parameter_sweep_rec_errors'
	get_MpVRE_for_all_files_in_repo(rec_error_repo_path)

	rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/double_lane_ring_road_attack_parameter_sweep_rec_errors'
	get_MpVRE_for_all_files_in_repo(rec_error_repo_path)


	rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_parameter_sweep_rec_errors'
	get_MpVRE_for_all_files_in_repo(rec_error_repo_path)

	rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/double_lane_ring_road_attack_parameter_sweep_rec_errors'
	get_MpVRE_for_all_files_in_repo(rec_error_repo_path)

	print('Finished writing all MpVRE values.')

















