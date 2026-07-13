import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import ray


def get_label_dict(file_path,warmup_period=0.0):
    row_num = 1
    curr_veh_id = 'id'
    label_dict = {}

    with open(file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        
        label_index = 2
        veh_id_index = 1

        row_num += 1

        for row in csvreader:
            if(row_num > 1):
                veh_id = row[veh_id_index]
                label = row[label_index]
                if(veh_id not in label_dict):
                	label_dict[veh_id] = label
            row_num += 1

    return label_dict


def get_all_csv_paths_in_repo(repo_path):
	all_csv_file_paths = []

	for file in os.listdir(repo_path):
		if('csv' in file):
			file_path = os.path.join(repo_path,file)
			all_csv_file_paths.append(file_path)

	return all_csv_file_paths



def parse_float_list(float_list_string):
	all_floats = []

	j = 1
	currently_on_float = False

	# while(i < len(float_list_string)-1):
	for i in range(1,len(float_list_string)-1):

		try:
			# will break if can't conver to 
			if(float_list_string[i] != '.'):
				curr_float = float(float_list_string[i])
				# if currently on float
				if(not currently_on_float):
					j = i
				currently_on_float = True
		except:
			if(currently_on_float):
				all_floats.append(float(float_list_string[j:i]))
			currently_on_float = False

	return all_floats

def get_rec_errors_from_csv(file_path):
    loss_dict = {}
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:

            veh_id = row[0]
            float_list = parse_float_list(row[1])

            if(veh_id not in loss_dict.keys()):
                loss_dict[veh_id] = []
                loss_dict[veh_id].append(float_list)
            else:
                loss_dict[veh_id].append(float_list)
    for veh_id in loss_dict:
        loss_dict[veh_id] = np.array(loss_dict[veh_id])
    return loss_dict


def write_rec_errors_to_csv(rec_errors_dict,file_name):
	rec_errors_list = []
	for veh_id in rec_errors_dict:
		t = rec_errors_dict[veh_id][0]
		RE = rec_errors_dict[veh_id][1]
		for i in range(np.min([len(t),len(RE)])):
			rec_errors_list.append([veh_id,t[i],RE[i]])


	with open(file_name, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for row in rec_errors_list:
				writer.writerow(row)

	print('Rec error csv written to file: '+str(file_name))


def write_max_per_vehicle_RE_to_file(rec_errors_dict,file_path):
	label_dict = get_label_dict(file_path)

	MpV_RE_values = []
	for veh_id in rec_errors_dict:
		RE_vals = rec_errors_dict[veh_id][1]
		label = label_dict[veh_id]
		MpV_RE_values.append([veh_id,str(np.max(RE_vals)),label_dict])

	np.savetxt(file_path,np.array(MpV_RE_values),delimiter=',',fmt="%s")
	print('MpV_RE values written to file: '+file_path)

def get_MpV_RE_to_file(rec_error_file_path,write_file_path,rec_error_rewrite_file_path):
	rec_errors_dict = get_rec_errors_from_csv(rec_error_file_path)
	# write_rec_errors_to_csv(rec_errors_dict,rec_error_rewrite_file_path)
	write_max_per_vehicle_RE_to_file(rec_errors_dict,write_file_path)



@ray.remote
def get_MpV_RE_to_file_ray(rec_error_file_path,write_file_path,rec_error_rewrite_file_path):
	return get_MpV_RE_to_file(rec_error_file_path,write_file_path,rec_error_rewrite_file_path)


def get_MpVRE_for_all_files_in_repo(rec_error_repo_path,rec_error_rewrite_repo_path):
	MpV_RE_repo_path = os.path.join(rec_error_repo_path,'max_per_vehicle_rec_errors')

	# check if relevant repository already exists:
	all_files_in_repo = os.listdir(rec_error_repo_path)
	if(not 'max_per_vehicle_rec_errors' in all_files_in_repo):
		os.mkdir(MpV_RE_repo_path)

	rec_error_file_paths = []

	rec_error_rewrite_file_paths = []

	write_file_paths = []

	all_files_in_repo = os.listdir(rec_error_repo_path)

	for file in all_files_in_repo:
		if('.csv' in file):
			rec_error_rewrite_file_paths.append(os.path.join(rec_error_rewrite_repo_path,file))
			rec_error_file_paths.append(os.path.join(rec_error_repo_path,file))
			write_file_paths.append(os.path.join(MpV_RE_repo_path,file))

	

	ray_res_ids = []

	for i in range(len(rec_error_file_paths)):

		ray_res_ids.append(get_MpV_RE_to_file_ray.remote(rec_error_file_paths[i],write_file_paths[i],rec_error_rewrite_file_paths[i]))

		# get_MpV_RE_to_file(rec_error_file_paths[i],write_file_paths[i],rec_error_rewrite_file_paths[i])

	results = ray.get(ray_res_ids)

	print('Finished writing MpV_RE values to file.')







if __name__ == '__main__':

	ray.init(num_cpus=4)

	if(True):
		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample_rec_errors'

		rec_error_rewrite_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample_rec_errors_original'


		# rec_error_rewrite_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample_rec_errors'

		# rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/i24_random_sample_rec_errors_original'
		

		get_MpVRE_for_all_files_in_repo(rec_error_repo_path,rec_error_rewrite_repo_path)


		print('Finished writing all MpVRE values.')


	if(False):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/i24_random_sample'

		rec_error_rewrite_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/i24_random_sample_rec_errors'

		rec_error_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/i24_random_sample_rec_errors_original'
		

		get_MpVRE_for_all_files_in_repo(rec_error_repo_path,rec_error_rewrite_repo_path)


		print('Finished writing all MpVRE values.')












