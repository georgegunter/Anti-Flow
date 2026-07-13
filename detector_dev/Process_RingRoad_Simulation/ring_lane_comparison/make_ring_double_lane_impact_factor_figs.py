import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import csv

from Data_Processing.sim_processing_utils import get_trajectory_timeseries


from Data_Processing.impact_factor_calculation_utils import get_mean_traffic_speed_ring,get_min_TTC_per_vehicle_ring, get_min_TimeGap_per_vehicle_ring

import ray


WARMUP_TIME = 100.0



def get_sim_name(emission_file_path):
	num_chars = len(emission_file_path)
	begin_file_name = 0
	j = 0
	while(j<num_chars):
		if(emission_file_path[j] == '/'):
			begin_file_name = j
		j += 1

	return emission_file_path[begin_file_name+1:]



def get_MTS_and_MTTC(trajectory_dict):
	mean_traffic_speed = get_mean_traffic_speed_ring(trajectory_dict=trajectory_dict)
	min_TTC = get_min_TTC_per_vehicle_ring(trajectory_dict=trajectory_dict)
	min_TTC = np.min(min_TTC)

	return [mean_traffic_speed,min_TTC]



def get_MTS_and_MTG(trajectory_dict):
	mean_traffic_speed = get_mean_traffic_speed_ring(trajectory_dict=trajectory_dict)
	min_TGs = get_min_TimeGap_per_vehicle_ring(trajectory_dict=trajectory_dict)
	mean_min_TG = np.mean(min_TGs)

	return [mean_traffic_speed,mean_min_TG]





# @ray.remote
# def get_relevant_data_ray(file_path):
# 	trajectory_dict = get_trajectory_timeseries(file_path,warmup_period=WARMUP_TIME,want_print_finished_loading=False)
# 	mean_traffic_speed,min_TTC = get_MTS_and_MTTC(trajectory_dict)
# 	sim_name = get_sim_name(file_path)
# 	print(sim_name+' : '+str(mean_traffic_speed)+', '+str(min_TTC))

# 	return [sim_name,mean_traffic_speed,min_TTC]


@ray.remote
def get_relevant_data_ray(file_path):
	trajectory_dict = get_trajectory_timeseries(file_path,warmup_period=WARMUP_TIME,want_print_finished_loading=False)

	mean_traffic_speed = get_mean_traffic_speed_ring(trajectory_dict=trajectory_dict)
	min_TTCs = get_min_TTC_per_vehicle_ring(trajectory_dict=trajectory_dict)
	min_TGs = get_min_TimeGap_per_vehicle_ring(trajectory_dict=trajectory_dict)

	sim_name = get_sim_name(file_path)
	print(sim_name+' : '+str(mean_traffic_speed)+', '+str(np.mean(min_TGs))+', '+str(np.min(min_TGs))+', '+str(np.mean(min_TTCs)) +', '+str(np.min(min_TTCs)))

	return [sim_name,mean_traffic_speed,np.mean(min_TGs),np.min(min_TGs),np.mean(min_TTCs),np.min(min_TTCs)]





if __name__ == '__main__':


	# NOTE: need to refactor to not include attacking vehicles in calculations.


	sim_repo_path = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'



	all_files_in_sim_repo = os.listdir(sim_repo_path)

	sim_files = []

	for file in all_files_in_sim_repo:
		if(('ver' in file) and ('csv' in file)):
			sim_files.append(os.path.join(sim_repo_path,file))


	ray.init(num_cpus = 3)

	results_ids = []

	for file_path in sim_files:
		try:
			results_ids.append(get_relevant_data_ray.remote(file_path))
		except:
			print('Issue with data processing file '+file_path)

	impact_results = ray.get(results_ids)

	with open('performance_metrics_double_lane.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for impact in impacts_list:
        writer.writerow(impact)




