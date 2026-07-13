import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import csv
from scipy.signal import savgol_filter
from Data_Processing.sim_processing_utils import get_trajectory_timeseries

from Data_Processing.impact_factor_calculation_utils import get_mean_traffic_speed_ring,get_min_TTC_per_vehicle_ring, get_min_TimeGap_per_vehicle_ring,get_mean_traffic_speed_variance_ring

from energy_models import PFM2019RAV4


import ray

import warnings
warnings.filterwarnings("ignore")


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



def get_average_energy(trajectory_dict):
	energy_consumption_model =  PFM2019RAV4()
	E_vals = []

	for veh_id in trajectory_dict:
		v = np.array(trajectory_dict[veh_id][:,1])
		dv_dt = np.gradient(savgol_filter(v,11,3),0.1)
		for i in range(len(v)):
			E_vals.append(energy_consumption_model.get_instantaneous_fuel_consumption(v[i],dv_dt[i],grade=0.0))

	return np.mean(E_vals)




def get_relevant_data(file_path):
	trajectory_dict = get_trajectory_timeseries(file_path,warmup_period=100.0,want_print_finished_loading=False)
	sim_name = get_sim_name(file_path)

	veh_ids = list(trajectory_dict.keys())
	for veh_id in veh_ids:
		v = np.array(trajectory_dict[veh_id][:,1])
		if(np.min(v) < -1.0):
			print('Collision present: '+sim_name)
			return [sim_name,np.inf,np.inf,np.inf]

	try:

		mean_traffic_speed = get_mean_traffic_speed_ring(trajectory_dict=trajectory_dict)

		mean_traffic_speed_variance = get_mean_traffic_speed_variance_ring(trajectory_dict=trajectory_dict)

		mean_energy_consumption = get_average_energy(trajectory_dict=trajectory_dict)

		sim_name = get_sim_name(file_path)
		print(sim_name+' : '+str(mean_traffic_speed)+', '+str(mean_traffic_speed_variance)+', '+str(mean_energy_consumption))
	except:
		print('Issue with data: '+sim_name)
		return [sim_name,np.nan,np.nan,np.nan]


	return [sim_name,mean_traffic_speed,mean_traffic_speed_variance,mean_energy_consumption]



@ray.remote
def get_relevant_data_ray(file_path):
	return get_relevant_data(file_path)


def get_num_sims_performed(sim_name,sim_files):
	num_sims_performed = 0
	for file in sim_files:
		if(sim_name in file): num_sims_performed+=1

	return num_sims_performed

def get_sim_name_no_version(sim_name):
	i=0
	while(sim_name[i:i+4] != '_ver'):i+=1
	return sim_name[:i]









if __name__ == '__main__':

	ray.init(num_cpus=4)

	want_process_max_velocity = False
	if(want_process_max_velocity):
		print('Processing max velocity ring road attack:')

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_monte_carlo'

		print(sim_repo_path)

		all_files_in_sim_repo = os.listdir(sim_repo_path)

		sim_files = []

		for file in all_files_in_sim_repo:
			if(('ver' in file) and ('csv' in file)):
				sim_files.append(os.path.join(sim_repo_path,file))

		results_ids = []

		for file_path in sim_files:
			try:
				results_ids.append(get_relevant_data_ray.remote(file_path))
			except:
				print('Issue with data processing file '+file_path)

		impact_results = ray.get(results_ids)

		with open('performance_metrics_max_velocity.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for impact in impact_results:
				writer.writerow(impact)


	want_process_radar_warp = True
	if(want_process_radar_warp):
		print('Processing radar warp ring road attack:')

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_monte_carlo'

		print(sim_repo_path)

		all_files_in_sim_repo = os.listdir(sim_repo_path)

		sim_files = []

		for file in all_files_in_sim_repo:
			if(('ver' in file) and ('csv' in file)):
				sim_files.append(os.path.join(sim_repo_path,file))

		results_ids = []

		for file_path in sim_files:
			try:
				results_ids.append(get_relevant_data_ray.remote(file_path))
			except:
				print('Issue with data processing file '+file_path)

		impact_results = ray.get(results_ids)

		with open('performance_metrics_radar_warp.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for impact in impact_results:
				writer.writerow(impact)


	want_process_RDA = True
	if(want_process_radar_warp):
		print('Processing RDA ring road attack:')

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo'

		print(sim_repo_path)

		all_files_in_sim_repo = os.listdir(sim_repo_path)

		sim_files = []

		for file in all_files_in_sim_repo:
			if(('ver' in file) and ('csv' in file)):
				sim_files.append(os.path.join(sim_repo_path,file))

		results_ids = []

		for file_path in sim_files:
			try:
				results_ids.append(get_relevant_data_ray.remote(file_path))
			except:
				print('Issue with data processing file '+file_path)

		impact_results = ray.get(results_ids)

		with open('performance_metrics_RDA.csv', 'w', newline='') as csvfile:
			writer = csv.writer(csvfile, delimiter=',')
			for impact in impact_results:
				writer.writerow(impact)


