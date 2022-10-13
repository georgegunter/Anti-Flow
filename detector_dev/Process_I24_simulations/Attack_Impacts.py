import numpy as np
import os
import matplotlib.pyplot as pt
from i24_utils import get_losses_csv,get_sim_timeseries_all_data,get_sim_timeseries
from i24_utils import get_sim_name,get_attack_params
import ray

def get_ave_speed(timeseries_dict):
	mean_speed_per_commuter = []
	for veh_id in timeseries_dict:
		veh_data = timeseries_dict[veh_id]
		speed = veh_data[:,1]
		if(len(speed)>0):
			mean_speed_per_commuter.append(np.mean(speed))
	return np.mean(mean_speed_per_commuter)

def get_std_dev_speed(timeseries_dict):
	std_dev_speed_per_commuter = []
	for veh_id in timeseries_dict:
		veh_data = timeseries_dict[veh_id]
		speed = veh_data[:,1]
		if(len(speed)>0):
			std_dev_speed_per_commuter.append(np.std(speed))
	return np.mean(std_dev_speed_per_commuter)

def get_sim_impact_data(csv_path,warmup_period=1200.0):
	timeseries_dict = get_sim_timeseries(csv_path,warmup_period=warmup_period)
	ave_speed = get_ave_speed(timeseries_dict)
	std_dev_speed = get_std_dev_speed(timeseries_dict)
	sim_name = get_sim_name(csv_path)
	attack_params = get_attack_params(csv_path)
	data = [attack_params[0],attack_params[1],ave_speed,std_dev_speed]
	print(data)
	return data

@ray.remote
def get_sim_impact_data_ray(csv_path,warmup_period=1200.0):
	return get_sim_impact_data(csv_path,warmup_period=warmup_period)

def get_all_sim_impact_data_with_ray(sim_data_file_paths,warmup_period=1200.0):
	attack_impact_data_ray_ids = []
	for csv_path in sim_data_file_paths:
		attack_impact_data_ray_ids.append(get_sim_impact_data_ray.remote(csv_path,warmup_period=warmup_period))
	attack_impact_data = ray.get(attack_impact_data_ray_ids)
	return attack_impact_data

if __name__ == '__main__':
	sim_data_repo_path = '/Volumes/My Passport for Mac/i24_random_sample/simulations'
	all_sim_files = os.listdir(sim_data_repo_path)
	sim_data_file_paths = []
	for file in all_sim_files:
		sim_data_file_paths.append(os.path.join(sim_data_repo_path,file))











