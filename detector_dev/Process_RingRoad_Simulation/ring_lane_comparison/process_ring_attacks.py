from flow.visualize.visualize_ring import get_sim_timeseries_all_data
import numpy as np

import os

import ray


import csv


def get_attack_impact_metrics(emission_path):
	timeseries_dict = get_sim_timeseries_all_data(emission_path,warmup_period=100)

	average_traffic_speed = 0.0
	traffic_speed_variance = 0.0

	total_time_in_danger = 0.0

	all_speeds = []
	all_spacings = []

	total_lane_changes = 0

	for veh_id in timeseries_dict:
		data = timeseries_dict[veh_id]
		speeds = []
		spacings = []

		dt = float(data[1][0])- float(data[0][0])

		prev_lane = data[0][-3]

		for i in range(len(data)):
			v = float(data[i][4])
			s = float(data[i][5])

			t_g = s/(v + .001) #avoid divide by zero

			if(t_g < 1.0): total_time_in_danger += dt

			speeds.append(v)
			spacings.append(s)

			if(prev_lane != data[i][-3]):
				total_lane_changes += 1

			prev_lane = data[i][-3]

		average_traffic_speed += np.mean(speeds)
		traffic_speed_variance += np.var(speeds)


	average_traffic_speed = average_traffic_speed/len(timeseries_dict)
	traffic_speed_variance = traffic_speed_variance/len(timeseries_dict)

	return [average_traffic_speed,traffic_speed_variance,total_time_in_danger,total_lane_changes]


def get_attack_params(emission_path):
	num_chars = len(emission_path)

	i = 0
	while(emission_path[i:i+3] != 'TAD' and i < num_chars):i += 1
	i += 4
	j = i
	while(emission_path[j] != '_' and j < num_chars): j+= 1

	TAD = float(emission_path[i:j])

	i=j
	while(emission_path[i:i+3] != 'ADR' and i < num_chars):i += 1
	i += 4

	j = i
	while(emission_path[j] != '_' and j < num_chars): j+= 1

	ADR = float(emission_path[i:j])

	return [TAD,ADR]

@ray.remote
def get_attack_impact_metrics_ray(emission_path):
	return [emission_path,get_attack_impact_metrics(emission_path)]


def get_attack_impact_metrics_list_ray(emission_repo):
	emission_files = os.listdir(emission_repo)

	impact_metric_ids = []	

	for file in emission_files:
		if(file[-3:] == 'csv'):
			emission_path = os.path.join(emission_repo,file)

			impact_metric_ids.append(get_attack_impact_metrics_ray.remote(emission_path))

	impact_metric_list = ray.get(impact_metric_ids)

	return impact_metric_list


def get_attack_impact_dict_all_ray(emission_repo):

	emission_files = os.listdir(emission_repo)

	impact_metric_ids = []	

	num_files = len(emission_files)

	files_processed = 0

	for file in emission_files:
		if(file[-3:] == 'csv'):
			emission_path = os.path.join(emission_repo,file)

			impact_metric_ids.append(get_attack_impact_metrics_ray.remote(emission_path))

			files_processed += 1

			print('Files processed: '+str(files_processed)+'/'+str(num_files))

	impact_metric_list = ray.get(impact_metric_ids)

	impact_dict = {}

	[TAD,ADR] = get_attack_params(emission_path)

	key =  str(TAD) + '_'+str(ADR)

	if(key in impact_dict.keys()):
		impact_dict[key].append(impact_metric_list)

	else:
		impact_dict[key] = []
		impact_dict[key].append(impact_metric_list)

	return impact_dict, impact_metric_list


def write_impact_metric_list(impact_metric_list,file_name='all_impacts.csv'):
	with open(file_name, 'w', newline='') as csvfile:
	    file_writer = csv.writer(csvfile, delimiter=',')

	    num_samples = len(impact_metric_list)

	    for i in range(num_samples):
	    	row = []
	    	row.append(impact_metric_list[i][0])
	    	for j in range(len(impact_metric_list[i][1])):
	    		row.append(impact_metric_list[i][1][j])


	    	file_writer.writerow(row)

	print('Finished writing.')



if __name__ == '__main__':
	emission_repo_double_lane = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'

	ray.init()

	double_lane_impact_results_dict = get_attack_impact_dict_all_ray(emission_repo)


	emission_repo_single_lane = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep'

	impact_dict_single_lane,impact_metrics_list_single_lane = get_attack_impact_dict_all_ray(emission_repo_single_lane)

	file_name_single_lane = os.path.join(emission_repo_single_lane,'all_impact_metrics_single_lane.csv')

	write_impact_metric_list(impact_metrics_list_single_lane,file_name_single_lane)








