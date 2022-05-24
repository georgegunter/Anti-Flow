from flow.visualize.visualize_ring import get_sim_timeseries_all_data
import numpy as np

import os

import ray

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

	for file in emission_files:
		if(file[-3:] == 'csv'):
			emission_path = os.path.join(emission_repo,file)

			impact_metric_ids.append(get_attack_impact_metrics_ray.remote(emission_path))

	impact_metric_list = ray.get(impact_metric_ids)




	impact_dict = {}

	[TAD,ADR] = get_attack_params(emission_path)

	key =  str(TAD) + '_'+str(ADR)

	if(key in impact_dict.keys()):

	else:
		impact_dict[key] = []


if __name__ == '__main__':
	emission_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'









