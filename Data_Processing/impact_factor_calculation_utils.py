import numpy as np
import matplotlib.pyplot as plt


#related to I24:
from detector_dev.Process_I24_simulations.i24_utils import get_sim_timeseries as get_sim_timeseries_i24
from detector_dev.Process_I24_simulations.i24_utils import get_attack_params as get_attack_params_i24


#related to ring road:
from flow.visualize.visualize_ring import get_sim_timeseries as get_sim_timeseries_ring
from detector_dev.Process_RingRoad_Simulation.process_ring_attacks import get_attack_params as get_attack_params_ring


from Data_Processing.sim_processing_utils import get_trajectory_timeseries



import os
import ray
import csv




def make_timeseries_list(trajectory_dict):
    timeseries_list = []
    for veh_id in trajectory_dict:
        trajectory_samples = []
        trajectory_data = trajectory_dict[veh_id]
        
        speed = trajectory_data[:,1]
        accel = np.gradient(speed,.1)
        head_way = trajectory_data[:,2]
        rel_vel = trajectory_data[:,3]
        
        trajectory_samples.append(speed)
        trajectory_samples.append(accel)
        trajectory_samples.append(head_way)
        trajectory_samples.append(rel_vel)
        
        timeseries_list.append(trajectory_samples)
    return timeseries_list







## related to processing the ring-road:
def get_mean_speeds_ring(emission_path=None,trajectory_dict=None):
	
	if(trajectory_dict is None):
		trajectory_dict = get_trajectory_timeseries(emission_path)


	veh_ids = list(trajectory_dict.keys())

	num_attacking_vehicles = 0

	for veh_id in veh_ids:
		if('adv' in veh_id): num_attacking_vehicles += 1


	num_samples = len(trajectory_dict[veh_ids[0]][:,1])

	num_benign_vehicles = len(veh_ids) - num_attacking_vehicles

	average_traffic_speeds = np.zeros(num_samples,)

	for veh_id in trajectory_dict:
		if('adv' not in veh_id):
			speeds = trajectory_dict[veh_id][:,1]
			average_traffic_speeds = average_traffic_speeds + speeds.reshape(num_samples,)

	average_traffic_speeds = average_traffic_speeds/num_benign_vehicles

	return average_traffic_speeds


def get_mean_traffic_speed_ring(emission_path=None,trajectory_dict=None):
	mean_traffic_speed_list = get_mean_speeds_ring(emission_path,trajectory_dict)
	return np.mean(mean_traffic_speed_list)




def get_mean_traffic_speed_variance_ring(emission_path=None,trajectory_dict=None):
	
	if(trajectory_dict is None):
		trajectory_dict = get_trajectory_timeseries(emission_path)


	veh_ids = list(trajectory_dict.keys())

	num_attacking_vehicles = 0

	for veh_id in veh_ids:
		if('adv' in veh_id): num_attacking_vehicles += 1


	num_samples = len(trajectory_dict[veh_ids[0]][:,1])

	num_benign_vehicles = len(veh_ids) - num_attacking_vehicles

	speed_variances = np.zeros(num_benign_vehicles,)

	i = 0

	for veh_id in trajectory_dict:
		if('adv' not in veh_id):
			speeds = trajectory_dict[veh_id][:,1]
			speed_variances[i] = np.var(speeds)
			i += 1

	return np.mean(speed_variances)


def get_min_TTC_per_vehicle_ring(emission_path=None,trajectory_dict=None):

	if(trajectory_dict is None):
		trajectory_dict = get_trajectory_timeseries(emission_path)

	veh_ids = list(trajectory_dict.keys())

	num_attacking_vehicles = 0

	for veh_id in veh_ids:
		if('adv' in veh_id): num_attacking_vehicles += 1


	num_samples = len(trajectory_dict[veh_ids[0]][:,1])

	num_benign_vehicles = len(veh_ids) - num_attacking_vehicles



	min_TTCs_per_vehicle = np.zeros(num_benign_vehicles,)


	i = 0

	for veh_id in trajectory_dict:

		if('adv' not in veh_id):

			speed_diff = trajectory_dict[veh_id][:,3]
			spacing = trajectory_dict[veh_ids[0]][:,2]

			TTC = np.divide(spacing,speed_diff)
			
			min_TTC = np.inf


			for t in range(len(TTC)):
				if((speed_diff[t] > 0.0) and (TTC[t] < min_TTC)):
					min_TTC = TTC[t]

			min_TTCs_per_vehicle[i] = min_TTC

			i += 1

	return min_TTCs_per_vehicle


def get_min_TimeGap_per_vehicle_ring(emission_path=None,trajectory_dict=None):

	if(trajectory_dict is None):
		trajectory_dict = get_trajectory_timeseries(emission_path)

	veh_ids = list(trajectory_dict.keys())

	num_attacking_vehicles = 0

	for veh_id in veh_ids:
		if('adv' in veh_id): num_attacking_vehicles += 1


	num_samples = len(trajectory_dict[veh_ids[0]][:,1])

	num_benign_vehicles = len(veh_ids) - num_attacking_vehicles



	min_TGs_per_vehicle = np.zeros(num_benign_vehicles,)


	i = 0

	for veh_id in trajectory_dict:

		if('adv' not in veh_id):

			speed = trajectory_dict[veh_id][:,1]
			spacing = trajectory_dict[veh_ids[0]][:,2]

			min_TGs_per_vehicle[i] = np.min(np.divide(spacing,speed))

			i += 1

	return min_TGs_per_vehicle


## related to processing i24:

# def get_mean_traffic_speed_i24(trajectory_dict,dt=0.1):
# 	total_time_samples = 0
# 	total_speed_sum = 0

# 	for veh_id in trajectory_dict:
# 		speeds = 







def get_speeds_by_time(timeseries_dict,dt=0.1):
    speeds_by_time = {}

    total_vehicles = len(timeseries_dict)

    num_veh_processed = 0

    for veh_id in timeseries_dict:
        times = timeseries_dict[veh_id][:,0]
        speeds = timeseries_dict[veh_id][:,1]
        for i in range(len(times)):
            t = times[i]
            t = np.round(t,1)
            v = speeds[i]

            if(t in speeds_by_time):
                speeds_by_time[t].append(v)
            else:
                speeds_by_time[t] = [v]

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'/'+str(total_vehicles)+'\r')

        num_veh_processed += 1

    return speeds_by_time

def get_speeds_by_time_from_csv(csv_path,dt=0.1):
    timeseries_dict = get_sim_timeseries(csv_path)
    speeds_by_time = get_speeds_by_time(timeseries_dict,dt=dt)
    return speeds_by_time












