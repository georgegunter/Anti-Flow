import numpy as np
import csv
import os
import time
import ray

'''
Functions for loading in timeseries data:
'''
def get_trajectory_timeseries(csv_path,warmup_period=0.0):
    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    begin_time = time.time()

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        id_index = 0
        time_index = 0
        speed_index = 0
        headway_index = 0
        relvel_index = 0
        edge_index = 0
        pos_index = 0

        row1 = next(csvreader)
        num_entries = len(row1)
        while(row1[id_index]!='id' and id_index<num_entries):id_index +=1
        while(row1[edge_index]!='edge_id' and edge_index<num_entries):edge_index +=1
        while(row1[time_index]!='time' and time_index<num_entries):time_index +=1
        while(row1[speed_index]!='speed' and speed_index<num_entries):speed_index +=1
        while(row1[headway_index]!='headway' and headway_index<num_entries):headway_index +=1
        while(row1[relvel_index]!='leader_rel_speed' and relvel_index<num_entries):relvel_index +=1

        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[id_index]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>0):
                        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[id_index] # Set new veh id
                    #Allocate space for storing:
                    # sim_dict[curr_veh_id] = []

                curr_veh_id = row[id_index]
                sim_time = float(row[time_index])
                edge = row[edge_index]
                if(sim_time > warmup_period):
                    # data = [time,speed,headway,leader_rel_speed]

                    # Check what was filled in if missing a leader:
                    s = float(row[headway_index])
                    dv = float(row[relvel_index])
                    v = float(row[speed_index])
                    t = float(row[time_index])

                    data = [t,v,s,dv]
                    curr_veh_data.append(data)
            row_num += 1

        #Add the very last vehicle's information:
        sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
        end_time = time.time()
        print('Data loaded, total time: '+str(end_time-begin_time))
        

    return sim_dict





'''
Functions below for getting attack impacts:

'''

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






