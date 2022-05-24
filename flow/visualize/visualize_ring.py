# import numpy as np
import matplotlib.pyplot as pt
import sys
import numpy as np
import time
from copy import deepcopy

import csv

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth


def get_sim_timeseries(csv_path,warmup_period=0.0):
	row_num = 1
	curr_veh_id = 'id'
	sim_dict = {}
	curr_veh_data = []

	with open(csv_path, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if(row_num > 1):
				# Don't read header
				if(curr_veh_id != row[1]):
					#Add in new data to the dictionary:
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
					#Rest where data is being stashed:
					curr_veh_data = []
					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				if(time > warmup_period):
					# data = [time,speed,headway,leader_rel_speed]
					data = [row[0],row[4],row[5],row[8]]
					curr_veh_data.append(data)
			row_num += 1

		#Add the very last vehicle's information:
		sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
		print('Data loaded.')
	return sim_dict


def get_sim_timeseries_all_data(csv_path,warmup_period=0.0):
	row_num = 1
	curr_veh_id = 'id'
	sim_dict = {}
	curr_veh_data = []

	with open(csv_path, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if(row_num > 1):
				# Don't read header
				if(curr_veh_id != row[1]):
					#Add in new data to the dictionary:
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = curr_veh_data
					#Rest where data is being stashed:
					curr_veh_data = []
					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				if(time > warmup_period):
					# data = [time,speed,headway,leader_rel_speed]
					curr_veh_data.append(row)
			row_num += 1

		#Add the very last vehicle's information:
		sim_dict[curr_veh_id] = curr_veh_data
		print('Data loaded.')
	return sim_dict



def get_sim_data_dict_ring(csv_path,warmup_period=50.0):
	row_num = 1
	curr_veh_id = 'id'
	sim_dict = {}
	curr_veh_data = []

	with open(csv_path, newline='') as csvfile:
		csvreader = csv.reader(csvfile, delimiter=',')
		for row in csvreader:
			if(row_num > 1):
				# Don't read header
				if(curr_veh_id != row[1]):
					#Add in new data to the dictionary:
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = curr_veh_data
					#Rest where data is being stashed:
					curr_veh_data = []
					curr_veh_id = row[1] # Set new veh id
					#Allocate space for storing:
					sim_dict[curr_veh_id] = []

				curr_veh_id = row[1]
				time = float(row[0])
				if(time > warmup_period):
					curr_veh_data.append(row)
				# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
			row_num += 1

		#Add the very last vehicle's information:
		sim_dict[curr_veh_id] = curr_veh_data
		# sys.stdout.write('\r'+'Veh id: '+curr_veh_id+ ' row: ' +str(row_num)+'\r')
		print('Data loaded.')
	return sim_dict	

def get_ring_positions(sim_data_dict,ring_length):
	veh_ids = list(sim_data_dict.keys())
	ring_positions = dict.fromkeys(veh_ids)

	edge_length = ring_length/4.0

	for veh_id in veh_ids:
		temp_veh_data = np.array(sim_data_dict[veh_id])

		init_edge = temp_veh_data[0,-9]
		init_rel_position = temp_veh_data[0,-6].astype(float)

		distances = temp_veh_data[:,-7].astype(float)

		distances = distances - distances[0]

		init_dist = 0

		# Find initial distance along all edges:
		if(init_edge=='right'):
			init_dist = init_rel_position
		elif(init_edge=='top'):
			init_dist = init_rel_position + edge_length
		elif(init_edge=='left'):
			init_dist = init_rel_position + 2*edge_length
		elif(init_edge=='bottom'):
			init_dist = init_rel_position + 3*edge_length

		distances = distances + init_dist

		ring_positions[veh_id] = distances

	return ring_positions

def stack_data_for_spacetime(sim_data_dict,
	ring_positions,
	want_losses=False,
	losses_dict=None):

	veh_ids = list(sim_data_dict.keys())

	times_list = [] 
	pos_list = []
	speed_list = []

	for veh_id in veh_ids:
		#Look only at losses for those vehicles being measured:

		temp_veh_data = np.array(sim_data_dict[veh_id]) 
		time = temp_veh_data[:,0].astype(float) 
		ring_pos = ring_positions[veh_id]
		speed = temp_veh_data[:,4].astype(float) 

		for i in range(len(time)): 
			times_list.append(time[i]) 
			pos_list.append(ring_pos[i]) 
			speed_list.append(speed[i])

	if(want_losses):
		loss_list = []
		for veh_id in veh_ids:
			temp_veh_data = np.array(sim_data_dict[veh_id])
			loss = losses_dict[veh_id]
			for i in range(len(loss)):
				loss_list.append(loss[i])

		return np.array(times_list),np.array(pos_list),np.array(speed_list),np.array(loss_list)

	else:
		return np.array(times_list),np.array(pos_list),np.array(speed_list)

def make_ring_spacetime_fig(sim_data_dict=None,csv_path=None,ring_length=300):
	if(sim_data_dict is None):
		sim_data_dict = get_sim_data_dict_ring(csv_path)

	veh_ids = list(sim_data_dict.keys())
	ring_positions = get_ring_positions(sim_data_dict,ring_length)
	times,positions,speeds = stack_data_for_spacetime(sim_data_dict,ring_positions)

	positions_mod_ring_length = np.mod(positions,ring_length)

	fontsize=15
	pt.figure(figsize=[15,7])
	pt.title('Space time plot, ring length: '+str(ring_length),fontsize=fontsize)
	pt.scatter(times,positions_mod_ring_length,c=speeds)
	pt.ylabel('Position [m]',fontsize=fontsize)
	pt.xlabel('Time [s]',fontsize=fontsize)
	cbar = pt.colorbar(label='Speed [m/s]')
	cbar.ax.tick_params(labelsize=10)
	pt.show()

def make_ring_spacetime_fig_multilane(sim_data_dict=None,csv_path=None,ring_length=300,num_lanes=2):
	# DO THIS


	return None

def make_ring_spacetime_fig_with_losses(losses_smoothed,
	sim_data_dict=None,
	csv_path=None,
	ring_length=300):

	if(sim_data_dict is None):
		sim_data_dict = get_sim_data_dict_ring(csv_path)

	veh_ids = list(sim_data_dict.keys())
	ring_positions = get_ring_positions(sim_data_dict,ring_length)
	
	# Need to get losses:
	times,positions,speeds,losses = stack_data_for_spacetime(sim_data_dict,ring_positions,want_losses=True,losses_dict=losses_smoothed)

	positions_mod_ring_length = np.mod(positions,ring_length)

	fontsize=15
	pt.figure(figsize=[15,7])
	pt.title('Space time plot, ring length: '+str(ring_length),fontsize=fontsize)
	pt.scatter(times,positions_mod_ring_length,c=losses,s=3.0)
	pt.ylabel('Position [m]',fontsize=fontsize)
	pt.xlabel('Time [s]',fontsize=fontsize)
	cbar = pt.colorbar(label='Anomaly loss score')
	cbar.ax.tick_params(labelsize=10)
	pt.show()

# For when doing partial measurement via GPS:

'''
ring_sim_dict should come from get_sim_data_dict_ring(), veh_id_curr is a
current vehicle for which we want info on it's effective followers,
all_vehicle_ids_measured is a list of veh_ids that are observed.

'''


# The following three methods are depreciated:
def get_measured_leader(ring_sim_dict,veh_id_curr,all_vehicle_ids_measured):
	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	while(curr_leader not in(all_vehicle_ids_measured)):
		curr_leader = ring_sim_dict[curr_leader][0][6]
	return curr_leader

def get_rel_dist_to_measured_leader(ring_sim_dict,veh_id_curr,measured_leader):

	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	temp_data = np.array(ring_sim_dict[veh_id_curr])
	total_spacing = temp_data[:,5].astype('float')
	while(curr_leader != measured_leader):
		temp_data = np.array(ring_sim_dict[curr_leader])
		next_spacing = temp_data[:,5].astype('float')
		total_spacing += next_spacing
		curr_leader = ring_sim_dict[curr_leader][0][6]
	return total_spacing

def get_vel_of_measured_leader(ring_sim_dict,veh_id_curr,measured_leader):

	curr_leader = ring_sim_dict[veh_id_curr][0][6]
	temp_data = np.array(ring_sim_dict[veh_id_curr])
	while(curr_leader != measured_leader):
		temp_data = np.array(ring_sim_dict[curr_leader])
		curr_leader = ring_sim_dict[curr_leader][0][6]
	temp_data = np.array(ring_sim_dict[curr_leader])
	effective_leader_speed = temp_data[:,4].astype('float')
	return effective_leader_speed




def get_smoothed_losses(ae_model,emission_path,timeseries_dict=None,n_features=1):
    
    if(timeseries_dict is None):
        timeseries_dict = visualize_ring.get_sim_timeseries(csv_path = emission_path)
    
    i = 0

    losses = []
    for veh_id in timeseries_dict:
        speed = timeseries_dict[veh_id][:,1]
        p,l = sliding_window(ae_model,speed)
        losses.append(l)
        i += 1
        sys.stdout.write('\r'+'Vehicle, '+str(i))

    #veh ids for all vehicles in the flow:
    veh_ids = list(timeseries_dict.keys())
    #A dictionary which holds smoothed losses for vehicles:
    smoothed_losses = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][:,0]
    #Get smoothed loss values:
    for i in range(len(losses)):
        loss = losses[i]
        smoothed_losses[veh_ids[i]] =  loss_smooth(time,loss)
    
    print('\nReturning smoothed losses.')
    
    return smoothed_losses




