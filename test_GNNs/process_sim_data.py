import numpy as np
import csv

def get_sim_timeseries_all_data(csv_path,warmup_period=0.0,edge_list=None):
	row_num = 1
	curr_veh_id = 'id'
	sim_dict = {}
	curr_veh_data = []

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


		row_num += 1

		for row in csvreader:
			if(row_num > 1):
				# Don't read header
				if(curr_veh_id != row[id_index]):
					#Add in new data to the dictionary:
					
					#Store old data:
					if(len(curr_veh_data)>0):
						sim_dict[curr_veh_id] = curr_veh_data
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
					curr_veh_data.append(row)
			row_num += 1

		#Add the very last vehicle's information:
		if(len(curr_veh_data) > 0):
			sim_dict[curr_veh_id] = curr_veh_data
		print('Data loaded.')
	return sim_dict

def gen_road_segments(col_num, row_num):
	"""Generate the names of the roads in the grid network and assign indices.

	Parameters
	----------
	col_num : int
		number of columns in the grid
	row_num : int
		number of rows in the grid

	Returns
	-------
	list of str
		names of all the outer roads
	"""
	roads = []
	road_indices = {}
	road_num = 0

	# build the left and then the right roads
	for i in range(col_num):
		for j in range(row_num+1):
			# left road:
			road = 'left' + str(j) + '_' + str(i)
			road_indices[road] = road_num
			road_num += 1
			roads += [road]
			# right road:
			road = 'right' + str(j) + '_' + str(i)
			road_indices[road] = road_num
			road_num += 1
			roads += [road]

	# build the bottom and then top roads
	for i in range(col_num+1):
		for j in range(row_num):
			road = 'bot' + str(j) + '_' + str(i)
			road_indices[road] = road_num
			road_num += 1
			roads += [road]
			# right road:
			road = 'top' + str(j) + '_' + str(i)
			road_indices[road] = road_num
			road_num += 1
			roads += [road]

	return roads,road_indices

def get_intersection_connections(i,j):
	connections = []

	outgoing_right = 'right' + str(j+1) + '_' + str(i)
	outgoing_left = 'left' + str(j) + '_' + str(i)
	outgoing_top = 'top' + str(j) + '_' + str(i)
	outgoing_bottom = 'bot' + str(j) + '_' + str(i+1)

	# bottom:
	road = 'bot' + str(j) + '_' + str(i)
	connections.append([road,outgoing_bottom])
	connections.append([road,outgoing_right])
	connections.append([road,outgoing_left])

	# top:
	road = 'top' + str(j) + '_' + str(i+1)
	connections.append([road,outgoing_bottom])
	connections.append([road,outgoing_right])
	connections.append([road,outgoing_left])

	#right:
	road = 'right' + str(j) + '_' + str(i)
	connections.append([road,outgoing_top])
	connections.append([road,outgoing_bottom])
	connections.append([road,outgoing_right])


	#left
	road = 'left' + str(j+1) + '_' + str(i)
	connections.append([road,outgoing_top])
	connections.append([road,outgoing_bottom])
	connections.append([road,outgoing_left])

	return connections

def get_road_segment_connections(road_indices,row_num,col_num):
	road_segment_connections = []

	for i in range(col_num):
		for j in range(row_num):
			intersection_connections = get_intersection_connections(i,j)
			for connection in intersection_connections:
				incoming = connection[0]
				outgoing = connection[1]
				road_segment_connections.append([road_indices[incoming],road_indices[outgoing]])

	road_segment_connections = np.array(road_segment_connections)

	return road_segment_connections.T

def get_road_segment_counts(roads,road_indices,sim_dict,dt=.1,warmup_period=0.0):

	min_time = warmup_period
	max_time = warmup_period

	time_index = 0
	road_segment_index = 13

	for veh_id in sim_dict:
		veh_data = sim_dict[veh_id]
		end_time = float(veh_data[-1][0])

		max_time = np.max([end_time,max_time])
		min_time = np.min([end_time,min_time])

	# times = np.arange(min_time,max_time+dt,dt)
	times = np.linspace(min_time,max_time,int((max_time-min_time)/dt))


	num_roads = len(roads)
	road_counts = np.zeros([len(times),num_roads])

	for veh_id in sim_dict:

		veh_data = sim_dict[veh_id]

		for row in veh_data:
			t = round(float(row[time_index]),1)
			road = row[road_segment_index]

			if(road in roads):

				t_effective = t - warmup_period

				times_index = int(round((t_effective-dt)/dt))
				road_index = road_indices[road]
				road_counts[times_index,road_index] = road_counts[times_index,road_index] + 1

	return times,road_counts

def get_road_segment_fuel_consumption(roads,road_indices,sim_dict,dt=.1,warmup_period=0.0):

	min_time = warmup_period
	max_time = warmup_period

	time_index = 0
	road_segment_index = 13

	for veh_id in sim_dict:
		veh_data = sim_dict[veh_id]
		end_time = float(veh_data[-1][0])

		max_time = np.max([end_time,max_time])
		min_time = np.min([end_time,min_time])

	# times = np.arange(min_time,max_time+dt,dt)
	times = np.linspace(min_time,max_time,int((max_time-min_time)/dt))

	num_roads = len(roads)
	total_fuel_consumptions = np.zeros([len(times),num_roads])

	for veh_id in sim_dict:

		veh_data = sim_dict[veh_id]

		for row in veh_data:
			t = round(float(row[time_index]),1)
			road = row[road_segment_index]

			if(road in roads):

				t_effective = t - warmup_period

				times_index = int(round((t_effective-dt)/dt))
				road_index = road_indices[road]

				fuel_consumption_individual_vehicle = float(row[19])


				total_fuel_consumptions[times_index,road_index] = total_fuel_consumptions[times_index,road_index] + fuel_consumption_individual_vehicle

	return times,total_fuel_consumptions




















