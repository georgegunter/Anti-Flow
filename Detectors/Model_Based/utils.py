import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import csv
import ray
import time

def get_sim_timeseries(csv_path,warmup_period=0.0):
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
					data = [row[time_index],row[speed_index],row[headway_index],row[relvel_index]]
					curr_veh_data.append(data)
			row_num += 1

		#Add the very last vehicle's information:
		sim_dict[curr_veh_id] = np.array(curr_veh_data).astype(float)
		
	print('Data loaded.')
	return sim_dict

def bando_ovm_accel(v,s,v_l,p):
	Vm = p[0]
	s0 = p[1]
	s1 = p[2]
	a = p[3]
	b = p[4]
		
	v_opt = Vm*((np.tanh(s/s0-s1)+np.tanh(s1))/(1+np.tanh(s1)))
	ftl = (v_l-v)/(s**2)
	u = a*(v_opt-v) + b*ftl
	
	return u


def idm_accel(v,s,v_l,p):
	
	return u



def sim_cfm(v0,s0,v_l_vals,accel_func,p,dt):
	num_steps = len(v_l_vals)
	v_vals = np.zeros_like(v_l_vals)
	s_vals = np.zeros_like(v_l_vals)
	v_vals[0] = v0
	s_vals[0] = s0
	
	for i in range(1,num_steps):
		v = v_vals[i-1]
		s = s_vals[i-1]
		v_l = v_l_vals[i-1]
		dv_dt = accel_func(v,s,v_l,p)
		ds_dt = v_l -v
		v_new = v + dv_dt*dt
		s_new = s + ds_dt*dt
		
		v_vals[i] = v_new
		s_vals[i] = s_new
		
	return v_vals,s_vals

def sim_cfm_with_pos(v0,p0,p_l_vals,v_l_vals,accel_func,cfm_params,dt):
	num_steps = len(v_l_vals)
	v_vals = np.zeros_like(v_l_vals)
	s_vals = np.zeros_like(v_l_vals)
	p_vals = np.zeros_like(v_l_vals)
	v_vals[0] = v0
	p_vals[0] = 0
	s_vals[0] = p_l_vals[0]
	
	for i in range(1,num_steps):
		v = v_vals[i-1]
		s = s_vals[i-1]
		v_l = v_l_vals[i-1]

		dv_dt = accel_func(v,s,v_l,cfm_params)
		ds_dt = v_l - v
		dp_dt = v
		v_new = v + dv_dt*dt
		p_new = p + dp_dt*dt
		s_new = p_new
		
		v_vals[i] = v_new
		s_vals[i] = s_new
		p_vals[i] = p_new
		
	return v_vals,p_vals,s_vals



def spacing_rmse(s_vals_real,s_vals_sim):
	error = s_vals_real - s_vals_sim
	squared_error = np.multiply(error,error)
	return np.sqrt(np.mean(squared_error))

def get_spacing_rmse(v_vals_real,s_vals_real,v_l_vals,accel_func,p,dt):
	v0 = v_vals_real[0]
	s0 = s_vals_real[0]
	v_vals_sim,s_vals_sim = sim_cfm(v0,s0,v_l_vals,accel_func,p,dt)
	return spacing_rmse(s_vals_real,s_vals_sim)

def get_timeseries_dict_mean_spacing_rmse(timeseries_dict,accel_func,p,dt):
	
	spacing_rmse_vals = []
	
	for veh_id in timeseries_dict:
		v_vals_real = timeseries_dict[veh_id][:,1]
		s_vals_real = timeseries_dict[veh_id][:,2]
		v_l_vals = v_vals_real + timeseries_dict[veh_id][:,3]
		rmse = get_spacing_rmse(v_vals_real,s_vals_real,v_l_vals,accel_func,p,dt)
		spacing_rmse_vals.append(rmse)
		
	mean_spacing_rmse = np.mean(spacing_rmse_vals)
	
	return mean_spacing_rmse

def get_opt_cfm_params(timeseries_dict,accel_func,p_init,dt):
	def obj_func(p):
		return get_timeseries_dict_mean_spacing_rmse(timeseries_dict,accel_func,p,dt)

	res = minimize(obj_func, p_init, method='BFGS',options={'gtol': 1e-6, 'disp': True})

	print('Finished.')

def sliding_window_spacing_error(v_vals_real,s_vals_real,v_l_vals,accel_func,p,seq_len,dt):
	num_time_samples = len(v_vals_real)
	spacing_rmse_vals = []
	
	num_evals = num_time_samples-seq_len
	
	for i in range(num_evals):
		rmse = get_spacing_rmse(v_vals_real[i:i+seq_len],s_vals_real[i:i+seq_len],v_l_vals[i:i+seq_len],accel_func,p,dt)
		spacing_rmse_vals.append(rmse)
		
	return spacing_rmse_vals

def get_rec_error_timeseries_dict(timeseries_dict,seq_len,accel_func,p,dt):
	reconstruction_losses = {}
	
	for veh_id in timeseries_dict:
		v_vals_real = timeseries_dict[veh_id][:,1]
		s_vals_real = timeseries_dict[veh_id][:,2]
		v_l_vals = v_vals_real + timeseries_dict[veh_id][:,3]
		
		rec_errors = sliding_window_spacing_error(v_vals_real,s_vals_real,v_l_vals,accel_func,p,seq_len,dt)
		
		reconstruction_losses[veh_id] = np.array(rec_errors)
		
	return reconstruction_losses

def get_and_write_rec_errors(csv_path,rec_error_repo,accel_func,p,seq_len,warmup_period,dt):
	begin_time = time.time()
	timeseries_dict = get_sim_timeseries(csv_path,warmup_period)
	reconstruction_losses = get_rec_error_timeseries_dict(timeseries_dict,seq_len,accel_func,p,dt)

	i = 0
	while(csv_path[i:i+3]!='TAD'):
		i+=1

	file_name = os.path.join(rec_error_repo,csv_path[i:])

	with open(file_name, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for veh_id in reconstruction_losses:
			losses = reconstruction_losses[veh_id]
			num_samples = len(losses)
			for i in range(num_samples):
				writer.writerow([veh_id,losses[i]])
	print('Total compute time: '+str(time.time()-begin_time))
	return reconstruction_losses


@ray.remote
def ray_get_and_write_rec_errors(csv_path,rec_error_repo,accel_func,p,seq_len,warmup_period,dt):
	return get_and_write_rec_errors(csv_path,rec_error_repo,accel_func,p,seq_len,warmup_period,dt)


def get_all_rec_errors(sim_repo,rec_error_repo,accel_func,p,seq_len,warmup_period,dt):
	csv_path_list = []
	files = os.listdir(sim_repo)
	for file in files:
		if('csv' in file and 'ring' in file):
			csv_path_list.append(os.path.join(sim_repo,file))

	num_files = len(csv_path_list)
	files_processed = 0

	rec_errors_ids = []
	for csv_path in csv_path_list:
		rec_errors_ids.append(ray_get_and_write_rec_errors.remote(csv_path,rec_error_repo,accel_func,p,seq_len,warmup_period,dt))
		files_processed += 1
		print('Files processed: '+str(files_processed)+'/'+str(num_files))

	rec_error_results = ray.get(rec_errors_ids)

	return rec_error_results


if __name__ == '__main__':
	ray.init(num_cpus=5)

	# Analysis parameters:
	sim_repo = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep'
	rec_error_repo = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep/model_based_rec_errors/'
	accel_func = bando_ovm_accel
	p = [8.86655756,2.09865379,3.01753401,0.67993545,25.1010223] #Found via a calibration routine on non-attacked data
	dt = 0.1
	seq_len = 200
	warmup_period = 100.0

	# Find and write reconstruction errors:
	get_all_rec_errors(sim_repo,rec_error_repo,accel_func,p,seq_len,warmup_period,dt)

	print('Finished calculating reconstruction errors.')


	





