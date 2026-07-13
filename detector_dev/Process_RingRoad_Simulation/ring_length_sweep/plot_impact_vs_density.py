import numpy as np
import matplotlib.pyplot as plt
from sweep_utils import *

def get_ring_length(sim_name):
	i = 5
	while(sim_name[i] != 'm'):
		i += 1
	return float(sim_name[5:i])


def plot_min_max_ave_IFs():

	aggregate_MTS_dict = dict.fromkeys(list(ring_lengths))
	for length in ring_lengths:
		aggregate_MTS_dict[length] = [[],[],[]]

	aggregate_TSV_dict = dict.fromkeys(list(ring_lengths))
	for length in ring_lengths:
		aggregate_TSV_dict[length] = [[],[],[]]

	for i in range(len(strong_attack_results[:,0])):
		curr_MTS = strong_attack_results[i,1]
		curr_TSV = strong_attack_results[i,2]
		length = strong_attack_results[i,0]
		aggregate_MTS_dict[length][0].append(curr_MTS)
		aggregate_TSV_dict[length][0].append(curr_TSV)

	for i in range(len(medium_attack_results[:,0])):
		curr_MTS = medium_attack_results[i,1]
		curr_TSV = medium_attack_results[i,2]
		length = medium_attack_results[i,0]
		aggregate_MTS_dict[length][1].append(curr_MTS)
		aggregate_TSV_dict[length][1].append(curr_TSV)

	for i in range(len(benign_results[:,0])):
		curr_MTS = benign_results[i,1]
		curr_TSV = benign_results[i,2]
		length = benign_results[i,0]
		aggregate_MTS_dict[length][2].append(curr_MTS)
		aggregate_TSV_dict[length][2].append(curr_TSV)

	if(True):

		ave_strong_MTS = []
		for length in ring_lengths:
			ave_strong_MTS.append(np.mean(aggregate_MTS_dict[length][0]))
		min_strong_MTS = []
		for length in ring_lengths:
			min_strong_MTS.append(np.min(aggregate_MTS_dict[length][0]))
		max_strong_MTS = []
		for length in ring_lengths:
			max_strong_MTS.append(np.max(aggregate_MTS_dict[length][0]))

		ave_medium_MTS = []
		for length in ring_lengths:
			ave_medium_MTS.append(np.mean(aggregate_MTS_dict[length][1]))
		min_medium_MTS = []
		for length in ring_lengths:
			min_medium_MTS.append(np.min(aggregate_MTS_dict[length][1]))
		max_medium_MTS = []
		for length in ring_lengths:
			max_medium_MTS.append(np.max(aggregate_MTS_dict[length][1]))

		ave_benign_MTS = []
		for length in ring_lengths:
			ave_benign_MTS.append(np.mean(aggregate_MTS_dict[length][2]))
		min_benign_MTS = []
		for length in ring_lengths:
			min_benign_MTS.append(np.min(aggregate_MTS_dict[length][2]))
		max_benign_MTS = []
		for length in ring_lengths:
			max_benign_MTS.append(np.max(aggregate_MTS_dict[length][2]))

		ave_strong_TSV = []
		for length in ring_lengths:
			ave_strong_TSV.append(np.mean(aggregate_TSV_dict[length][0]))
		min_strong_TSV = []
		for length in ring_lengths:
			min_strong_TSV.append(np.min(aggregate_TSV_dict[length][0]))
		max_strong_TSV = []
		for length in ring_lengths:
			max_strong_TSV.append(np.max(aggregate_TSV_dict[length][0]))

		ave_medium_TSV = []
		for length in ring_lengths:
			ave_medium_TSV.append(np.mean(aggregate_TSV_dict[length][1]))
		min_medium_TSV = []
		for length in ring_lengths:
			min_medium_TSV.append(np.min(aggregate_TSV_dict[length][1]))
		max_medium_TSV = []
		for length in ring_lengths:
			max_medium_TSV.append(np.max(aggregate_TSV_dict[length][1]))

		ave_benign_TSV = []
		for length in ring_lengths:
			ave_benign_TSV.append(np.mean(aggregate_TSV_dict[length][2]))
		min_benign_TSV = []
		for length in ring_lengths:
			min_benign_TSV.append(np.min(aggregate_TSV_dict[length][2]))
		max_benign_TSV = []
		for length in ring_lengths:
			max_benign_TSV.append(np.max(aggregate_TSV_dict[length][2]))


	# Plotting:

	plt.figure()
	plt.subplot(1,2,1)
	plt.fill_between(ring_lengths,min_strong_MTS,max_strong_MTS,alpha=0.1,color='r')
	plt.plot(ring_lengths,ave_strong_MTS,'r',linewidth=5)
	# plt.fill_between(ring_lengths,min_medium_MTS,max_medium_MTS,alpha=0.1,color='b')
	# plt.plot(ring_lengths,ave_medium_MTS,'b',linewidth=5)
	plt.fill_between(ring_lengths,min_benign_MTS,max_benign_MTS,alpha=0.1,color='g')
	plt.plot(ring_lengths,ave_benign_MTS,'g',linewidth=5)
	
	plt.subplot(1,2,2)
	plt.fill_between(ring_lengths,min_strong_TSV,max_strong_TSV,alpha=0.1,color='r')
	plt.plot(ring_lengths,ave_strong_TSV,'r',linewidth=5)
	# plt.fill_between(ring_lengths,min_medium_TSV,max_medium_TSV,alpha=0.1,color='b')
	# plt.plot(ring_lengths,ave_medium_TSV,'b',linewidth=5)
	plt.fill_between(ring_lengths,min_benign_TSV,max_benign_TSV,alpha=0.1,color='g')
	plt.plot(ring_lengths,ave_benign_TSV,'g',linewidth=5)

	plt.show()



if __name__ == '__main__':
	sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_sweep_ring_length'

	ring_sweep_sim_results = []

	load_sims_from_file = False
	if(load_sims_from_file):

		all_files_in_sim_repo = os.listdir(sim_repo_path)

		sim_files = []

		for file in all_files_in_sim_repo:
			if(('ver' in file) and ('csv' in file)):
				sim_files.append(os.path.join(sim_repo_path,file))

		
		for file_path in sim_files:
			ring_sweep_sim_results.append(get_relevant_data(file_path))
	else:
		ring_sweep_sim_np = np.loadtxt('performance_metrics_single_lane_ring_road_sweep_ring_length.csv',delimiter=',',dtype=str)
		ring_sweep_sim_results = list(ring_sweep_sim_np)

	strong_attack_results = []
	medium_attack_results = []
	benign_results = []

	for sim_res in ring_sweep_sim_results:
		if('TAD_10' in sim_res[0]):
			ring_length = get_ring_length(sim_res[0])
			MTS = sim_res[1]
			MTSV = sim_res[2]
			strong_attack_results.append([ring_length,MTS,MTSV])

		if('TAD_5.0' in sim_res[0]):
			ring_length = get_ring_length(sim_res[0])
			MTS = sim_res[1]
			MTSV = sim_res[2]
			medium_attack_results.append([ring_length,MTS,MTSV])


		if('TAD_0.0' in sim_res[0]):
			ring_length = get_ring_length(sim_res[0])
			MTS = sim_res[1]
			MTSV = sim_res[2]
			benign_results.append([ring_length,MTS,MTSV])

	strong_attack_results = np.array(strong_attack_results).astype(float)
	medium_attack_results = np.array(medium_attack_results).astype(float)
	benign_results = np.array(benign_results).astype(float)


	# get the average IF for each attack and length:

	make_fill_plot = True

	if(make_fill_plot):

		ring_lengths = np.unique(strong_attack_results[:,0])



	else:

		# convert ring lengths to densities in veh/km:
		num_vehicles = 40
		strong_attack_results[:,0] = np.divide(np.ones_like(strong_attack_results[:,0])*num_vehicles,strong_attack_results[:,0])*1000
		medium_attack_results[:,0] = np.divide(np.ones_like(medium_attack_results[:,0])*num_vehicles,medium_attack_results[:,0])*1000
		benign_results[:,0] = np.divide(np.ones_like(benign_results[:,0])*num_vehicles,benign_results[:,0])*1000






		plt.figure(figsize=[20,10])
		plt.subplot(1,2,1)
		# plt.title('ring length vs. MTS')
		plt.plot(benign_results[:,0],benign_results[:,1],'g.')
		plt.plot(medium_attack_results[:,0],medium_attack_results[:,1],'b.')
		plt.plot(strong_attack_results[:,0],strong_attack_results[:,1],'r.')

		plt.subplot(1,2,2)
		# plt.title('Traffic density vs. TSV')
		plt.plot(benign_results[:,0],benign_results[:,2],'g.')
		plt.plot(medium_attack_results[:,0],medium_attack_results[:,2],'b.')
		plt.plot(strong_attack_results[:,0],strong_attack_results[:,2],'r.')
		plt.show()




