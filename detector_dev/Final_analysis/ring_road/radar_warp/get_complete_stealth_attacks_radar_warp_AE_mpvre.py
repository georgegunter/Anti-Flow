import os
import numpy as np
import flow
from copy import deepcopy
import sys
import matplotlib.pyplot as plt

from Data_Processing.sim_processing_utils import get_trajectory_timeseries

from hull_classification_utils import *

mpvre_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_monte_carlo_rec_errors/max_per_vehicle_rec_errors'

def get_attack_params(sim_name):
	sim_params = []
	j = 7
	k = 7
	while (sim_name[j] != '_'):j+=1
	sim_params.append(float(sim_name[k:j]))

	j += 4
	k = j
	while (sim_name[j] != '_'):j+=1
	sim_params.append(float(sim_name[k:j]))

	j += 4
	k = j
	while (sim_name[j] != '_'):j+=1
	sim_params.append(float(sim_name[k:j]))

	return np.array(sim_params)


def get_mpvre_vals(sim_name):
	mpvre_file_path = os.path.join(mpvre_repo_path,sim_name)

	mpvre_data = np.loadtxt(mpvre_file_path,delimiter=',',dtype=str)

	mpvre_vals = []

	for mpvre_datum in mpvre_data:
		is_mal = 'adv' in mpvre_datum[0]
		mpvre_vals.append([is_mal,float(mpvre_datum[1])])

	return np.array(mpvre_vals)


def get_mpvre_labels(mpvre_vals,max_mpvre):
	return mpvre_vals[:,1] > max_mpvre 

def is_mpvre_complete_stealth(sim_name,max_mpvre):
	mpvre_vals = get_mpvre_vals(sim_name)
	
	mpvre_labels = get_mpvre_labels(mpvre_vals,max_mpvre)

	is_complete_stealth = np.sum(np.logical_and(mpvre_vals[:,0],mpvre_labels)) == 0

	return is_complete_stealth

def get_all_mpvre_complete_stealth_attacks(all_sim_files,max_mpvre):
	complete_stealth_attacks = []
	for sim_name in all_sim_files:
		if(is_mpvre_complete_stealth(sim_name,max_mpvre)):
			complete_stealth_attacks.append(sim_name)

	return complete_stealth_attacks


def get_performance_metrics_dict(performance_metrics_list_path):
	performance_metrics_list = np.loadtxt(performance_metrics_list_path,delimiter=',',dtype=str)

	performance_metrics_dict = {}
	for datum in performance_metrics_list:
		sim_name = datum[0]
		performance_metrics_dict[sim_name] = np.array(datum[1:]).astype(float)

	return performance_metrics_dict



def get_performance_metrics_from_sim_names(sim_name_list,performance_metrics_dict):
	performance_metrics_list = []
	for sim_name in sim_name_list:
		sim_params = get_attack_params(sim_name)
		performance_metrics = performance_metrics_dict[sim_name]
		datum = [sim_params[0],sim_params[1],sim_params[2],performance_metrics[0],performance_metrics[1],performance_metrics[2]]
		performance_metrics_list.append(datum)

	return np.array(performance_metrics_list)





if __name__ == '__main__':

	# Load data:

	files = os.listdir(mpvre_repo_path)

	all_sim_files = []

	for file in files:
		if('.csv' in file):
			all_sim_files.append(file)

	complete_stealth_attacks = get_all_mpvre_complete_stealth_attacks(all_sim_files,max_mpvre=170)

	performance_metrics_list_path = 'performance_metrics_radar_warp.csv'

	performance_metrics_dict = get_performance_metrics_dict(performance_metrics_list_path)

	complete_stealth_performance_metrics_array = get_performance_metrics_from_sim_names(
		sim_name_list=complete_stealth_attacks,
		performance_metrics_dict=performance_metrics_dict)


	all_attacks_performance_metrics_array = get_performance_metrics_from_sim_names(
		sim_name_list=all_sim_files,
		performance_metrics_dict=performance_metrics_dict)


	want_figures = True

	if(want_figures):
		U_stealthy = complete_stealth_performance_metrics_array
		U_all = all_attacks_performance_metrics_array

		want_pareto_subplot = True
		if(want_pareto_subplot):

			fig = plt.figure(figsize=[20,5])

			plt.subplot(1,3,1)
			plt.plot(U_all[:,3],U_all[:,4],'r.',markersize=10,label='Detected')
			plt.plot(U_stealthy[:,3],U_stealthy[:,4],'b.',markersize=20,label='Complete stealth')
			plt.legend(fontsize=15)
			
			plt.xlabel('MTS [m/s]',fontsize=20)
			plt.ylabel('TSV [m/s]',fontsize=20)
			plt.xticks(fontsize=20)
			plt.yticks(fontsize=20)
			plt.ylim([5.0,17.0])
			plt.xlim([4.5,7.5])
			plt.grid()

			plt.subplot(1,3,2)
			plt.plot(U_all[:,3],U_all[:,5],'r.',markersize=10,label='Detected')
			plt.plot(U_stealthy[:,3],U_stealthy[:,5],'b.',markersize=20,label='Complete stealth')
			# plt.legend(fontsize=20)
			
			plt.xlabel('MTS [m/s]',fontsize=20)
			plt.ylabel('Energy [grams/s]',fontsize=20)
			plt.xticks(fontsize=20)
			plt.yticks(fontsize=20)
			plt.ylim([.6,1.5])
			plt.xlim([4.5,7.5])
			plt.grid()

			plt.subplot(1,3,3)
			plt.plot(U_all[:,4],U_all[:,5],'r.',markersize=10,label='Detected')
			plt.plot(U_stealthy[:,4],U_stealthy[:,5],'b.',markersize=20,label='Complete stealth')
			# plt.legend(fontsize=20)
			
			plt.xlabel('TSV [m/s]',fontsize=20)
			plt.ylabel('Energy [grams/s]',fontsize=20)
			plt.xticks(fontsize=20)
			plt.yticks(fontsize=20)
			plt.xlim([5.0,17.0])
			plt.ylim([.6,1.5])
			plt.grid()

			plt.savefig('ring_complete_stealth_AE_mpvre_pareto.pdf',bbox_inches='tight')

			print('Finished analysis, figure saved.')

			# plt.show()






