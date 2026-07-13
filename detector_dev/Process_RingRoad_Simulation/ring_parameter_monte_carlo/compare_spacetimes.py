import numpy as np
import matplotlib.pyplot as plt


#related to ring road:
from flow.visualize.visualize_ring import *
from detector_dev.Process_RingRoad_Simulation.process_ring_attacks import get_attack_params as get_attack_params_ring


from Data_Processing.sim_processing_utils import get_trajectory_timeseries


from detector_dev.Process_RingRoad_Simulation.get_performance_metrics_all import *
import os
import ray
import csv
from tqdm import tqdm


def plot_one_spacetime(sim_data_dict):
	veh_ids = list(sim_data_dict.keys())
	ring_positions = get_ring_positions(sim_data_dict,csv_path,ring_length)
	times,positions,speeds = stack_data_for_spacetime(sim_data_dict,ring_positions)

	positions_mod_ring_length = np.mod(positions,ring_length)

	fontsize=15
	pt.scatter(times,positions_mod_ring_length,c=speeds)
	pt.ylabel('Position [m]',fontsize=fontsize)
	pt.xlabel('Time [s]',fontsize=fontsize)
	cbar = pt.colorbar(label='Speed [m/s]')
	cbar.ax.tick_params(labelsize=10)
	plt.clim([0,15.0])


def plot_space_time_side_by_side(attack_sim_data_dict,benign_sim_data_dict,save_fig=True,file_name=None,fig_title=None):
	fig = plt.figure(figsize=[15,7])
	plt.subplot(1,2,1)
	plot_one_spacetime(sim_data_dict=benign_sim_data_dict)
	plt.title('Benign',fontsize=15)
	plt.subplot(1,2,2)
	plot_one_spacetime(sim_data_dict=attack_sim_data_dict)
	plt.title('Attacked',fontsize=15)

	if(fig_title != None):
		fig.suptitle(fig_title)

	if(save_fig):
		if(file_name != None):
			plt.savefig(file_name,bbox_inches='tight')
		else:
			plt.savefig('spacetime_comp.png',bbox_inches='tight')

	print('Saved spacetime figure.')


def compare_veh_ids(attack_sim_data_dict,benign_sim_data_dict):
	attack_veh_ids = list(attack_sim_data_dict.keys())

	benign_veh_ids = list(benign_sim_data_dict.keys())

	for veh_id in benign_veh_ids:
		if(veh_id not in attack_veh_ids):
			print(veh_id)




def plot_spacetime_comp_version_based(run_num,emission_path,default_attack_name,default_benign_name):

	attack_file_path = os.path.join(emission_path,default_attack_name+'_ver_'+str(run_num)+'.csv')

	benign_file_path = os.path.join(emission_path,default_benign_name+'_ver_'+str(run_num)+'.csv')

	attack_sim_data_dict = get_sim_data_dict_ring(csv_path=attack_file_path,warmup_period=0.0)

	benign_sim_data_dict = get_sim_data_dict_ring(csv_path=benign_file_path,warmup_period=0.0)

	fig_title = 'Run '+str(run_num)

	file_name = 'spacetime_comp_ver_'+str(run_num)+'.png'

	plot_space_time_side_by_side(attack_sim_data_dict,
		benign_sim_data_dict,
		save_fig=True,
		file_name=file_name,
		fig_title=fig_title)


if __name__ == '__main__':



	if(True):
		emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/strong_attack'

		default_attack_name = 'ring_600m_single_lane_TAD_10.0_ADR_-0.5'

		default_benign_name = 'ring_600m_single_lane_benign'

		run_num = 10

		plot_spacetime_comp_version_based(run_num,emission_path,default_attack_name,default_benign_name)


	if(False):
		emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/medium_attack'

		default_attack_name = 'ring_600m_single_lane_TAD_5.0_ADR_-0.25'

		default_benign_name = 'ring_600m_single_lane_benign'


		run_num = 81+1


		run_num = 91+1
		run_num = 93+1


		run_num = 100

		attack_file_path = os.path.join(emission_path,default_attack_name+'_ver_'+str(run_num)+'.csv')

		benign_file_path = os.path.join(emission_path,default_benign_name+'_ver_'+str(run_num)+'.csv')

		attack_sim_data_dict = get_sim_data_dict_ring(csv_path=attack_file_path,warmup_period=0.0)

		benign_sim_data_dict = get_sim_data_dict_ring(csv_path=benign_file_path,warmup_period=0.0)

		attack_veh_ids = list(attack_sim_data_dict.keys())

		benign_veh_ids = list(benign_sim_data_dict.keys())

		for veh_id in benign_veh_ids:
			if(veh_id not in attack_veh_ids):
				print(veh_id)



		##### SCRATCH: #####
		num_run = 4
		benign_file_path = os.path.join(emission_path,default_benign_name+'_ver_'+str(run_num)+'.csv')
		benign_sim_data_dict = get_sim_data_dict_ring(csv_path=benign_file_path,warmup_period=0.0)
		benign_veh_ids = list(benign_sim_data_dict.keys())

		for i in tqdm(range(1,100)):
			attack_file_path = os.path.join(emission_path,default_attack_name+'_ver_'+str(i)+'.csv')
			attack_sim_data_dict = get_sim_data_dict_ring(csv_path=attack_file_path,warmup_period=0.0)
			attack_veh_ids = list(attack_sim_data_dict.keys())

			if(benign_veh_ids[0] in attack_veh_ids):
				print('found match: '+str(i))
				break




