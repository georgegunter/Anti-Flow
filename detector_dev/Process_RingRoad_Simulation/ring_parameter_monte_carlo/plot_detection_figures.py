import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# from sweep_utils import *

FIGURE_SIZE = [35,10]
TITLE_FONTSIZE = 40
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 35
MARKERSIZE = 30

def get_labeled_REs(file_path):
	RE_data = np.loadtxt(file_path,dtype=str,delimiter=',')
	veh_ids = np.unique(RE_data[:,0])

	RE_timeseries_dict = dict.fromkeys(veh_ids)
	for veh_id in veh_ids:
		RE_timeseries_dict[veh_id] = []

	for i in range(len(RE_data[:,0])):
		veh_id = str(RE_data[i,0])
		RE = float(RE_data[i,2])
		RE_timeseries_dict[veh_id].append(RE)

	return RE_timeseries_dict



def get_max_REs(RE_timeseries_dict):
	max_REs = []
	for veh_id in RE_timeseries_dict:
		is_attacker = 'RDA' in veh_id
		if(is_attacker):
			max_REs.append([1,np.max(RE_timeseries_dict[veh_id])])
		else:
			max_REs.append([0,np.max(RE_timeseries_dict[veh_id])])
	max_REs = np.array(max_REs)
	return max_REs


def get_ring_length(file_path):
	i = 0
	while(file_path[i:i+7] !='_single'):i +=1
	i -= 1
	j = i
	while(file_path[i] != '_'): i-=1
	i+=1
	return float(file_path[i:j])


def get_classifications(benign_max_RE_values,attacked_max_RE_values):
	max_RE_benign = np.max(benign_max_RE_values[:,1])
	classifications = attacked_max_RE_values[:,1] > max_RE_benign
	return classifications


if __name__ == '__main__':

	RE_vals_repo = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/strong_attack/normalized_rec_errors'

	all_RE_file_paths = []
	all_RE_file_names = os.listdir(RE_vals_repo)
	for file in all_RE_file_names:
		if('.csv' in file):
			all_RE_file_paths.append(os.path.join(RE_vals_repo,file))


	get_all_max_REs = True

	if(get_all_max_REs):

		benign_max_REs = []
		attacked_max_REs = []

		for file_path in tqdm(all_RE_file_paths):

			print('Processing: '+file_path)

			ring_length = get_ring_length(file_path)
			is_benign = 'benign' in file_path
			is_attacked = 'TAD_10.0_ADR_-0.5' in file_path

			if(is_benign):
				RE_timeseries_dict = get_labeled_REs(file_path)
				max_REs = get_max_REs(RE_timeseries_dict)
				benign_max_REs.append(max_REs)
			if(is_attacked):
				RE_timeseries_dict = get_labeled_REs(file_path)
				max_REs = get_max_REs(RE_timeseries_dict)
				attacked_max_REs.append(max_REs)


		attacked_max_REs_agg = np.array(attacked_max_REs).reshape((100*40,2))

		comp_max_REs_agg = attacked_max_REs_agg[np.where(attacked_max_REs_agg[:,0] == 1),:]

		comp_max_REs_agg = comp_max_REs_agg[0,:,1]

		noncomp_max_REs_agg = attacked_max_REs_agg[np.where(attacked_max_REs_agg[:,0] == 0),:]

		noncomp_max_REs_agg = noncomp_max_REs_agg[0,:,1]

		non_attack_max_REs_agg = np.array(benign_max_REs).reshape((100*40,2))

		benign_max_REs_agg = non_attack_max_REs_agg[:,1]


		### Plotting: ###

		fig = plt.figure(figsize = FIGURE_SIZE)

		plt.violinplot([noncomp_max_REs_agg,comp_max_REs_agg,benign_max_REs_agg])

		ax = fig.axes[0]
		ax.yaxis.grid(True)
		ax.set_xticks([y + 1 for y in range(3)],labels=['Attacked traffic, non compromised','Attacked traffic, compromised','Benign traffic'],fontsize=TICK_FONTSIZE)
		ax.set_ylabel('MpVREs',fontsize=LABEL_FONTSIZE)
		# ax.set_ylabel('Observed values').set_label(['Non comp','Comp','Benign'])
		plt.yticks(fontsize=TICK_FONTSIZE)
		

		figures_path = '/Users/vanderbilt/Desktop/Research_2024/Usenix 2024/Figures/'

		save_figure_path = os.path.join(figures_path,'ring_MC_RE_comparison.png')

		plt.savefig(save_figure_path,bbox_inches='tight')

		print('Saved figure.')


