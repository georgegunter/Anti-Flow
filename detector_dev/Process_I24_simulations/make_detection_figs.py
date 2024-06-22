import numpy as np
import matplotlib.pyplot as plt

import os

from detector_dev.Process_I24_simulations.utils_classification import *

from detector_dev.Process_I24_simulations.i24_utils import get_sim_timeseries as get_sim_timeseries_i24
from detector_dev.Process_I24_simulations.i24_utils import get_attack_params as get_attack_params_i24




def get_f1_from_max_thresh(true_labels,max_rec_errors,thresh=7.481976509094238):
	assigned_labels_list = max_rec_errors > thresh

	f1_score = f1_calculation(assigned_labels_list=assigned_labels_list,vehicle_labels_list=true_labels)
	return f1_score

def get_f1_from_kmeans(true_labels,max_rec_errors):

	assigned_labels_list,_,_ = k_means_cluster(max_losses=max_rec_errors,true_labels=true_labels)

	f1_score = f1_calculation(assigned_labels_list=assigned_labels_list,vehicle_labels_list=true_labels)

	return f1_score


if __name__ == '__main__':

	loss_repo_path = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/Process_I24_simulations/normalized_detection_classification_results/max_rec_errors'

	benign_max_rec_errors_file = os.path.join(loss_repo_path,'max_rec_errors_Dur_0.0027839584656419447_Mag_-1.1173175105601907_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.1_ver_1.csv')

	rec_error_files = os.listdir(loss_repo_path)

	max_benign_rec_error = np.max(benign_max_rec_errors[:,1])

	all_files = os.listdir(loss_repo_path)

	all_max_rec_error_files = []

	for file in all_files:
		if('csv' in file):
			if(file != benign_max_rec_errors_file):
				all_max_rec_error_files.append(os.path.join(loss_repo_path,file))


	durations = []
	magnitudes = []
	f1_thresh = []
	f1_kmeans = []

	for file in all_max_rec_error_files:

		max_rec_error_data = np.loadtxt(file,delimiter=',')
		true_labels = max_rec_error_data[:,0]
		max_rec_errors = max_rec_error_data[:,1]

		f1_score_thresh = get_f1_from_max_thresh(true_labels,max_rec_errors)

		f1_score_kmeans = get_f1_from_kmeans(true_labels,max_rec_errors)

		attack_params = get_attack_params_i24(file)

		durations.append(attack_params[0])
		magnitudes.append(attack_params[1])
		f1_thresh.append(f1_score_thresh)
		f1_kmeans.append(f1_score_kmeans)

	fig = plt.figure(figsize=[15,10])
	plt.subplot(1,2,1)
	plt.scatter(durations,magnitudes,c=f1_thresh,s=500)
	plt.ylabel('Duration [s]',fontsize=20)
	plt.xlabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title('F1-scores using thresholding',fontsize=20)
	plt.colorbar()
	plt.clim([0,1.0])

	plt.subplot(1,2,2)
	plt.scatter(durations,magnitudes,c=f1_kmeans,s=500)
	# plt.ylabel('Duration [s]',fontsize=20)
	plt.xlabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.title('F1-scores using k-means',fontsize=20)
	plt.colorbar()
	plt.clim([0,1.0])

	plt.savefig("MLFW_f1_scatter.png",bbox_inches='tight')


	impact_data = np.loadtxt('attack_impact_param_sample.csv',delimiter=',')

	MTS_vals = []
	TSV_vals = []

	impact_durations = impact_data[:,0]


	for duration in durations:
		try:
			i = 0
			while(np.abs(duration - impact_durations[i])>=1e-3): i+= 1

			MTS_vals.append(impact_data[i,2])
			TSV_vals.append(impact_data[i,3])
		except:
			MTS_vals.append(None)
			TSV_vals.append(None)


	fig = plt.figure(figsize=[10,10])
	plt.subplot(2,2,1)
	plt.plot(MTS_vals,f1_kmeans,'.',markersize=20)
	# plt.xlabel('MTS [m/s]',fontsize=20)
	plt.ylabel('F1 score k-means',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.subplot(2,2,2)
	plt.plot(TSV_vals,f1_kmeans,'.',markersize=20)
	# plt.xlabel('TSV [m/s]',fontsize=20)
	# plt.ylabel('F1 score k-means',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.subplot(2,2,3)
	plt.plot(MTS_vals,f1_thresh,'.',markersize=20)
	plt.xlabel('MTS [m/s]',fontsize=20)
	plt.ylabel('F1 score thresh',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.subplot(2,2,4)
	plt.plot(TSV_vals,f1_thresh,'.',markersize=20)
	plt.xlabel('TSV [m/s]',fontsize=20)
	# plt.ylabel('F1 score thresh',fontsize=20)
	plt.yticks(fontsize=15)
	plt.yticks(fontsize=15)

	fig.suptitle('MLFW: detection vs. attack impact',fontsize=25)
	plt.savefig("MLFW_impact_vs_detection.png",bbox_inches='tight')


	






