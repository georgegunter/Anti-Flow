import numpy as np
import os
import matplotlib.pyplot as pt
from i24_utils import get_losses_csv,get_sim_timeseries_all_data,get_sim_timeseries
from i24_utils import get_sim_name,get_attack_params,get_vehicle_types

from utils_classification import k_means_cluster,f1_calculation,threshold_classifer


from sklearn import metrics

import ray


def get_vehicle_labels(csv_path):
	vehicle_types = get_vehicle_types(csv_path)
	veh_ids = list(vehicle_types.keys())
	vehicle_labels_dict = dict.fromkeys(veh_ids)

	for veh_id in vehicle_types:
		vehicle_type = vehicle_types[veh_id]
		if('attack' in vehicle_type):
			vehicle_labels_dict[veh_id] = 1
		else:
			vehicle_labels_dict[veh_id] = 0
	return vehicle_labels_dict


def get_relevant_vehicle_labels(vehicle_labels_dict,rec_error_dict):
	veh_ids_relevant = list(rec_error_dict.keys())
	vehicle_labels_dict_relevant = dict.fromkeys(veh_ids_relevant)

	for veh_id in veh_ids_relevant:
		vehicle_labels_dict_relevant[veh_id] = vehicle_labels_dict[veh_id]

	return vehicle_labels_dict_relevant


def get_f1_values_k_means_classifier(rec_error_csv_path,sim_data_csv_path):
	rec_error_dict = get_losses_csv(rec_error_csv_path)
	k_means_labels_dict,_ = k_means_cluster(rec_error_dict,cluster_diff=0.1)

	vehicle_labels_dict = get_vehicle_labels(sim_data_csv_path)
	vehicle_labels_dict = get_relevant_vehicle_labels(vehicle_labels_dict,rec_error_dict)

	assigned_labels_list = []
	vehicle_labels_list = []

	for veh_id in vehicle_labels_dict:
		assigned_labels_list.append(k_means_labels_dict[veh_id])
		vehicle_labels_list.append(vehicle_labels_dict[veh_id])

	f1_score = f1_calculation(assigned_labels_list,vehicle_labels_list)

	return f1_score


def get_f1_values_threshold_classifier(rec_error_csv_path,sim_data_csv_path):
	rec_error_dict = get_losses_csv(rec_error_csv_path)

	#change to threshold classifier:
	k_means_labels_dict,_ = k_means_cluster(rec_error_dict,cluster_diff=0.1)

	vehicle_labels_dict = get_vehicle_labels(sim_data_csv_path)
	vehicle_labels_dict = get_relevant_vehicle_labels(vehicle_labels_dict,rec_error_dict)

	assigned_labels_list = []
	vehicle_labels_list = []

	for veh_id in vehicle_labels_dict:
		assigned_labels_list.append(k_means_labels_dict[veh_id])
		vehicle_labels_list.append(vehicle_labels_dict[veh_id])

	f1_score = f1_calculation(assigned_labels_list,vehicle_labels_list)

	return f1_score


def get_f1_value_both_classifiers(rec_error_csv_path,sim_data_csv_path,max_rec_error_in_training=3449.2494329637098):
	rec_error_dict = get_losses_csv(rec_error_csv_path)

	k_means_labels_dict,_ = k_means_cluster(rec_error_dict,cluster_diff=0.1)
	threshold_labels_dict = threshold_classifer(rec_error_dict,threshold=max_rec_error_in_training)

	vehicle_labels_dict = get_vehicle_labels(sim_data_csv_path)
	vehicle_labels_dict = get_relevant_vehicle_labels(vehicle_labels_dict,rec_error_dict)

	assigned_labels_list_k_means = []
	assigned_labels_list_threshold = []
	vehicle_labels_list = []

	for veh_id in vehicle_labels_dict:
		assigned_labels_list_k_means.append(k_means_labels_dict[veh_id])
		assigned_labels_list_threshold.append(threshold_labels_dict[veh_id])
		vehicle_labels_list.append(vehicle_labels_dict[veh_id])


	f1_score_k_means = f1_calculation(assigned_labels_list_k_means,vehicle_labels_list)
	f1_score_threshold = f1_calculation(assigned_labels_list_threshold,vehicle_labels_list)

	return f1_score_k_means,f1_score_threshold



def get_auc(rec_error_csv_path,sim_data_csv_path):
	rec_error_dict = get_losses_csv(rec_error_csv_path)
	vehicle_labels_dict = get_vehicle_labels(sim_data_csv_path)
	vehicle_labels_dict = get_relevant_vehicle_labels(vehicle_labels_dict,rec_error_dict)


	pred = []
	y = []

	for veh_id in vehicle_labels_dict:
		MpVRE = np.max(rec_error_dict[veh_id])
		y.append(vehicle_labels_dict[veh_id])
		pred.append(MpVRE)

	fpr_vals, tpr_vals, thresholds_vals = metrics.roc_curve(y, pred, pos_label=1)

	auc_val = metrics.auc(fpr_vals, tpr_vals)

	return auc_val,fpr_vals,tpr_vals



def get_attack_file_name_from_params(duration,magnitude,inflow=1800):
	file_name = 'Dur_'+str(duration)+'_Mag_'+str(magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_0.2_AttackPenetration_0.1_ver_1.csv'
	return file_name



if __name__ == '__main__':

	inflow = 1200

	all_rec_error_csv_paths = []
	rec_error_emission_repo = '/Volumes/My Passport for Mac/i24_random_sample/ae_rec_error_results/'+str(inflow)+'_inflow'
	all_rec_error_csv_file_names = os.listdir(rec_error_emission_repo)
	for csv_rec_error_file in all_rec_error_csv_file_names:
		all_rec_error_csv_paths.append(os.path.join(rec_error_emission_repo,csv_rec_error_file))

	all_sim_data_csv_paths = []
	sim_data_repo = '/Volumes/My Passport for Mac/i24_random_sample/simulations/'+str(inflow)+'_inflow'
	all_sim_data_csv_file_names = os.listdir(sim_data_repo)
	for csv_sim_data_file in all_sim_data_csv_file_names:
		all_sim_data_csv_paths.append(os.path.join(sim_data_repo,csv_sim_data_file))

	# Need to match rec error values with 

	f1_score_values_k_means = []
	f1_score_values_thershold = []

	max_rec_error_in_training = 3449.2494329637098

	for rec_error_csv_path in all_rec_error_csv_paths:
		sim_name = get_sim_name(rec_error_csv_path)
		print(sim_name)
		sim_data_csv_path = os.path.join(sim_data_repo,sim_name)

		f1_score_k_means,f1_score_threshold = get_f1_value_both_classifiers(rec_error_csv_path,sim_data_csv_path,max_rec_error_in_training)

		f1_score_values_k_means.append([sim_name,f1_score_k_means])

		f1_score_values_thershold.append([sim_name,f1_score_threshold])



	auc_val_list = []
	fpr_vals_list = []
	tpr_vals_list = []

	for rec_error_csv_path in all_rec_error_csv_paths:
		sim_name = get_sim_name(rec_error_csv_path)
		print(sim_name)
		sim_data_csv_path = os.path.join(sim_data_repo,sim_name)


		auc_val,fpr_vals,tpr_vals = get_auc(rec_error_csv_path,sim_data_csv_path)
		auc_val_list.append([auc_val])
		fpr_vals_list.append([fpr_vals])
		tpr_vals_list.append([tpr_vals])


	# Needed to fix an error with the auc vals assignment, this probably wouldn't be needed in retrospect:
	auc_val_list_temp = []
	fpr_vals_list_temp = []
	tpr_vals_list_temp = []

	for i,rec_error_csv_path in enumerate(all_rec_error_csv_paths):
		sim_name = get_sim_name(rec_error_csv_path)
		print(sim_name)

		auc_val_list_temp.append([sim_name,auc_val_list[i]])
		fpr_vals_list_temp.append([sim_name,fpr_vals_list[i]])
		tpr_vals_list_temp.append([sim_name,tpr_vals_list[i]])

	auc_val_list = auc_val_list_temp
	fpr_vals_list = fpr_vals_list_temp
	tpr_vals_list = tpr_vals_list_temp

	from copy import deepcopy
	attack_impact_data_with_detect = []
	attack_impact_data = np.loadtxt('i24_random_sample_attack_impacts.csv')
	for i in range(len(attack_impact_data)):
		row = list(attack_impact_data[i])
		mag = row[1]
		dur = row[0]
		attack_impact_data_with_detect.append(row)
		sim_name = get_attack_file_name_from_params(dur,mag)
		for row2 in f1_score_values_k_means:
			if(row2[0] == sim_name): attack_impact_data_with_detect[-1].append(row2[1])
		for row2 in f1_score_values_thershold:
			if(row2[0] == sim_name): attack_impact_data_with_detect[-1].append(row2[1])

	X = np.array(attack_impact_data_with_detect)

	dot_size = 250

	fontsize=25

	fig = pt.figure(figsize=[15,10])
	pt.subplot(1,2,1)
	pt.scatter(X[:,0],X[:,1],c=X[:,2],s=dot_size)
	pt.colorbar()
	pt.ylabel('Attack duration [s]',fontsize=fontsize)
	pt.xlabel('Attack magnitude [m/s^2]',fontsize=fontsize)
	pt.title('Traffic speed average (TSA) [m/s]',fontsize=fontsize)
	pt.xticks(fontsize=fontsize-5)
	pt.yticks(fontsize=fontsize-5)
	# pt.ylabel('')
	pt.subplot(1,2,2)
	pt.scatter(X[:,0],X[:,1],c=X[:,3],s=dot_size)
	pt.ylabel('Attack duration [s]',fontsize=fontsize)
	pt.xlabel('Attack magnitude [m/s^2]',fontsize=fontsize)
	pt.xticks(fontsize=fontsize-5)
	pt.yticks(fontsize=fontsize-5)
	pt.title('Traffic speed variance (TSV) [m/s]',fontsize=fontsize)
	pt.colorbar()
	fig.suptitle('Freeway environemnt attack impacts',fontsize=fontsize+10)
	pt.savefig("/Users/vanderbilt/Desktop/Research_2022/Usenix_2022/figures/in_progress/i24_impact_scatter.png",bbox_inches='tight')
	pt.show()




	attack_impact_data_with_detect = []
	# attack_impact_data = np.loadtxt('i24_random_sample_attack_impacts.csv')
	for i in range(len(attack_impact_data)):
		row = list(attack_impact_data[i])
		mag = row[1]
		dur = row[0]
		attack_impact_data_with_detect.append(row)
		sim_name = get_attack_file_name_from_params(dur,mag)
		for row2 in f1_score_values_k_means:
			if(row2[0] == sim_name): attack_impact_data_with_detect[-1].append(row2[1])
		for row2 in f1_score_values_thershold:
			if(row2[0] == sim_name): attack_impact_data_with_detect[-1].append(row2[1])
		for row2 in auc_val_list:
			if(row2[0] == sim_name): attack_impact_data_with_detect[-1].append(row2[1])

	X = np.array(attack_impact_data_with_detect)
	fig = pt.figure(figsize=[15,20])
	fontsize=25
	pt.subplot(3,1,1)
	pt.scatter(X[:,2],X[:,4],s=dot_size)
	pt.ylim([-0.1,0.1])
	pt.yticks(fontsize=fontsize)
	pt.xticks(fontsize=fontsize)
	pt.ylabel('F1-score',fontsize=fontsize)
	# pt.xlabel('Traffic speed average [TSA] [m/s]',fontsize=fontsize)
	pt.title('K-means relative classification',fontsize=fontsize)

	pt.subplot(3,1,2)
	pt.scatter(X[:,2],X[:,5],s=dot_size)
	pt.ylim([-0.1,0.1])
	pt.yticks(fontsize=fontsize)
	pt.xticks(fontsize=fontsize)
	pt.ylabel('F1-score',fontsize=fontsize)
	# pt.xlabel('Traffic speed average [TSA] [m/s]',fontsize=fontsize)
	pt.title('Threshold based classification',fontsize=fontsize)

	pt.subplot(3,1,3)
	pt.scatter(X[:,2],X[:,6],s=dot_size)
	pt.ylim([0.4,0.6])
	pt.yticks(fontsize=fontsize)
	pt.xticks(fontsize=fontsize)
	pt.ylabel('AUC',fontsize=fontsize)
	pt.xlabel('Traffic speed average [TSA] [m/s]',fontsize=fontsize)
	pt.title('AUC on MvPRE values',fontsize=fontsize)

	fig.suptitle('Freeway environment, attack impact vs. attack stealth',fontsize=fontsize+10)
	pt.savefig("/Users/vanderbilt/Desktop/Research_2022/Usenix_2022/figures/in_progress/i24_impact_vs_stealth.png",bbox_inches='tight')
	pt.show()


	fig = pt.figure()
















	attack_impact_data = 