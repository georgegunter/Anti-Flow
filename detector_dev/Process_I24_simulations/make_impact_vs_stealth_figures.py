import numpy as np
import matplotlib.pyplot as plt

import os

from detector_dev.Process_I24_simulations.utils_classification import *

from detector_dev.Process_I24_simulations.i24_utils import get_sim_timeseries as get_sim_timeseries_i24
from detector_dev.Process_I24_simulations.i24_utils import get_attack_params as get_attack_params_i24

CLUSTER_DIFF = 0.05

FIGURE_SIZE = [35,10]
TITLE_FONTSIZE = 40
LABEL_FONTSIZE = 40
TICK_FONTSIZE = 35


def get_f1_from_max_thresh(true_labels,max_rec_errors,thresh=7.481976509094238):
	assigned_labels_list = max_rec_errors > thresh

	f1_score = f1_calculation(assigned_labels_list=assigned_labels_list,vehicle_labels_list=true_labels)
	return f1_score, assigned_labels_list

def get_f1_from_kmeans(true_labels,max_rec_errors):

	assigned_labels_list,_,_ = k_means_cluster(max_losses=max_rec_errors,
		true_labels=true_labels,
		cluster_diff=CLUSTER_DIFF)

	f1_score = f1_calculation(assigned_labels_list=assigned_labels_list,vehicle_labels_list=true_labels)

	return f1_score,assigned_labels_list

def get_label_performance_rates(true_labels,assigned_labels):
	num_labels = len(true_labels)
	num_false_positives = 0
	num_true_positives = 0
	num_false_negatives = 0
	num_true_negatives = 0


	for i in range(num_labels):
		correct_label = true_labels[i]
		assigned_label = assigned_labels[i]

		assigned_positive = (assigned_label == 1)
		assigned_negative = (assigned_label == 0)

		is_negative = (correct_label == 0)
		is_positive = (correct_label == 1)

		if(assigned_positive):
			if(is_negative):
				num_false_positives += 1
			else:
				num_true_positives += 1
		else:
			if(is_negative):
				num_true_negatives += 1
			else:
				num_false_negatives += 1

	return num_false_positives,num_false_negatives,num_true_positives,num_true_negatives


if __name__ == '__main__':

	loss_repo_path = '/Users/vanderbilt/Desktop/General_research_tools/Anti-Flow/detector_dev/Process_I24_simulations/normalized_detection_classification_results/max_rec_errors'

	benign_max_rec_errors_file = os.path.join(loss_repo_path,'max_rec_errors_Dur_0.0027839584656419447_Mag_-1.1173175105601907_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.1_ver_1.csv')

	benign_max_rec_errors = np.loadtxt(benign_max_rec_errors_file,delimiter=',')

	max_benign_rec_error = np.max(benign_max_rec_errors[:,1])*1.1 #7.482


	#### Get all other max rec errors: ####

	all_files = os.listdir(loss_repo_path)

	all_max_rec_error_files = []

	for file in all_files:
		if('csv' in file):
			if(file != benign_max_rec_errors_file):
				all_max_rec_error_files.append(os.path.join(loss_repo_path,file))


	########## LOAD RECONSTRUCITON ERROR DATA: ##########

	durations = []
	magnitudes = []
	f1_thresh = []
	f1_kmeans = []

	FP_number_thresh = []
	FN_number_thresh = []
	TP_number_thresh = []
	TN_number_thresh = []

	FP_number_kmeans = []
	FN_number_kmeans = []
	TP_number_kmeans = []
	TN_number_kmeans = []


	max_RE_list = []

	true_labels_list = []

	all_labels_thresh = []

	all_labels_kmeans = []


	for file in all_max_rec_error_files:

		max_rec_error_data = np.loadtxt(file,delimiter=',')
		true_labels = max_rec_error_data[:,0]
		max_rec_errors = max_rec_error_data[:,1]

		max_RE_list.append(max_rec_errors)

		true_labels_list.append(true_labels)

		f1_score_thresh,assigned_labels_list_thresh = get_f1_from_max_thresh(true_labels,max_rec_errors)

		f1_score_kmeans,assigned_labels_list_kmeans = get_f1_from_kmeans(true_labels,max_rec_errors)


		all_labels_thresh.append(assigned_labels_list_thresh)

		all_labels_kmeans.append(assigned_labels_list_kmeans)


		n_fp_thresh,n_fn_thresh,n_tp_thresh,n_tn_thresh = get_label_performance_rates(true_labels=true_labels,
			assigned_labels=assigned_labels_list_thresh)

		n_fp_kmeans,n_fn_kmeans,n_tp_kmeans,n_tn_kmeans = get_label_performance_rates(true_labels=true_labels,
			assigned_labels=assigned_labels_list_kmeans)


		FP_number_thresh.append(n_fp_thresh)
		FN_number_thresh.append(n_fn_thresh)
		TP_number_thresh.append(n_tp_thresh)
		TN_number_thresh.append(n_tn_thresh)

		FP_number_kmeans.append(n_fp_kmeans)
		FN_number_kmeans.append(n_fn_kmeans)
		TP_number_kmeans.append(n_tp_kmeans)
		TN_number_kmeans.append(n_tn_kmeans)

		attack_params = get_attack_params_i24(file)

		durations.append(attack_params[0])
		magnitudes.append(attack_params[1])
		f1_thresh.append(f1_score_thresh)
		f1_kmeans.append(f1_score_kmeans)


	########## GET IMPACT DATA: ########## 


	attack_data_csv_path = '/Users/vanderbilt/Desktop/General_research_tools/Anti-Flow/detector_dev/Process_I24_simulations/attack_impact_param_sample.csv'

	impact_data = np.loadtxt(attack_data_csv_path,delimiter=',')

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


	print('Making figures.')


	########## GET AUC scores: ########## 

	from sklearn.metrics import roc_auc_score

	AUC_vals = []

	for i in range(len(true_labels_list)):
		AUC_vals.append(roc_auc_score(true_labels_list[i],max_RE_list[i]))


	########## GET AUC scores: ########## 

	# Calculate recall, precision, accuracy:

	from sklearn.metrics import recall_score,accuracy_score,precision_score

	recall_vals_thresh = []

	precision_vals_thresh = []

	accuracy_vals_thresh = []

	for i in range(len(all_labels_thresh)):
		y_pred = all_labels_thresh[i]
		y_true = true_labels_list[i]

		recall_vals_thresh.append(recall_score(y_true, y_pred))
		precision_vals_thresh.append(precision_score(y_true, y_pred))
		accuracy_vals_thresh.append(accuracy_score(y_true, y_pred))


	recall_vals_kmeans = []

	precision_vals_kmeans = []

	accuracy_vals_kmeans = []

	for i in range(len(all_labels_kmeans)):
		y_pred = all_labels_kmeans[i]
		y_true = true_labels_list[i]

		recall_vals_kmeans.append(recall_score(y_true, y_pred))
		precision_vals_kmeans.append(precision_score(y_true, y_pred))
		accuracy_vals_kmeans.append(accuracy_score(y_true, y_pred))


	########## MAKE FIGURES: ##########

	figure_repo_path = '/Users/vanderbilt/Desktop/Research_2024/Usenix 2024/Figures/'

	want_plot_F1_scatter = False
	if(want_plot_F1_scatter):

		fig = plt.figure(figsize=FIGURE_SIZE)
		plt.subplot(2,2,1)
		plt.plot(MTS_vals,f1_kmeans,'.',markersize=20)
		# plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('F1 score k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,2)
		plt.plot(TSV_vals,f1_kmeans,'.',markersize=20)
		# plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		plt.subplot(2,2,3)
		plt.plot(MTS_vals,f1_thresh,'.',markersize=20)
		plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('F1 score thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,4)
		plt.plot(TSV_vals,f1_thresh,'.',markersize=20)
		plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		fig.suptitle('MLFW: F1 (stealth) vs. attack impact',fontsize=25)
		plt.savefig(figure_repo_path+"MLFW_impact_vs_F1.png",bbox_inches='tight')


	want_plot_FP_num_scatter = False
	if(want_plot_FP_num_scatter):
		fig = plt.figure(figsize=FIGURE_SIZE)
		plt.subplot(2,2,1)
		plt.plot(MTS_vals,FP_number_kmeans,'.',markersize=20)
		# plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('Number FPS: k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,2)
		plt.plot(TSV_vals,FP_number_kmeans,'.',markersize=20)
		# plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		plt.subplot(2,2,3)
		plt.plot(MTS_vals,FP_number_thresh,'.',markersize=20)
		plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('Number FPS: thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,4)
		plt.plot(TSV_vals,FP_number_thresh,'.',markersize=20)
		plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		fig.suptitle('MLFW: Number FPs (stealth) vs. attack impact',fontsize=25)
		plt.savefig(figure_repo_path+"MLFW_impact_vs_FPs.png",bbox_inches='tight')

	want_plot_FN_num_scatter = False
	if(want_plot_FN_num_scatter):
		fig = plt.figure(figsize=FIGURE_SIZE)
		plt.subplot(2,2,1)
		plt.plot(MTS_vals,FN_number_kmeans,'.',markersize=20)
		# plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('Number FNS: k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,2)
		plt.plot(TSV_vals,FN_number_kmeans,'.',markersize=20)
		# plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score k-means',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		plt.subplot(2,2,3)
		plt.plot(MTS_vals,FN_number_thresh,'.',markersize=20)
		plt.xlabel('MTS [m/s]',fontsize=TICK_FONTSIZE)
		plt.ylabel('Number FNS: thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)
		plt.subplot(2,2,4)
		plt.plot(TSV_vals,FN_number_thresh,'.',markersize=20)
		plt.xlabel('TSV [m/s]',fontsize=TICK_FONTSIZE)
		# plt.ylabel('F1 score thresh',fontsize=TICK_FONTSIZE)
		plt.yticks(fontsize=15)
		plt.xticks(fontsize=15)

		fig.suptitle('MLFW: Number FNs (stealth) vs. attack impact',fontsize=25)
		plt.savefig(figure_repo_path+"MLFW_impact_vs_FNs.png",bbox_inches='tight')

	########## SUBFIGURE WITH ALL RESULTS FOR THRESH: ##########

	want_all_classification_res = True
	if(want_all_classification_res):

		fig = plt.figure(figsize=[45,20])

		plt.subplot(2,3,1)
		plt.plot(MTS_vals,f1_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,f1_thresh,'*',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('F1',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)
		_ = plt.ylim([-0.1,1])

		plt.subplot(2,3,2)
		plt.plot(MTS_vals,FP_number_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,FP_number_thresh,'*',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('False positives',fontsize=LABEL_FONTSIZE+5)
		_ = plt.locator_params(axis='y', nbins=8)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)
		_ = plt.legend(fontsize=LABEL_FONTSIZE+5)

		plt.subplot(2,3,3)
		plt.plot(MTS_vals,TP_number_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,TP_number_thresh,'*',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('True Positives',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.subplot(2,3,4)
		plt.plot(MTS_vals,recall_vals_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,recall_vals_thresh,'*',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Recall',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)


		plt.subplot(2,3,5)
		plt.plot(MTS_vals,precision_vals_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,precision_vals_thresh,'*',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Precision',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.subplot(2,3,6)
		plt.plot(MTS_vals,accuracy_vals_kmeans,'.',markersize=30,label='Relative')
		plt.plot(MTS_vals,accuracy_vals_thresh,'*',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Accuracy',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.savefig(figure_repo_path+"MLFW_impact_vs_stealth_all_metrics.png",bbox_inches='tight')




	want_all_classification_res_thresh_only = True
	if(want_all_classification_res_thresh_only):

		fig = plt.figure(figsize=[45,20])

		plt.subplot(2,3,1)
		plt.plot(MTS_vals,f1_thresh,'.',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('F1',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)
		_ = plt.ylim([-0.1,1])

		plt.subplot(2,3,2)
		plt.plot(MTS_vals,FP_number_thresh,'.',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('False positives',fontsize=LABEL_FONTSIZE+5)
		_ = plt.locator_params(axis='y', nbins=8)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.subplot(2,3,3)
		plt.plot(MTS_vals,TP_number_thresh,'.',markersize=30,label='Thresholding')
		# _ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
		_ = plt.ylabel('True Positives',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.subplot(2,3,4)
		plt.plot(MTS_vals,recall_vals_thresh,'.',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Recall',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)


		plt.subplot(2,3,5)
		plt.plot(MTS_vals,precision_vals_thresh,'.',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Precision',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.subplot(2,3,6)
		plt.plot(MTS_vals,accuracy_vals_thresh,'.',markersize=30,label='Thresholding')
		_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE+5)
		_ = plt.ylabel('Accuracy',fontsize=LABEL_FONTSIZE+5)
		_ = plt.yticks(fontsize=TICK_FONTSIZE)
		_ = plt.xticks(fontsize=TICK_FONTSIZE)

		plt.savefig(figure_repo_path+"MLFW_impact_vs_stealth_thresh_only_all_metrics.png",bbox_inches='tight')





	########## SCATTER OF RELATIONSHIP BETWEEN IMPACTS: ##########

	plt.figure(figsize=FIGURE_SIZE)
	_ = plt.grid()
	plt.scatter(MTS_vals,TSV_vals,s=250)
	_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
	_ = plt.ylabel('TSV [m/s]',fontsize=LABEL_FONTSIZE)
	_ = plt.yticks(fontsize=TICK_FONTSIZE)
	_ = plt.xticks(fontsize=TICK_FONTSIZE)
	_ = plt.ylim([0,8.0])
	plt.savefig(figure_repo_path+"MLFW_impact_scatter.png",bbox_inches='tight')

	########## SUBFIGURE AUC SCORES: ##########

	plt.figure(figsize=FIGURE_SIZE)
	plt.subplot(1,2,1)
	plt.plot(MTS_vals,AUC_vals,'.',markersize=25)
	_ = plt.xlabel('MTS [m/s]',fontsize=LABEL_FONTSIZE)
	_ = plt.ylabel('AUC',fontsize=LABEL_FONTSIZE)
	_ = plt.ylim([0.5,1.0])
	_ = plt.yticks(fontsize=TICK_FONTSIZE)
	_ = plt.xticks(fontsize=TICK_FONTSIZE)
	plt.grid()
	plt.subplot(1,2,2)
	plt.plot(TSV_vals,AUC_vals,'.',markersize=25)
	_ = plt.xlabel('TSV [m/s]',fontsize=LABEL_FONTSIZE)
	_ = plt.yticks(fontsize=TICK_FONTSIZE)
	_ = plt.xticks(fontsize=TICK_FONTSIZE)
	_ = plt.ylim([0.5,1.0])
	_ = plt.grid()
	plt.savefig(figure_repo_path+"AUC_vs_impact.png",bbox_inches='tight')

	plt.figure(figsize=FIGURE_SIZE)
	plt.subplot(1,2,1)
	plt.plot(magnitudes,AUC_vals,'.',markersize=25)
	_ = plt.xlabel('Attack duration [s]',fontsize=LABEL_FONTSIZE)
	_ = plt.ylabel('AUC',fontsize=LABEL_FONTSIZE)
	_ = plt.ylim([0.5,1.0])
	_ = plt.yticks(fontsize=TICK_FONTSIZE)
	_ = plt.xticks(fontsize=TICK_FONTSIZE)
	plt.grid()
	plt.subplot(1,2,2)
	plt.plot(durations,AUC_vals,'.',markersize=25)
	_ = plt.xlabel('Attack decel [m/s^2]',fontsize=LABEL_FONTSIZE)
	_ = plt.yticks(fontsize=TICK_FONTSIZE)
	_ = plt.xticks(fontsize=TICK_FONTSIZE)
	_ = plt.ylim([0.5,1.0])
	_ = plt.grid()
	plt.savefig(figure_repo_path+"AUC_vs_attack_params.png",bbox_inches='tight')

	print('Saved all figures.')






