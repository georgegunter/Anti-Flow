import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

from detector_dev.Process_RingRoad_Simulation.process_ring_attacks import get_attack_params as get_attack_params_ring


def get_identifier(file_name):
	i = 0
	while(file_name[i:i+3] != 'TAD'): i+=1
	j = i
	while(file_name[j:j+4] != '_ver'): j+=1

	identifier = file_name[i:j]

	return identifier


def reject_outliers(data, m=1.5):
	return data[abs(data - np.median(data)) < m * np.std(data)]

def is_outlier(data,sample,m=1.5):
	return abs(sample - np.median(data)) > m * np.std(data)

def get_outlier_indices(data,m=1.5):
	num_samples = len(data)
	outlier_indices = []
	for i in range(num_samples):
		if(is_outlier(data,data[i],m=m)):
			outlier_indices.append(i)

	return outlier_indices

def get_all_outlier_indices(data_np_array):
	num_samples = len(data_np_array[:,0])
	num_data_types = len(data_np_array[0,:])

	outlier_indices = []

	for i in range(num_data_types):
		data = data_np_array[:,i]
		outlier_indices = outlier_indices + get_outlier_indices(data)

	return np.unique(outlier_indices)


if __name__ == '__main__':
	
	impacts_list = np.loadtxt('performance_metrics_double_lane.csv',delimiter=',',dtype=str)


	impacts_sorted_by_attack = {}

	for impact in impacts_list:
		file_name = impact[0]
		identifier = get_identifier(file_name)
		if(identifier not in impacts_sorted_by_attack):
			impacts_sorted_by_attack[identifier] = []

		impacts_sorted_by_attack[identifier].append(impact[1:])




	impacts_sorted_by_attack_no_outlier_filter = {}
	for impact in impacts_list:
		file_name = impact[0]
		identifier = get_identifier(file_name)
		if(identifier not in impacts_sorted_by_attack_no_outlier_filter):
			impacts_sorted_by_attack_no_outlier_filter[identifier] = []

		impacts_sorted_by_attack_no_outlier_filter [identifier].append(impact[1:])


	impacts_sorted_by_attack_temp = dict.fromkeys(impacts_sorted_by_attack)

	for key in impacts_sorted_by_attack:
		impact_data = impacts_sorted_by_attack[key]

		data_np_array = np.array(impact_data).astype(float)

		outlier_indices = list(get_all_outlier_indices(data_np_array))

		data_to_include = []

		for i in range(len(data_np_array[:,0])):

			if(i not in outlier_indices):

				data_to_include.append(data_np_array[i,:])

		impacts_sorted_by_attack_temp[key] = np.array(data_to_include)



	impacts_sorted_by_attack = impacts_sorted_by_attack_temp

	mean_impacts_across_sims = {}

	for key in impacts_sorted_by_attack:
		data = np.array(impacts_sorted_by_attack[key]).astype(float)
		mean_impacts_across_sims[key] = np.mean(data,axis = 0)    


	durations = []
	magnitudes = []
	MTS_vals = []
	TSV_vals = []
	Mean_MinTG_vals = []
	Min_MinTG_vals = []
	Mean_MinTTC_vals = []
	Min_MinTTC_vals = []


	base_line_MTS = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][0]
	base_line_TSV = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][1]
	base_line_Mean_MinTG = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][2]
	base_line_Min_MinTG = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][3]
	base_line_Mean_MinTTC = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][4]
	base_line_Min_MinTTC = mean_impacts_across_sims['TAD_0.0_ADR_0.0'][5]

	for key in mean_impacts_across_sims:
		attack_params = get_attack_params_ring('_'+key+'_')
		durations.append(attack_params[0])
		magnitudes.append(attack_params[1])
		MTS_vals.append(mean_impacts_across_sims[key][0])
		TSV_vals.append(mean_impacts_across_sims[key][1])
		Mean_MinTG_vals.append(mean_impacts_across_sims[key][2])
		Min_MinTG_vals.append(mean_impacts_across_sims[key][3])
		Mean_MinTTC_vals.append(mean_impacts_across_sims[key][4])
		Min_MinTTC_vals.append(mean_impacts_across_sims[key][5])

	MTS_vals_percent_change = ((np.array(MTS_vals)-base_line_MTS)/base_line_MTS)*100

	TSV_vals_percent_change = ((np.array(TSV_vals)-base_line_TSV)/base_line_TSV)*100

	Mean_MinTG_vals_percent_change = ((np.array(Mean_MinTG_vals)-base_line_Mean_MinTG)/base_line_Mean_MinTG)*100

	Mean_MinTTC_vals_percent_change = ((np.array(Mean_MinTTC_vals)-base_line_Mean_MinTTC)/base_line_Mean_MinTTC)*100

	Min_MinTTC_vals_percent_change = ((np.array(Min_MinTTC_vals)-base_line_Min_MinTTC)/base_line_Min_MinTTC)*100

    # Final figure:

	fig = plt.figure(figsize=[10,15])
	plt.subplot(3,1,1)
	plt.scatter(durations,magnitudes,c=MTS_vals_percent_change,s=500)
	plt.colorbar()
	plt.ylabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	# plt.xlabel('Duration [s]',fontsize=20)
	plt.title('MTS % change',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)

	plt.subplot(3,1,2)
	plt.scatter(durations,magnitudes,c=Min_MinTTC_vals_percent_change,s=500)
	plt.colorbar()
	plt.clim([-70,10])
	plt.ylabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	# plt.xlabel('Duration [s]',fontsize=20)
	plt.title('M-TTC % change',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)

	plt.subplot(3,1,3)
	plt.scatter(durations,magnitudes,c=TSV_vals_percent_change,s=500)
	plt.colorbar()
	plt.clim([-15,40])
	plt.ylabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.xlabel('Duration [s]',fontsize=20)
	plt.title('TSV % change',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)

	fig.suptitle('Attack impacts on double-lane ring',fontsize=25)

	plt.savefig("double_lane_RDA_parameter_sweep_impacts.png",bbox_inches='tight')






	fig = plt.figure(figsize=[10,13])
	plt.subplot(3,2,1)
	plt.scatter(durations,MTS_vals,s=20)
	plt.ylabel('MTS [m/s]',fontsize=20)
	# plt.xlabel('Duration [s]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(durations,MTS_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)

	plt.subplot(3,2,3)
	plt.scatter(durations,Min_MinTTC_vals,s=20)
	plt.ylabel('M-TTC [m/s]',fontsize=20)
	# plt.xlabel('Duration [s]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(durations,Min_MinTTC_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)

	plt.subplot(3,2,5)
	plt.scatter(durations,TSV_vals,s=20)
	plt.ylabel('TSV [m/s]',fontsize=20)
	plt.xlabel('Duration [s]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(durations,TSV_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)


	plt.subplot(3,2,2)
	plt.scatter(magnitudes,MTS_vals,s=20)
	# plt.ylabel('MTS [m/s]',fontsize=20)
	plt.yticks(fontsize=15)
	# plt.xlabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(magnitudes,MTS_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)

	plt.subplot(3,2,4)
	plt.scatter(magnitudes,Min_MinTTC_vals,s=20)
	# plt.ylabel('M-TTC [s]',fontsize=20)
	# plt.xlabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(magnitudes,Min_MinTTC_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)

	plt.subplot(3,2,6)
	plt.scatter(magnitudes,TSV_vals,s=20)
	# plt.ylabel('M-TTC [s]',fontsize=20)
	plt.xlabel(r'Magnitude $[\frac{m}{s^{2}}]$',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	res = stats.spearmanr(magnitudes,TSV_vals)
	plt.title('Spearman:'+str(np.round(res.correlation,2)),fontsize=20)

	fig.suptitle('2 lane RR',fontsize=25)

	plt.savefig("Double_lane_RDA_parameter_sweep_correlations.png",bbox_inches='tight')





