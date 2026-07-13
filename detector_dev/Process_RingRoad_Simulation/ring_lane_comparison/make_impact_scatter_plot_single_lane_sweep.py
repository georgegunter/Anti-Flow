import numpy as np
import matplotlib.pyplot as plt


from detector_dev.Process_RingRoad_Simulation.process_ring_attacks import get_attack_params as get_attack_params_ring


def get_identifier(file_name):

	i = 0
	while(file_name[i] != '_'): i +=1

	j = i
	while(file_name[j] != 'm'): j+=1

	ring_length = file_name[i+1:j]

	while(file_name[i:i+3] != 'TAD'): i+=1
	j = i
	while(file_name[j:j+4] != '_ver'): j+=1

	identifier = ring_length+'_'+file_name[i:j]

	return identifier


def reject_outliers(data, m=1.5):
	return data[abs(data - np.median(data)) < m * np.std(data)]



def magnitude_too_large(sample,threshold=1e4):
	return sample > threshold


def is_outlier(data,sample,m=1.5):

	reject_is_outlier = abs(sample - np.median(data)) > m * np.std(data)

	return  reject_is_outlier

def get_outlier_indices(data,m=1.5,threshold=1e4):

	num_samples = len(data)

	outlier_indices = []
	in_distribution_indices = []

	for i in range(num_samples):
		if(magnitude_too_large(data[i],threshold=threshold)):
			outlier_indices.append(i)
		else:
			in_distribution_indices.append(i)


	data_filtered = data[in_distribution_indices]


	for i in range(len(in_distribution_indices)):
		if(is_outlier(data_filtered,data_filtered[i],m=m)):
			outlier_indices.append(in_distribution_indices[i])


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
	
	impacts_list = np.loadtxt('performance_metrics_single_lane_ring_road_sweep_ring_length.csv',
		delimiter=',',
		dtype=str)


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


	ring_lengths = [600,700,800,900,1000,1100,1200]

	impacts_by_ring_length = dict.fromkeys(ring_lengths)

	for ring_length in ring_lengths:
		durations = []
		magnitudes = []
		MTS_vals = []
		TSV_vals = []
		Min_MinTTC_vals = []

		for key in mean_impacts_across_sims:
			if(str(ring_length) in key):
				attack_params = get_attack_params_ring('_'+key+'_')
				durations.append(attack_params[0])
				magnitudes.append(attack_params[1])
				MTS_vals.append(mean_impacts_across_sims[key][0])
				TSV_vals.append(mean_impacts_across_sims[key][1])
				Min_MinTTC_vals.append(mean_impacts_across_sims[key][5])

		impacts_by_ring_length[ring_length] = [durations,magnitudes,MTS_vals,TSV_vals,Min_MinTTC_vals]




	no_attack_MTS = []
	weak_attack_MTS = []
	strong_attack_MTS = []

	no_attack_TSV = []
	weak_attack_TSV = []
	strong_attack_TSV = []

	no_attack_MTTC = []
	weak_attack_MTTC = []
	strong_attack_MTTC = []

	for ring_length in ring_lengths:
		impact_data = impacts_by_ring_length[ring_length]

		MTS_vals = impact_data[2]
		TSV_vals = impact_data[3]
		MTTC_vals = impact_data[4]


		# baseline_MTS_vals

		for i in range(3):
			if(impact_data[0][i]==0.0):
				no_attack_MTS.append(MTS_vals[i])
				no_attack_TSV.append(TSV_vals[i])
				no_attack_MTTC.append(MTTC_vals[i])

			elif(impact_data[0][i]==5.0):
				weak_attack_MTS.append(MTS_vals[i])
				weak_attack_TSV.append(TSV_vals[i])
				weak_attack_MTTC.append(MTTC_vals[i])

			elif(impact_data[0][i]==10.0):
				strong_attack_MTS.append(MTS_vals[i])
				strong_attack_TSV.append(TSV_vals[i])
				strong_attack_MTTC.append(MTTC_vals[i])

	fig = plt.figure(figsize=[10,15])

	plt.subplot(3,1,1)
	plt.plot(ring_lengths,no_attack_MTS,'b-o',markersize=20)
	plt.plot(ring_lengths,weak_attack_MTS,'k-o',markersize=20)
	plt.plot(ring_lengths,strong_attack_MTS,'r-o',markersize=20)
	plt.ylabel('MTS [m/s]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)


	plt.subplot(3,1,2)
	plt.plot(ring_lengths,no_attack_MTTC,'b-o',markersize=20)
	plt.plot(ring_lengths,weak_attack_MTTC,'k-o',markersize=20)
	plt.plot(ring_lengths,strong_attack_MTTC,'r-o',markersize=20)
	plt.ylabel('M-TTC [s]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)

	plt.subplot(3,1,3)
	plt.plot(ring_lengths,no_attack_TSV,'b-o',label='No attack',markersize=20)
	plt.plot(ring_lengths,weak_attack_TSV,'k-o',label='Weak attack',markersize=20)
	plt.plot(ring_lengths,strong_attack_TSV,'r-o',label='Strong attack',markersize=20)
	plt.ylabel('TSV [m/s]',fontsize=20)
	plt.xlabel('Ring length [m]',fontsize=20)
	plt.yticks(fontsize=15)
	plt.xticks(fontsize=15)
	plt.legend(fontsize=15)

	fig.suptitle('Effect of decreased traffic density',fontsize=25)

	plt.savefig("ring_length_sweep_impacts.png",bbox_inches='tight')



			# 	plt.subplot(3,1,1)
			# 	plt.plot(ring_length,MTS_vals[i],'r.',markersize=20)
			# 	plt.subplot(3,1,2)
			# 	plt.plot(ring_length,TSV_vals[i],'r.',markersize=20)
			# 	plt.subplot(3,1,3)
			# 	plt.plot(ring_length,Min_MinTTC_vals[i],'r.',markersize=20)
			# elif(impact_data[0][i]==10.0):
			# 	plt.subplot(3,1,1)
			# 	plt.plot(ring_length,MTS_vals[i],'k.',markersize=20)
			# 	plt.subplot(3,1,2)
			# 	plt.plot(ring_length,TSV_vals[i],'k.',markersize=20)
			# 	plt.subplot(3,1,3)
			# 	plt.plot(ring_length,Min_MinTTC_vals[i],'k.',markersize=20)










