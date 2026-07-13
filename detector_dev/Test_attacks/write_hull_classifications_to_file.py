import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv
from detector_dev.Test_detection.hull_classification_utils import *
from Data_Processing.sim_processing_utils import get_trajectory_timeseries


def get_anom_veh_ids(trajectory_dict,hull_detector):
	labels = []
	for veh_id in trajectory_dict:
		x = trajectory_dict[veh_id][:,1:]
		if(not hull_detector.classify_trajectory(x)):
			labels.append([veh_id,1])
		else:
			labels.append([veh_id,0])

	return labels

def write_anom_veh_ids(labels,file_name):
	with open(file_name, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',')
		for row in labels:
			csv_writer.writerow(row)
	return

def write_all_detection_results(sim_repo_path,hull_labels_repo_path):

	ring_hull_detector = get_ring_hull_detector()

	files = os.listdir(sim_repo_path)

	all_sim_files = []

	for file in files:
		if('.csv' in file):
			all_sim_files.append(file)

	print('Getting hull detector labels.: '+sim_repo_path)

	for file in tqdm(all_sim_files):
		sim_file_path = os.path.join(sim_repo_path,file)
		trajectory_dict = get_trajectory_timeseries(sim_file_path,warmup_period = 100,want_print_finished_loading=False)

		veh_ids = list(trajectory_dict.keys())
		for veh_id in veh_ids:
			v = np.array(trajectory_dict[veh_id][:,1])
			if(np.min(v) < -1.0):
				print('Collision present: '+file)
	

		labels = get_anom_veh_ids(trajectory_dict,ring_hull_detector)

		write_file_path = os.path.join(hull_labels_repo_path,file)

		write_anom_veh_ids(labels=labels,file_name=write_file_path)

	print('All labels written to file.')

	return






if __name__ == '__main__':

	get_max_velocity_hull_labels = False

	if(get_max_velocity_hull_labels):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_monte_carlo'

		hull_labels_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_monte_carlo_hull_labels'

		write_all_detection_results(sim_repo_path,hull_labels_repo_path)

	get_Radar_warp_hull_labels = False

	if(get_Radar_warp_hull_labels):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_monte_carlo'

		hull_labels_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/Radar_warp/single_lane_ring_road_attack_monte_carlo_hull_labels'

		write_all_detection_results(sim_repo_path,hull_labels_repo_path)

	get_RDA_hull_labels = False

	if(get_RDA_hull_labels):

		sim_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo'

		hull_labels_repo_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo_hull_labels'


		write_all_detection_results(sim_repo_path,hull_labels_repo_path)











