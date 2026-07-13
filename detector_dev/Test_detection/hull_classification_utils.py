import os
import numpy as np
import flow
from copy import deepcopy
import sys

from sklearn.cluster import KMeans

from Data_Processing.sim_processing_utils import get_trajectory_timeseries

from detector_dev.Process_I24_simulations.i24_utils import get_sim_timeseries as get_sim_timeseries_i24

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

def filter_timeseries_dict_for_length(timeseries_dict,seq_len):
	timeseries_dict_filtered = {}

	for veh_id in timeseries_dict:
		if(len(timeseries_dict[veh_id]) >= seq_len):
			timeseries_dict_filtered[veh_id] = timeseries_dict[veh_id]

	return timeseries_dict_filtered



def load_ring_timeseries(csv_path,warmup_period=0.0):
	trajectory_dict = get_trajectory_timeseries(csv_path,warmup_period=100.0,want_print_finished_loading=False)
	return trajectory_dict

def load_i24_timeseries(csv_pathm,warmup_period,min_traj_length=100):
	timeseries_dict = get_sim_timeseries(csv_path,warmup_period=warmup_period)
	return timeseries_dict



def get_trajectory_points(trajectory_dict,sampling_fidelity=10):
	trajectory_points = []
	for veh_id in trajectory_dict:
		trajectory_data = trajectory_dict[veh_id]
		
		speed = trajectory_data[:,1]
		# accel = np.gradient(speed,.1)
		head_way = trajectory_data[:,2]
		rel_vel = trajectory_data[:,3]
		
		sample_indices = np.arange(0,len(trajectory_data),sampling_fidelity)
		
		for i in sample_indices:
			point = [speed[i],head_way[i],rel_vel[i]]
			trajectory_points.append(point)

	trajectory_points = np.array(trajectory_points)
	return trajectory_points


def get_clustered_points(trajectory_points,num_clusters=2):

	X = trajectory_points
	kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
	labels = kmeans.labels_

	HV_points = X[np.logical_not(labels.astype(bool)),:]

	ACC_points = X[labels.astype(bool),:]

	return ACC_points,HV_points


def expand_hull(hull,points,scaling_factor = 1.5):

	exterior_points = points[hull.vertices,:]

	center = np.mean(exterior_points,axis=0)

	expanded_exterior_points = np.zeros_like(exterior_points)

	for i in range(expanded_exterior_points.shape[0]):
		expanded_exterior_points[i,:] = center + (exterior_points[i,:] - center)*scaling_factor

	expanded_hull =  ConvexHull(expanded_exterior_points)


	return expanded_hull


def in_hull(p, hull):
	"""
	Test if points in `p` are in `hull`

	`p` should be a `NxK` coordinates of `N` points in `K` dimensions
	`hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
	coordinates of `M` points in `K`dimensions for which Delaunay triangulation
	will be computed
	"""
	if not isinstance(hull,Delaunay):
		delaunay_hull = Delaunay(hull.points)

	return delaunay_hull.find_simplex(p)>=0


class hull_detector():
	def __init__(self,training_points,scaling_factor=1.2):
		self.hull = ConvexHull(training_points)
		self.training_points = training_points
		self.scaling_factor = scaling_factor
		self.expanded_hull,self.exterior_points = self.expand_hull()
		self.expanded_delaunay_hull = Delaunay(self.expanded_hull.points)


	def expand_hull(self):

		exterior_points = self.training_points[self.hull.vertices,:]

		center = np.mean(exterior_points,axis=0)

		expanded_exterior_points = np.zeros_like(exterior_points)

		for i in range(expanded_exterior_points.shape[0]):
			expanded_exterior_points[i,:] = center + (exterior_points[i,:] - center)*self.scaling_factor

		expanded_hull =  ConvexHull(expanded_exterior_points)


		return expanded_hull,expanded_exterior_points

	def in_hull(self,P):
		return self.expanded_delaunay_hull.find_simplex(P)>=0


	def classify_trajectory(self,x):
		is_in_hull = True
		for P in x:
			if(not self.in_hull(P)): is_in_hull = False
		return is_in_hull


	def plot_classified_points_in_s_v(self,x):
		points_in_hull = []
		points_outside_hull = []

		for P in x:
			if(not self.in_hull(P)):
				points_outside_hull.append(P)
			else:
				points_in_hull.append(P)

		points_in_hull = np.array(points_in_hull)

		points_outside_hull = np.array(points_outside_hull)

		if(len(points_in_hull)>0):
			plt.plot(points_in_hull[:,0],points_in_hull[:,1],'b.')
		if(len(points_outside_hull) > 0):
			plt.plot(points_outside_hull[:,0],points_outside_hull[:,1],'r.')


	def plot_trajectory_classifications(self,trajectory_dict):
		for veh_id in trajectory_dict:
			x = trajectory_dict[veh_id][:,1:]
			self.plot_classified_points_in_s_v(x)

class hull_detector_multi():
	def __init__(self,hull_detector_list):
		self.hull_detector_list = hull_detector_list


	def in_hull(self,P):
		in_any_hull = True
		for hull_detector in self.hull_detector_list:
				if(not hull_detector.in_hull(P)):
					in_any_hull = False
		return in_any_hull


	def classify_trajectory(self,x):
		is_in_hull = True
		for P in x:
			if(not self.in_hull(P)):
				is_in_hull = False
		return is_in_hull

	def get_classified_points(self,x):
		in_hull_points = []
		outside_hull_points = []
		for P in x:
			if(self.in_hull(P)):
				in_hull_points.append(P)
			else:
				outside_hull_points.append(P)
		in_hull_points = np.array(in_hull_points)
		outside_hull_points = np.array(outside_hull_points)

		return in_hull_points,outside_hull_points



	def plot_classified_points_in_s_v(self,x):
		in_hull_points,outside_hull_points = self.get_classified_points(x)
		plt.plot(in_hull_points[:,:2],'k.')
		plt.plot(outside_hull_points[:,:2],'r.')
		return


def get_trajectory_labels(trajectory_dict,hull_detector):
	veh_ids = list(trajectory_dict.keys())
	labels = []

	for veh_id in veh_ids:
		x = np.array(trajectory_dict[veh_id])[:,1:]
		labels.append([veh_id,hull_detector.classify_trajectory(x)])

	return labels


def get_ring_hull_detector(scaling_factor=1.5,sampling_fidelity=1):

	benign_ring_data_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/benign_single_lane_ring/Benign_single_lane_sim_for_attack_MC.csv'

	ring_trajectory_dict = load_ring_timeseries(benign_ring_data_path,warmup_period = 100)

	trajectory_points = get_trajectory_points(ring_trajectory_dict,sampling_fidelity)

	ring_hull_detector = hull_detector(trajectory_points,scaling_factor=scaling_factor)

	return ring_hull_detector










