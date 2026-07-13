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
			point = [head_way[i],speed[i],rel_vel[i]]
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



def plot_convex_hull(hull,points):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

	for ax in (ax1, ax2):
		ax.plot(points[:, 0], points[:, 1], '.', color='k')
		if ax == ax1:
			ax.set_title('Given points')
		else:
			ax.set_title('Convex hull')
			for simplex in hull.simplices:
				ax.plot(points[simplex, 0], points[simplex, 1], 'c')
			ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)

	plt.show()


def plot_convex_hull_new(hull,points):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

	for ax in (ax1, ax2):
		ax.plot(points[:, 0], points[:, 1], '.', color='k')
		if ax == ax1:
			ax.set_title('Given points')
		else:
			ax.set_title('Convex hull')
			ax.plot(hull.points[:,0],hull.points[:,1], '-o', mec='r', color='none', lw=1, markersize=10)

	plt.show()



def plot_convex_hull_comparison(ACC_hull,ACC_points,HV_hull,HV_points):

	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

	for ax in (ax1, ax2):
		
		ax.plot(ACC_points[:, 0], ACC_points[:, 1], '.', color='k')

		ax.plot(HV_points[:, 0], HV_points[:, 1], '.', color='k')


		if ax == ax1:
			ax.set_title('Given points')
		else:
			ax.set_title('Convex hulls')

			plt.plot(ACC_hull.points[:,0],ACC_hull.points[:,1],'r',linewidth=3,markersize=10)

			plt.plot(HV_hull.points[:,0],HV_hull.points[:,1],'b',linewidth=3,markersize=10)
	
	plt.show()





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
			expanded_exterior_points[i,:] = center + (exterior_points[i,:] - center)*scaling_factor

		expanded_hull =  ConvexHull(expanded_exterior_points)


		return expanded_hull,expanded_exterior_points

	def in_hull(self,P):
		return self.expanded_delaunay_hull.find_simplex(P)>=0


	def classify_trajectory(self,x):
		return np.sum(self.in_hull(x))


	#### Initial testing: ####

	# # Get initial convex hulls:

	# ACC_hull = ConvexHull(ACC_points[:,:2])

	# HV_hull = ConvexHull(HV_points[:,:2])


	# # get expanded convex hulls:

	# ACC_hull_expanded = expand_hull(ACC_hull,ACC_points[:,:2],scaling_factor = 1.5)



if __name__ == '__main__':
	

	# benign_ring_data_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/benign_single_lane_ring/ring_single_lane_length_600_ver_1.csv'


	benign_ring_data_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo/Benign_sim.csv'

	ring_trajectory_dict = load_ring_timeseries(benign_ring_data_path)

	trajectory_points = get_trajectory_points(ring_trajectory_dict,1)

	ACC_points,HV_points = get_clustered_points(trajectory_points)

	ACC_hull_detector = hull_detector(ACC_points[:,:2])

	HV_hull_detector = hull_detector(HV_points[:,:2])



	# get some attacked sim data:

	
	attack_ring_data_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo/RDA_AD_0.5053303779798934_AF_27.195612148834872_DR_-0.7601416058948469_ver_1.csv'


	# attack_ring_data_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/single_lane_ring_road_attack_parameter_sweep/ring_600m_single_lane_TAD_8.0_vmcomp_3.0_ver_5.csv'

	ring_attack_trajectory_dict = load_ring_timeseries(attack_ring_data_path)

	attack_trajectory_points = get_trajectory_points(ring_attack_trajectory_dict,1)










	# HV_hull_expanded = expand_hull(HV_hull,HV_points[:,:2],scaling_factor = 1.5)




	# Compare benign and attacked:



	want_make_hull_cpmparison_figure = True

	if(want_make_hull_cpmparison_figure):

		HV_exterior = HV_hull_detector.exterior_points

		ACC_exterior = ACC_hull_detector.exterior_points


		plt.figure(figsize=[20,8])

		plt.subplot(1,2,1)
		plt.grid()
		plt.plot(trajectory_points[:,0],trajectory_points[:,1],'k.',label='trajectory points')

		plt.plot(HV_exterior[:,0],HV_exterior[:,1],'b')
		plt.plot([HV_exterior[-1,0],HV_exterior[0,0]],[HV_exterior[-1,1],HV_exterior[0,1]],'b',label=r'$\mathcal{CH}_{H}$')

		plt.plot(ACC_exterior[:,0],ACC_exterior[:,1],'g')
		plt.plot([ACC_exterior[-1,0],ACC_exterior[0,0]],[ACC_exterior[-1,1],ACC_exterior[0,1]],'g',label=r'$\mathcal{CH}_{ACC}$')

		plt.legend(fontsize=20)
		plt.ylabel('Speed [m/s]',fontsize=20)
		plt.xlabel('Spacing [m]',fontsize=20)
		plt.yticks(fontsize=20)
		plt.xticks(fontsize=20)
		plt.title('Benign',fontsize=20)

		_ = plt.ylim([0,15])
		_ = plt.xlim([0,35])

		plt.subplot(1,2,2)
		plt.grid()
		plt.plot(attack_trajectory_points[:,0],attack_trajectory_points[:,1],'k.')

		plt.plot(HV_exterior[:,0],HV_exterior[:,1],'b')
		plt.plot([HV_exterior[-1,0],HV_exterior[0,0]],[HV_exterior[-1,1],HV_exterior[0,1]],'b')

		plt.plot(ACC_exterior[:,0],ACC_exterior[:,1],'g')
		plt.plot([ACC_exterior[-1,0],ACC_exterior[0,0]],[ACC_exterior[-1,1],ACC_exterior[0,1]],'g')
		_ = plt.title('Attacked',fontsize=20)
		_ = plt.ylim([0,15])
		_ = plt.xlim([0,35])

		_ = plt.xlabel('Spacing [m]',fontsize=20)
		_ = plt.yticks(fontsize=20)
		_ = plt.xticks(fontsize=20)

		# plt.savefig('ConvexHull_comparison.pdf',bbox_inches='tight')

		plt.show()

