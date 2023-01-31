"""I-24 subnetwork example."""
import os

import numpy as np

from flow.controllers.car_following_models import IDMController

#Specific to using to control adverarial vehicles:
# from flow.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.routing_controllers import i24_adversarial_router
from flow.controllers.routing_controllers import I24Router

from Adversaries.controllers.car_following_adversarial import ACC_Benign,ACC_Switched_Controller_Attacked,ACC_Switched_Controller_Attacked_Single

# from flow.controllers.lane_change_controllers import AILaneChangeController
# from flow.controllers.lane_change_controllers import I24_routing_LC_controller
# from flow.controllers.routing_controllers import I210Router

# For flow:
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import SumoLaneChangeParams
from flow.core.params import VehicleParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows

from flow.core.params import SumoCarFollowingParams

import flow.config as config
from flow.envs import TestEnv

#Needed for i24 network:
from flow.networks.I24_subnetwork import I24SubNetwork
from flow.networks.I24_subnetwork import EDGES_DISTRIBUTION

#For running a simulation:
from flow.core.experiment import Experiment

import time

import ray

from utils import *

from detector_dev.Process_I24_simulations.i24_utils import get_attack_params



def get_old_random_sample_attack_params(
	emission_path = '/Volumes/My Passport for Mac/i24_random_sample/simulations/1800_inflow'):
	
	files = os.listdir(emission_path)

	csv_files = []

	for file in files:
		if('csv' in file): csv_files.append(file)

	attack_params_list = []

	for file in csv_files:
		attack_params_list.append(get_attack_params(file))

	return attack_params_list




def rename_file(csv_path,
	emission_path,
	attack_duration,
	attack_magnitude,
	acc_penetration,
	attack_penetration,
	defender_penetration,
	v_des,
	inflow):

	files = os.listdir(emission_path)

	# This is hacky, but it should look in the right place...

	# files = os.listdir('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_csv_repo')

	file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	file_name_no_version = file_name_no_version + '_DefenderPenetration_'+str(defender_penetration) + '_vdes_'+str(v_des)

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(emission_path,file_name_with_version)

	os.rename(csv_path,file_path)

	return file_name_with_version


def simulate(attack_duration,
	attack_magnitude,
	acc_penetration,
	inflow,
	emission_path,
	attack_penetration,
	defender_penetration,
	v_des):

	flow_params = get_flow_params_with_attack_and_defense(attack_duration,
		attack_magnitude,
		acc_penetration,
		inflow,
		emission_path,
		attack_penetration,
		defender_penetration,
		v_des=v_des,
		want_render=False)

	csv_path = run_sim(flow_params,emission_path)

	csv_path = rename_file(csv_path,
		emission_path,
		attack_duration,
		attack_magnitude,
		acc_penetration,
		attack_penetration,
		defender_penetration,
		v_des,
		inflow)

	return csv_path


@ray.remote
def simulate_ray(attack_duration,
	attack_magnitude,
	acc_penetration,
	inflow,
	emission_path,
	attack_penetration,
	defender_penetration,
	v_des):


	return simulate(attack_duration,attack_magnitude,acc_penetration,inflow,emission_path,attack_penetration,defender_penetration,v_des)


def batch_simulate_ray(attack_duration_list,
	attack_magnitude_list,
	acc_penetration_list,
	inflow_list,
	emission_path_list,
	attack_penetration_list,
	defender_penetration_list,
	v_des_list):

	sim_results_id_list = []

	num_samples = len(attack_duration_list)

	for i in range(num_samples):

		attack_duration = attack_duration_list[i]
		attack_magnitude = attack_magnitude_list[i]
		acc_penetration = acc_penetration_list[i]
		inflow = inflow_list[i]
		emission_path = emission_path_list[i]
		attack_penetration = attack_penetration_list[i]
		defender_penetration = defender_penetration_list[i]
		v_des = v_des_list[i]


		sim_results_id_list.append(
			simulate_ray.remote(
				attack_duration,
				attack_magnitude,
				acc_penetration,
				inflow,
				emission_path,
				attack_penetration,
				defender_penetration,
				v_des))

	sim_results_all = ray.get(sim_results_id_list)

	return sim_results_all








	return None

if __name__ == '__main__':
	# attack_magnitude = -0.5
	# attack_duration = 300

	inflow = 1800

	# emission_path = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/Process_I24_simulations/misc_I24_data/'

	emission_path = '/Volumes/My Passport for Mac/i24_attack_and_defend/1800'

	v_des = 21.0

	acc_penetration = 0.2
	defender_penetration = 0.1
	attack_penetration = 0.1

	attack_params_list = get_old_random_sample_attack_params()

	num_samples = len(attack_params_list)


	attack_duration_list = []
	attack_magnitude_list = []
	acc_penetration_list = []
	inflow_list = []
	emission_path_list = []
	attack_penetration_list = []
	defender_penetration_list = []
	v_des_list = []



	for i in range(num_samples):

		attack_duration = attack_params_list[i][0]
		attack_magnitude = attack_params_list[i][1]

		attack_duration_list.append(attack_duration)
		attack_magnitude_list.append(attack_magnitude)
		acc_penetration_list.append(acc_penetration)
		inflow_list.append(inflow)
		emission_path_list.append(emission_path)
		attack_penetration_list.append(attack_penetration)
		defender_penetration_list.append(defender_penetration)
		v_des_list.append(v_des)

	print('number of samples: '+str(len(v_des_list)))


	begin_simulation_time = time.time()


	batch_simulate_ray(attack_duration_list,
		attack_magnitude_list,
		acc_penetration_list,
		inflow_list,
		emission_path_list,
		attack_penetration_list,
		defender_penetration_list,
		v_des_list)



	end_simulation_time = time.time()

	print('Simulations finished.')
	print('Computation time: '+str(end_simulation_time - begin_simulation_time))






