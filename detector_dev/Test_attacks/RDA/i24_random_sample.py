#Misc:
import os
import numpy as np
import time
import ray
import matplotlib.pyplot as plt

#Human driver model:
from flow.controllers.car_following_models import IDMController


from Adversaries.controllers.car_following_adversarial import *

# For routing and lane-changing:
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.routing_controllers import i24_adversarial_router
from flow.controllers.routing_controllers import I24Router


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


def get_flow_params(attack_duration,
	attack_magnitude,
	attack_frequency,
	acc_penetration,
	inflow,
	emission_path,
	attack_penetration,
	want_render=False,
	display_attack_info=True,
	ACC_comp_params=None,
	ACC_benign_params=None):

	SIM_LENGTH = 1200 #simulation length in seconds

	sim_step = .1 #Simulation step size

	horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

	WARMUP_STEPS = 2000 #Attack vehicles don't attack before this # of steps

	BASELINE_INFLOW_PER_LANE = inflow #Per lane flow rate in veh/hr

	inflow_speed = 25.5

	ON_RAMP_FLOW = 1000

	highway_start_edge = 'Eastbound_2'

	ACC_PENETRATION_RATE = acc_penetration

	HUMAN_INFLOW = (1-ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	ACC_INFLOW = (ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	ACC_ATTACK_INFLOW = (attack_penetration)*ACC_INFLOW

	ACC_BENIGN_INFLOW = (1-attack_penetration)*ACC_INFLOW

	##################################
	#ATTACK VEHICLE PARAMETERS:
	##################################

	vehicles = VehicleParams()

	inflow = InFlows()


	if(ACC_comp_params is not None):
		# For if want to execute 'platoon attack' by changing ACC params:

		k_1 = ACC_comp_params[0]
		k_2 = ACC_comp_params[1]
		h = ACC_comp_params[2]
		d_min = ACC_comp_params[3]

		adversary_accel_controller = (ACC_comp_RDA,{
			'k_1':k_1,
			'k_2':k_2,
			'h':h,
			'd_min':d_min,
			'warmup_steps':WARMUP_STEPS,
			'attack_decel_rate':attack_magnitude,
			'g':attack_magnitude,
			'SS_Threshold_min':attack_frequency,
			'SS_Threshold_range':0,
			'display_attack_info':display_attack_info})
	else:
		adversary_accel_controller = (ACC_comp_RDA,{
			'warmup_steps':WARMUP_STEPS,
			'Total_Attack_Duration':attack_duration,
			'attack_decel_rate':attack_magnitude,
			'SS_Threshold_min':attack_frequency,
			'SS_Threshold_range':0,
			'display_attack_info':display_attack_info})

	adversarial_router = (i24_adversarial_router,{})

	#Should never attack, so just a regular ACC:

	if(ACC_benign_params is not None):
		k_1 = ACC_benign_params[0]
		k_2 = ACC_benign_params[1]
		h = ACC_benign_params[2]
		d_min = ACC_benign_params[3]

		benign_ACC_controller = (ACC_Benign,{
			'k_1':k_1,
			'k_2':k_2,
			'h':h,
			'd_min':d_min})

	else:
		benign_ACC_controller = (ACC_Benign,{})

	##################################
	#DRIVER TYPES AND INFLOWS:
	##################################
	# lane_list = ['0','1','2','3']
	lane_list = ['1','2','3','4']

	# Attack ACC params and inflows:
	vehicles.add(
		veh_id="attacker_ACC",
		num_vehicles=0,
		color="red",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=0,
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			speed_mode=0  # right of way at intersections + obey limits on deceleration
		),
		acceleration_controller=adversary_accel_controller,
		lane_change_controller=(StaticLaneChanger,{}),
		routing_controller=adversarial_router,
	)

	vehicles.add(
		veh_id="benign_ACC",
		num_vehicles=0,
		color="blue",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=0,
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			speed_mode=0  # right of way at intersections + obey limits on deceleration
		),
		acceleration_controller=benign_ACC_controller,
		lane_change_controller=(StaticLaneChanger,{}),
		routing_controller=adversarial_router, #This breaks everything
	)


	for i,lane in enumerate(lane_list):
		if(ACC_ATTACK_INFLOW > 0):
			inflow.add(
				veh_type="attacker_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_ATTACK_INFLOW ,
				depart_lane=lane,
				depart_speed=inflow_speed)

		if(ACC_BENIGN_INFLOW > 0):
			inflow.add(
				veh_type="benign_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_BENIGN_INFLOW ,
				depart_lane=lane,
				depart_speed=inflow_speed)

	#handles when vehicles wait too long to try and merge and get stuck on merge:
	human_routing_controller = (I24Router,{'position_to_switch_routes':75})

	#Human params and inflows (main line and on-ramp)
	vehicles.add(
		veh_id="human_main",
		num_vehicles=0,
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=597,
			lc_speed_gain=5.0
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			min_gap=0.5,
			speed_mode=12  # right of way at intersections + obey limits on deceleration
		),
		routing_controller=human_routing_controller,
	)

	vehicles.add(
		veh_id="human_on_ramp",
		num_vehicles=0,
		# color="red",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=597,
			lc_speed_gain=5.0
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			min_gap=0.5,
			speed_mode=12  # right of way at intersections + obey limits on deceleration
		),
		routing_controller=human_routing_controller,
	)

	for i,lane in enumerate(lane_list):
		inflow.add(
			veh_type="human_main",
			edge=highway_start_edge,
			vehs_per_hour=HUMAN_INFLOW,
			depart_lane=lane,
			depart_speed=inflow_speed)

	inflow.add(
		veh_type="human_on_ramp",
		edge='Eastbound_On_1',
		vehs_per_hour=ON_RAMP_FLOW,
		depart_lane='random',
		depart_speed=20)

	##################################
	#INITIALIZE FLOW PARAMETERS DICT:
	##################################


	NET_TEMPLATE = os.path.join(
			config.PROJECT_PATH,
			"examples/exp_configs/templates/sumo/i24_subnetwork_fix_merges.net.xml")

	flow_params = dict(
		# name of the experiment
		exp_tag='I-24_subnetwork',

		# name of the flow environment the experiment is running on
		env_name=TestEnv,

		# name of the network class the experiment is running on
		network=I24SubNetwork,

		# simulator that is used by the experiment
		simulator='traci',

		# simulation-related parameters
		sim=SumoParams(
			sim_step=sim_step,
			render=want_render,
			color_by_speed=False,
			use_ballistic=True,
			emission_path=emission_path,
			print_warnings=False,
			restart_instance=True
		),

		# environment related parameters (see flow.core.params.EnvParams)
		env=EnvParams(
			horizon=horizon,
		),

		# network-related parameters (see flow.core.params.NetParams and the
		# network's documentation or ADDITIONAL_NET_PARAMS component)
		net=NetParams(
			inflows=inflow,
			template=NET_TEMPLATE,
			additional_params={"on_ramp": False,'ghost_edge':False}
		),

		# vehicles to be placed in the network at the start of a rollout (see
		# flow.core.params.VehicleParams)
		veh=vehicles,

		# parameters specifying the positioning of vehicles upon initialization/
		# reset (see flow.core.params.InitialConfig)
		initial=InitialConfig(
			edges_distribution=EDGES_DISTRIBUTION,
		),
	)

	return flow_params

def rename_file(csv_path,emission_path,attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow):

	files = os.listdir(emission_path)

	# This is hacky, but it should look in the right place...

	# files = os.listdir('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_csv_repo')

	file_name_no_version = 'MVA_AD_'+str(attack_duration)+'_AF_'+str(attack_frequency)+'_DA_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(emission_path,file_name_with_version)

	os.rename(csv_path,file_path)

	return file_name_with_version

def run_attack_sim(attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow,emission_path,get_results=False,delete_file=False,want_render=False):

	flow_params = get_flow_params(attack_duration,attack_magnitude,attack_frequency,acc_penetration,inflow,emission_path,attack_penetration,want_render=want_render)

	exp = Experiment(flow_params)

	[info_dict,csv_path] = exp.run(num_runs=1,convert_to_csv=True)

	file_name_with_version = rename_file(csv_path,emission_path,attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow)

	file_path = os.path.join(emission_path,file_name_with_version)

	[file_path,file_name_with_version]


@ray.remote
def run_attack_sim_ray(attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow,emission_path):

	sim_results = run_attack_sim(attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow,emission_path)

	return sim_results

def run_sim_list(attack_duration_list,
	attack_magnitude_list,
	attack_frequency_list,
	acc_penetration_list,
	attack_penetration_list,
	inflow_list,
	emission_path,
	want_render=False,
	write_results=True,
	delete_file=False):

	sim_info_ids = []
	num_sims = len(attack_duration_list)

	for i in range(num_sims):

		attack_duration = attack_duration_list[i]
		attack_magnitude = attack_magnitude_list[i]
		attack_frequency = attack_frequency_list[i]
		acc_penetration = acc_penetration_list[i]
		attack_penetration = attack_penetration_list[i]
		inflow = inflow_list[i]

		print('running sim: '+str(i))

		sim_info_ids.append(
			run_attack_sim_ray.remote(
				attack_duration=attack_duration,
				attack_magnitude=attack_magnitude,
				attack_frequency=attack_frequency,
				acc_penetration=acc_penetration,
				attack_penetration=attack_penetration,
				inflow=inflow,
				emission_path=emission_path,
				want_render=want_render,
				get_results=False,
				delete_file=delete_file))


	sim_info_list = ray.get(sim_info_ids)

	return sim_info_list







def run_batch_random_sims(attack_magnitude_range,attack_duration_range,attack_frequency_range,acc_penetration,attack_penetration,inflow,num_samples,emission_path):

	

	attack_magnitude_vals = np.random.uniform(low = attack_magnitude_range[0],high = attack_magnitude_range[1],size= num_samples )
	# attack_magnitude_vals = attack_magnitude_vals*(attack_magnitude_range[1]-attack_magnitude_range[0])
	# attack_magnitude_vals = attack_magnitude_vals + attack_magnitude_range[0]
	

	attack_duration_vals = np.random.uniform(low = attack_duration_range[0],high = attack_duration_range[1],size= num_samples )
	# attack_duration_vals = attack_duration_vals*(attack_duration_range[1]-attack_duration_range[0])
	# attack_duration_vals = attack_duration_vals + attack_duration_range[0]

	attack_frequency_vals = np.random.uniform(low = attack_frequency_range[0],high = attack_frequency_range[1],size= num_samples )
	# attack_frequency_vals = attack_frequency_vals*(attack_frequency_range[1]-attack_frequency_range[0])
	# attack_frequency_vals = attack_frequency_vals + attack_frequency_range[0]

	plt.figure()
	plt.subplot(1,3,1)
	plt.plot(np.ones_like(attack_magnitude_vals),attack_magnitude_vals,'.')
	plt.title('Magnitude')
	plt.subplot(1,3,2)
	plt.plot(np.ones_like(attack_frequency_vals),attack_frequency_vals,'.')
	plt.title('Frequency')
	plt.subplot(1,3,3)
	plt.plot(np.ones_like(attack_duration_vals),attack_duration_vals,'.')
	plt.title('Duration')
	plt.show()

	print('Selected attack parameters.')

	print('Beginning simulations...')

	attack_result_ids = []

	for i in range(num_samples):
		attack_magnitude = attack_magnitude_vals[i]
		attack_duration = attack_duration_vals[i]
		attack_frequency = attack_frequency_vals[i]

		attack_result_ids.append(run_attack_sim_ray.remote(attack_duration,attack_magnitude,attack_frequency,acc_penetration,attack_penetration,inflow,emission_path))

	attack_file_paths = ray.get(attack_result_ids)


	return attack_file_paths



if __name__ == '__main__':


	ray.init(num_cpus=4)

	want_random_sample = True

	if(want_random_sample):

		attack_magnitude_range = [-3,0.0]
		attack_duration_range = [0,20]
		attack_frequency_range = [0,300]

		inflow = 1800

		emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/i24_random_sample'
		acc_penetration = 0.2
		attack_penetration = 0.1
		

		num_samples = 50
		run_batch_random_sims(attack_magnitude_range,attack_duration_range,attack_frequency_range,acc_penetration,attack_penetration,inflow,num_samples,emission_path)

		print('Finished with random sample for max velocity compromise.')



