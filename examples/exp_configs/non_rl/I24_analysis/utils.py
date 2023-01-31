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


from Defenders.controllers.car_following_defense import FollowerStopper_ACC_switch_on_time



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


def get_flow_params_with_attack_and_defense(attack_duration,
	attack_magnitude,
	acc_penetration,
	inflow,
	emission_path,
	attack_penetration,
	defender_penetration,
	v_des=24.0,
	sim_length=1800.0,
	inflow_speed=25.0,
	want_render=True,
	display_attack_info=True,
	ACC_comp_params=None,
	ACC_benign_params=None):

	SIM_LENGTH = sim_length #simulation length in seconds
	# SIM_LENGTH = 300

	sim_step = .1 #Simulation step size

	horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

	 #Attack vehicles don't attack before this # of steps

	warmup_time = (horizon/3)*sim_step

	WARMUP_STEPS = int(horizon/3)

	BASELINE_INFLOW_PER_LANE = inflow #Per lane flow rate in veh/hr

	ON_RAMP_FLOW = int(inflow/3.0)

	highway_start_edge = 'Eastbound_2'

	ACC_PENETRATION_RATE = acc_penetration

	HUMAN_INFLOW = (1-ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	ALL_AV_INFLOW = (ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	ACC_ATTACK_INFLOW = attack_penetration*ALL_AV_INFLOW

	FSTOP_INFLOW = defender_penetration*ALL_AV_INFLOW

	ACC_BENIGN_INFLOW = ALL_AV_INFLOW - ACC_ATTACK_INFLOW - FSTOP_INFLOW

	##################################
	#VEHICLE PARAMETERS:
	##################################

	vehicles = VehicleParams()

	inflow = InFlows()

	attack_magnitude = -np.abs(attack_magnitude)

	attack_duration = attack_duration
	attack_magnitude = attack_magnitude

	if(ACC_comp_params is not None):
		# For if want to execute 'platoon attack' by changing ACC params:

		k_1 = ACC_comp_params[0]
		k_2 = ACC_comp_params[1]
		h = ACC_comp_params[2]
		d_min = ACC_comp_params[3]

		adversary_accel_controller = (ACC_Switched_Controller_Attacked,{
			'k_1':k_1,
			'k_2':k_2,
			'h':h,
			'd_min':d_min,
			'warmup_steps':WARMUP_STEPS,
			'Total_Attack_Duration':attack_magnitude,
			'attack_decel_rate':attack_magnitude,
			'display_attack_info':display_attack_info,
			'want_multiple_attacks':True})
	else:
		adversary_accel_controller = (ACC_Switched_Controller_Attacked,{
			'warmup_steps':WARMUP_STEPS,
			'Total_Attack_Duration':attack_duration,
			'attack_decel_rate':attack_magnitude,
			'display_attack_info':display_attack_info,
			'want_multiple_attacks':True})

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



	# Creating follower stopper that turns on at a certain time:
	if(ACC_benign_params is not None):
		k_1 = ACC_benign_params[0]
		k_2 = ACC_benign_params[1]
		h = ACC_benign_params[2]
		d_min = ACC_benign_params[3]

		defense_controller = (FollowerStopper_ACC_switch_on_time,{
			'k_1':k_1,
			'k_2':k_2,
			'h':h,
			'd_min':d_min,
			'v_des':v_des,
			'warmup_time':warmup_time})

	else:
		defense_controller = (FollowerStopper_ACC_switch_on_time,{})





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
			print('Attacking ACCs will be present.')
			inflow.add(
				veh_type="attacker_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_ATTACK_INFLOW ,
				depart_lane=lane,
				depart_speed=inflow_speed)

		if(ACC_BENIGN_INFLOW > 0):
			print('Benign ACCs will be present.')
			inflow.add(
				veh_type="benign_ACC",
				edge=highway_start_edge,
				vehs_per_hour=ACC_BENIGN_INFLOW ,
				depart_lane=lane,
				depart_speed=inflow_speed)

	# Define sim parameters for the follower stopper:
	vehicles.add(
		veh_id="defender_FSTOP",
		num_vehicles=0,
		color="green",
		lane_change_params=SumoLaneChangeParams(
			lane_change_mode=0,
		),
		# this is only right of way on
		car_following_params=SumoCarFollowingParams(
			speed_mode=0  # right of way at intersections + obey limits on deceleration
		),
		acceleration_controller=defense_controller,
		lane_change_controller=(StaticLaneChanger,{}),
		routing_controller=adversarial_router,
	)

	for i,lane in enumerate(lane_list):
		if(FSTOP_INFLOW > 0):
			print('Defenders present in simulation.')
			inflow.add(
				veh_type="defender_FSTOP",
				edge=highway_start_edge,
				vehs_per_hour=FSTOP_INFLOW,
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


	# initial_load_state = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/examples/exp_configs/non_rl/i24_initial_state_with_inflow_2400.xml'

	sim_params=SumoParams(
			sim_step=sim_step,
			render=want_render,
			# load_state=initial_load_state,
			color_by_speed=False,
			use_ballistic=True,
			emission_path=emission_path,
			print_warnings=False,
			restart_instance=True
		)




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
		sim=sim_params,

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


# Generally useful functions for running sims and naming conventions:

def run_sim(flow_params,emission_path):

	exp = Experiment(flow_params)

	[info_dict,csv_path] = exp.run(num_runs=1,convert_to_csv=True)

	return os.path.join(emission_path,csv_path)

def rename_file(csv_path,emission_path,attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow):

	files = os.listdir(emission_path)

	# This is hacky, but it should look in the right place...

	# files = os.listdir('/Users/vanderbilt/Desktop/Research_2020/Traffic_Attack/flow/examples/i24_adversarial_sims/results_csv_repo')

	file_name_no_version = 'Dur_'+str(attack_duration)+'_Mag_'+str(attack_magnitude)+'_Inflow_'+str(inflow)+'_ACCPenetration_'+str(acc_penetration)+'_AttackPenetration_'+str(attack_penetration)

	file_version = 1

	for file in files:
		if(file_name_no_version in file):
			file_version += 1

	file_name_with_version = file_name_no_version+'_ver_'+str(file_version)+'.csv'

	file_path = os.path.join(emission_path,file_name_with_version)

	os.rename(csv_path,file_path)

	return file_name_with_version
