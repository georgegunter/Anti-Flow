"""I-24 subnetwork example."""
import os

import numpy as np

from flow.controllers.car_following_models import IDMController

#Specific to using to control adverarial vehicles:
# from flow.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.routing_controllers import i24_adversarial_router
from flow.controllers.routing_controllers import I24Router

from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact
from Adversaries.controllers.car_following_adversarial import ACC_Benign
from Adversaries.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked


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

# For procesing results:
from load_sim_results import get_sim_results_csv
from load_sim_results import write_results_to_csv
from load_sim_results import get_all_params

# Ray for parrallelization:
import ray



def get_flow_params(acc_penetration,
	inflow,
	emission_path,
	want_render=False):

	SIM_LENGTH = 1000 #simulation length in seconds

	sim_step = .1 #Simulation step size

	horizon = int(np.floor(SIM_LENGTH/sim_step)) #Number of simulation steps

	# WARMUP_STEPS = 4000 #Attack vehicles don't attack before this # of steps

	BASELINE_INFLOW_PER_LANE = inflow #Per lane flow rate in veh/hr

	inflow_speed = 25.5

	ON_RAMP_FLOW = 1000

	highway_start_edge = 'Eastbound_2'

	ACC_PENETRATION_RATE = acc_penetration

	HUMAN_INFLOW = (1-ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	ACC_INFLOW = (ACC_PENETRATION_RATE)*BASELINE_INFLOW_PER_LANE

	##################################
	#ATTACK VEHICLE PARAMETERS:
	##################################

	vehicles = VehicleParams()

	inflow = InFlows()

	adversarial_router = (i24_adversarial_router,{})

	#Should never attack, so just a regular ACC:

	# if(ACC_benign_params is not None):
	# 	k_1 = ACC_benign_params[0]
	# 	k_2 = ACC_benign_params[1]
	# 	h = ACC_benign_params[2]
	# 	d_min = ACC_benign_params[3]

	# 	benign_ACC_controller = (ACC_Benign,{
	# 		'k_1':k_1,
	# 		'k_2':k_2,
	# 		'h':h,
	# 		'd_min':d_min})

	# else:
	# 	benign_ACC_controller = (ACC_Benign,{})

	benign_ACC_controller = (ACC_Benign,{})

	##################################
	#DRIVER TYPES AND INFLOWS:
	##################################
	# lane_list = ['0','1','2','3']
	lane_list = ['1','2','3','4']

	# Attack ACC params and inflows:

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





	save_state_time = 999.0,
    save_state_file = 'i24_initial_state_with_inflow_'+str(inflow)+'.xml'


	sim_params = None

	if(emission_path is not None):
		sim_params = SumoParams(
				sim_step=sim_step,
				render=want_render,
				save_state_time = save_state_time,
				save_state_file = save_file_path,
				color_by_speed=False,
				use_ballistic=True,
				emission_path=emission_path,
				print_warnings=False,
				restart_instance=True)
	else:
		sim_params = SumoParams(
				sim_step=sim_step,
				render=want_render,
				save_state_time = save_state_time,
				save_state_file = save_file_path,
				color_by_speed=False,
				use_ballistic=True,
				print_warnings=False,
				restart_instance=True)

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


if __name__ == '__main__':

# attack_duration,
# attack_magnitude,
# acc_penetration,
# inflow,
# emission_path,
# attack_penetration,
# want_render=False,
# display_attack_info=True,
# ACC_comp_params=None,
# ACC_benign_params=None

	inflow = 2400
	acc_penetration = 0.2
	emission_path = None
	want_render = True

	flow_params = get_flow_params(attack_duration,inflow,emission_path,
	want_render=False,
	display_attack_info=True,
	ACC_comp_params=None,
	ACC_benign_params=None)











