#Import different needed quantities from Flow:
from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController #Human driving model
from flow.controllers.routing_controllers import ContinuousRouter #Router that keeps vehicles on the ring-road
from flow.networks.ring import ADDITIONAL_NET_PARAMS
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.envs.ring.accel import AccelEnv
from flow.envs.ring.accel import ADDITIONAL_ENV_PARAMS
from flow.core.params import SumoParams
from flow.core.params import EnvParams
from flow.core.params import SumoCarFollowingParams
from flow.core.experiment import Experiment

from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact

import os
import numpy as np

def run_ring_sim_no_attack(ring_length=300,num_human_drivers=20):
	#Simulation parameters:
	time_step = 0.1 #In seconds, how far each step of the simulation goes.
	emission_path = 'data' #Where
	want_render = False #If we want SUMO to render the environment and display the simulation.
	sim_horizon = 5000 #How many simulation steps will be taken -> Runs for 300 seconds

	#initialize the simulation using above parameters:
	traffic_lights = TrafficLightParams() #This is empty, so no traffic lights are used.
	initial_config = InitialConfig(spacing="uniform", perturbation=1) #Vehicles start out evenly spaced.
	vehicles = VehicleParams() #The vehicles object will store different classes of drivers:
	sim_params = SumoParams(sim_step=time_step, render=want_render, emission_path=emission_path) #Sets the simulation time-step and where data will be recorded.
	env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
	net_params = NetParams(additional_params={'length':ring_length,
											  'lanes':1,
											  'speed_limit': 30,
											  'resolution': 40})

	# Define a driver model human drivers:
	vehicles.add("human",
				 acceleration_controller=(IDMController, {'noise':0.1}),
				 routing_controller=(ContinuousRouter, {}),
				 num_vehicles=num_human_drivers)


	#initialize the simulation:
	flow_params = dict(
		exp_tag='ring_no_attack',
		env_name=AccelEnv,
		network=RingNetwork,
		simulator='traci',
		sim=sim_params,
		env=env_params,
		net=net_params,
		veh=vehicles,
		initial=initial_config,
		tls=traffic_lights,
	)

	flow_params['env'].horizon = sim_horizon
	exp = Experiment(flow_params)
	print('Running ring simulation, ring length: '+str(ring_length))
	
	sim_res_list = exp.run(1, convert_to_csv=True)
	
	return sim_res_list

def run_ring_sim_with_attack_FSTP(ring_length=300,
						   num_human_drivers=19,
						   v_des=10.0,
						   braking_period = 5.0,
						   braking_rate= -2.0):

	#Simulation parameters:
	time_step = 0.1 #In seconds, how far each step of the simulation goes.
	emission_path = 'data' #Where
	want_render = False #If we want SUMO to render the environment and display the simulation.
	sim_horizon = 5000 #How many simulation steps will be taken -> Runs for 300 seconds

	#initialize the simulation using above parameters:
	traffic_lights = TrafficLightParams() #This is empty, so no traffic lights are used.
	initial_config = InitialConfig(spacing="uniform", perturbation=1) #Vehicles start out evenly spaced.
	vehicles = VehicleParams() #The vehicles object will store different classes of drivers:
	sim_params = SumoParams(sim_step=time_step, render=want_render, emission_path=emission_path) #Sets the simulation time-step and where data will be recorded.
	env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
	net_params = NetParams(additional_params={'length':ring_length,
											  'lanes':1,
											  'speed_limit': 30,
											  'resolution': 40})

	#Specify how human driven vehicles will drive:
	num_human_drivers = num_human_drivers
	# Define a driver model human drivers:
	vehicles.add("human",
				car_following_params=SumoCarFollowingParams(speed_mode=0),
				acceleration_controller=(IDMController, {'noise':0.1}),
				routing_controller=(ContinuousRouter, {}),
				num_vehicles=num_human_drivers)
	
	vehicles.add(veh_id="Adv_AV",
			 car_following_params=SumoCarFollowingParams(speed_mode=0),
			 color="red", #Let's make the adversary red
			 acceleration_controller=(FollowerStopper_Overreact, {'v_des':v_des,
														'braking_rate':braking_rate,
														'braking_period':braking_period}),
			 routing_controller=(ContinuousRouter, {}),
			 num_vehicles=1)

	#initialize the simulation:
	flow_params = dict(
		exp_tag='ring_with_attack',
		env_name=AccelEnv,
		network=RingNetwork,
		simulator='traci',
		sim=sim_params,
		env=env_params,
		net=net_params,
		veh=vehicles,
		initial=initial_config,
		tls=traffic_lights,
	)

	flow_params['env'].horizon = sim_horizon
	exp = Experiment(flow_params)
	print('Running ring attack, v_des: '+str(v_des))
	
	sim_res_list = exp.run(1, convert_to_csv=True)
	
	return sim_res_list

def run_ring_sim_with_attack(ring_length=300,
						   num_human_drivers=19,
						   adversary=None):

	#Simulation parameters:
	time_step = 0.1 #In seconds, how far each step of the simulation goes.
	emission_path = 'data' #Where
	want_render = False #If we want SUMO to render the environment and display the simulation.
	sim_horizon = 5000 #How many simulation steps will be taken -> Runs for 300 seconds

	#initialize the simulation using above parameters:
	traffic_lights = TrafficLightParams() #This is empty, so no traffic lights are used.
	initial_config = InitialConfig(spacing="uniform", perturbation=1) #Vehicles start out evenly spaced.
	vehicles = VehicleParams() #The vehicles object will store different classes of drivers:
	sim_params = SumoParams(sim_step=time_step, render=want_render, emission_path=emission_path) #Sets the simulation time-step and where data will be recorded.
	env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
	net_params = NetParams(additional_params={'length':ring_length,
											  'lanes':1,
											  'speed_limit': 30,
											  'resolution': 40})

	#Specify how human driven vehicles will drive:
	num_human_drivers = num_human_drivers
	# Define a driver model human drivers:
	vehicles.add("human",
				car_following_params=SumoCarFollowingParams(speed_mode=0),
				acceleration_controller=(IDMController, {'noise':0.1}),
				routing_controller=(ContinuousRouter, {}),
				num_vehicles=num_human_drivers)
	

	if(adversary is None):
		vehicles.add(veh_id="Adv_AV",
				 car_following_params=SumoCarFollowingParams(speed_mode=0),
				 color="red", #Let's make the adversary red
				 acceleration_controller=(FollowerStopper_Overreact, {'v_des':10,
															'braking_rate':-2.0,
															'braking_period':5.0}),
				 routing_controller=(ContinuousRouter, {}),
				 num_vehicles=1)
	else:
		vehicles.add(veh_id="Adv_AV",
				 car_following_params=SumoCarFollowingParams(speed_mode=0),
				 color="red", #Let's make the adversary red
				 acceleration_controller=adversary,
				 routing_controller=(ContinuousRouter, {}),
				 num_vehicles=1)

	#initialize the simulation:
	flow_params = dict(
		exp_tag='ring_with_attack',
		env_name=AccelEnv,
		network=RingNetwork,
		simulator='traci',
		sim=sim_params,
		env=env_params,
		net=net_params,
		veh=vehicles,
		initial=initial_config,
		tls=traffic_lights,
	)

	flow_params['env'].horizon = sim_horizon
	exp = Experiment(flow_params)
	print('Running adversarial ring simulation: ')
	print(adversary)
	
	sim_res_list = exp.run(1, convert_to_csv=True)
	
	return sim_res_list


