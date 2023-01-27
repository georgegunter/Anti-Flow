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

from run_i24_attack_random_sample import *



if __name__ == '__main__':
	attack_magnitude = -0.5
	attack_duration = 300

	inflow = 1800

	emission_path = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/Process_I24_simulations/misc_I24_data/'


	acc_penetration = 0.2
	attack_penetration = 0.05

	run_attack_sim(attack_duration,attack_magnitude,acc_penetration,attack_penetration,inflow,emission_path)




