"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers import IDMController
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, SumoLaneChangeParams
from flow.core.params import VehicleParams, InFlows
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.networks.highway import HighwayNetwork, ADDITIONAL_NET_PARAMS
from flow.envs import LaneChangeAccelEnv

import os

from flow.core.experiment import Experiment


vehicles = VehicleParams()
vehicles.add(
    veh_id="human",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),)
    # num_vehicles=0)
vehicles.add(
    veh_id="human2",
    acceleration_controller=(IDMController, {}),
    lane_change_params=SumoLaneChangeParams(
        model="SL2015",
        lc_sublane=2.0,
    ),)
    # num_vehicles=0)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflow = InFlows()
inflow.add(
    veh_type="human",
    edge="highway_0",
    probability=0.25,
    departLane="free",
    departSpeed=20)
inflow.add(
    veh_type="human2",
    edge="highway_0",
    probability=0.25,
    departLane="free",
    departSpeed=20)

# save_file_path = os.path.join(os.getcwd(),'save_state/initial_state.xml')
 
save_file_path = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/examples/save_state/highway_initial_state.xml'

# load_path = os.path.join(os.getcwd(),'save_state/initial_state.xml')
# print(load_path)

SumoParams_to_run = SumoParams(
        render=True,
        # lateral_resolution=1.0,
        # load_state=load_path,
        save_state_time = 30.0,
        save_state_file = save_file_path

    )

flow_params = dict(
    # name of the experiment
    exp_tag='highway',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams_to_run,

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS.copy(),
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params=ADDITIONAL_NET_PARAMS.copy(),
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        shuffle=True,
    ),
)

exp = Experiment(flow_params)

exp.run(num_runs=1)



