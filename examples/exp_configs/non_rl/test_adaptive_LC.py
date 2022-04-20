"""Example of an open multi-lane network with human-driven vehicles."""

from flow.controllers.car_following_models import IDMController 
from flow.controllers.lane_change_controllers import AILaneChangeController as RutgersLC
from flow.core.params import EnvParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import InFlows
from flow.core.params import VehicleParams
from flow.core.params import SumoParams
# from flow.core.rewards import instantaneous_mpg
from flow.core.params import SumoCarFollowingParams
from flow.networks import HighwayNetwork
from flow.envs.ring.lane_change_accel import ADDITIONAL_ENV_PARAMS
from flow.envs import LaneChangeAccelEnv
from flow.networks.highway import ADDITIONAL_NET_PARAMS

import numpy as np



#%% Add network parameters:
# the speed of vehicles entering the network
TRAFFIC_SPEED = 24.0
# the maximum speed at the downstream boundary edge
END_SPEED = 5.0
# Nunmber of lanes:
NUM_LANES = 3
# Length of road:
ROAD_LENGTH = 2000
# the inflow rate of vehicles
TRAFFIC_FLOW = 2500
# the simulation time horizon (in steps)
HORIZON = 5000
# whether to include noise in the car-following models
INCLUDE_NOISE = True

additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    # length of the highway
    "length": ROAD_LENGTH,
    # number of lanes
    "lanes": NUM_LANES,
    # speed limit for all edges
    "speed_limit": 35,
    # number of edges to divide the highway into
    "num_edges": 1
})

additional_env_params = ADDITIONAL_ENV_PARAMS.copy()

vehicles = VehicleParams()

lc_controller = (RutgersLC, {'left_delta':0.5,'right_delta':0.5,'want_print_LC_info':True})

vehicles.add(
    "human",
    num_vehicles=0,
    lane_change_controller=lc_controller,
    acceleration_controller=(IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
    }),
    car_following_params=SumoCarFollowingParams(
        min_gap=0.0,
        speed_mode=12  # right of way at intersections + obey limits on deceleration
    ),
)

env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

inflows = InFlows()

# Have varying inflows for each lane:
inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW),
    depart_lane=0,
    depart_speed=TRAFFIC_SPEED,
    name="idm")

inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW*.75),
    depart_lane=1,
    depart_speed=TRAFFIC_SPEED,
    name="idm")

inflows.add(
    veh_type="human",
    edge="highway_0",
    vehs_per_hour=int(TRAFFIC_FLOW*.1),
    depart_lane=2,
    depart_speed=TRAFFIC_SPEED,
    name="idm")

flow_params = dict(
    # name of the experiment
    exp_tag='highway-single',

    # name of the flow environment the experiment is running on
    env_name=LaneChangeAccelEnv,

    # name of the network class the experiment is running on
    network=HighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params
    ),

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
        use_ballistic=True,
        restart_instance=False
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflows,
        additional_params=additional_net_params
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(),
)

# custom_callables = {
#     "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
#         env.k.vehicle.get_speed(env.k.vehicle.get_ids()))),
#     "avg_outflow": lambda env: np.nan_to_num(
#         env.k.vehicle.get_outflow_rate(120)),
#     "miles_per_gallon": lambda env: np.nan_to_num(
#         instantaneous_mpg(env, env.k.vehicle.get_ids(), gain=1.0)
#     )
# }