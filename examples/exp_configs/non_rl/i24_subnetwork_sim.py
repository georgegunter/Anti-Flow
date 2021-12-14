"""I-24 subnetwork example."""
import os

import numpy as np

from flow.controllers.car_following_models import IDMController
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.lane_change_controllers import AILaneChangeController
from flow.controllers.lane_change_controllers import I24_routing_LC_controller
from flow.controllers.lane_change_controllers import I24_LC_controller
from flow.controllers.routing_controllers import I24Router
# from flow.controllers.routing_controllers import I210Router
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

from flow.networks.I24_Subnetwork import I24SubNetwork
from flow.networks.I24_Subnetwork import EDGES_DISTRIBUTION


# =========================================================================== #
# Specify model parameters:                                                   #
# =========================================================================== #

WANT_GHOST_CELL = True #Not integrated yet
WANT_DOWNSTREAM_BOUNDARY = True #Not integrated yet

WANT_ON_RAMPS = True #Whether want vehicles to merge on
WANT_TRUCKS = True #Whether want trucks in the sim

horizon = 2000 #number of simulation steps
sim_step = .2 #Simulation step size

HUMAN_INFLOW = 2050 #Per lane flow rate in veh/hr on the main highway.
inflow_speed = 25.5 #Speed corresponding to this inflow rate

TRUCK_INFLOW = 200 #Inflow of trucks/hr on the right most late

# downstream_speed = 3.5 #What the downstream congestion speed should be

on_ramp_inflow = 500 #on ramp inflow rate

# =========================================================================== #
# Specify driver control models                                               #
# =========================================================================== #

#Currently set so the only safety measure is to follow speed limit:
FLOW_FAIL_SAFE = ['obey_speed_limit']
CFP_SPEED_MODE = 0 #Sumo speed_mode for car following
LCP_LC_MODE = 0 #Sumo LC_mode

want_custom_LC = True


#Passenger vehicles coming on to the main road:
human_accel_controller = (IDMController, {
        "a": 1.3,
        "b": 2.0,
        "noise": 0.3,
        "v0": 27.0,
        "display_warnings": False,
        "fail_safe": FLOW_FAIL_SAFE,
    })

#LC for passenger vehicles spawned on main higway:
human_lc_controller = (AILaneChangeController,{})

#Routing controller to handle vehicles in exit lane:
I24_routing_controller = (I24Router,{})

#Accel for trucks:
truck_accel_controller = (IDMController, {
        "a": 1.3,
        "b": 1.75,
        "T":2.5,
        "s0":5,
        "noise": 0.3,
        "v0": 27.0,
        "display_warnings": False,
        "fail_safe": FLOW_FAIL_SAFE,
    })

#LC for trucks:
truck_lc_controller = (StaticLaneChanger,{})


# A combined insentive and routing based LC controller:
I24_LC_controller = (I24_LC_controller,{
    'display_warning_messages':False,
    'delta_follower':5.0,
    'delta_leader':5.0,
    })

# =========================================================================== #
# Specify inflows:                                                            #
# =========================================================================== #

vehicles = VehicleParams()

inflow = InFlows()

#Add human drivers:
if(want_custom_LC):
    vehicles.add(
        "human_main",
        num_vehicles=0,
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=LCP_LC_MODE,
        ),
        # this is only right of way on
        car_following_params=SumoCarFollowingParams(
            speed_mode=CFP_SPEED_MODE  # right of way at intersections + obey limits on deceleration
        ),
        lane_change_controller=human_lc_controller,
        acceleration_controller=human_accel_controller,
        routing_controller=I24_routing_controller,
    )
else:
    vehicles.add(
        "human_main",
        num_vehicles=0,
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=LCP_LC_MODE,
        ),
        # this is only right of way on
        car_following_params=SumoCarFollowingParams(
            speed_mode=CFP_SPEED_MODE  # right of way at intersections + obey limits on deceleration
        ),
        acceleration_controller=human_accel_controller,
        routing_controller=I24_routing_controller,
    )


lane_list = ['0','1','2','3']
HUMAN_INFLOW_RATES_MAIN = [HUMAN_INFLOW,HUMAN_INFLOW,HUMAN_INFLOW,HUMAN_INFLOW]
highway_start_edge = 'Eastbound_3'

for i,lane in enumerate(lane_list):
    inflow.add(
        veh_type="human_main",
        edge=highway_start_edge,
        vehs_per_hour=HUMAN_INFLOW_RATES_MAIN[i],
        departLane=lane,
        departSpeed=inflow_speed)

if(WANT_TRUCKS):
    vehicles.add(
        "truck_main",
        num_vehicles=0,
        color="blue",
        lane_change_params=SumoLaneChangeParams(
            lane_change_mode=LCP_LC_MODE,
        ),
        # this is only right of way on
        car_following_params=SumoCarFollowingParams(
            min_gap=0.5,
            speed_mode=CFP_SPEED_MODE,
            length=15,  # right of way at intersections + obey limits on deceleration
        ),
        lane_change_controller=truck_lc_controller,
        acceleration_controller=truck_accel_controller,
        routing_controller=I24_routing_controller,
    )

    inflow.add(
        veh_type="truck_main",
        edge=highway_start_edge,
        vehs_per_hour=TRUCK_INFLOW,
        departLane=0,
        departSpeed=inflow_speed)

if(WANT_ON_RAMPS):

    if(want_custom_LC):
        vehicles.add(
            "human_on_ramp",
            num_vehicles=0,
            color="red",
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=LCP_LC_MODE,
            ),
            # this is only right of way on
            car_following_params=SumoCarFollowingParams(
                min_gap=0.0,
                speed_mode=CFP_SPEED_MODE  # right of way at intersections + obey limits on deceleration
            ),
            lane_change_controller=I24_LC_controller,
            acceleration_controller=human_accel_controller,
            routing_controller=I24_routing_controller,
        )
    else:
        vehicles.add(
            "human_on_ramp",
            num_vehicles=0,
            color="red",
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=LCP_LC_MODE,
            ),
            # this is only right of way on
            car_following_params=SumoCarFollowingParams(
                min_gap=0.0,
                speed_mode=CFP_SPEED_MODE  # right of way at intersections + obey limits on deceleration
            ),
            acceleration_controller=human_accel_controller,
            routing_controller=I24_routing_controller,
        )




    inflow.add(
        veh_type="human_on_ramp",
        edge='Eastbound_On_1',
        vehs_per_hour=1000,
        departLane='random',
        departSpeed=20)

# =========================================================================== #
# Specify network level details:                                              #
# =========================================================================== #

#This is the default file used:
NET_TEMPLATE = os.path.join(
        config.PROJECT_PATH,
        "examples/exp_configs/templates/sumo/i24_subnetwork_fix_merges.net.xml")

# =========================================================================== #
# TODO: Add function to set edge speed limits                                 #
# =========================================================================== #

# def Set_i24_congestion(sumo_templates_path=None,speed=5.0,boundary_edge='10'):
#     '''
#     TODO: Change to work properly with the i24 network

#     '''
#     speed = str(speed)

#     #Original xml file:
#     fileName = 'i24_subnetwork.net.xml'
#     fileName = os.path.join(sumo_templates_path,fileName)

#     file_lines = []




#     # lines_to_change = [2415,2418,2421,2424,2427,2430]

#     with open(fileName) as f:
#         file_lines = f.readlines()

#     edge_descriptor_line = None

#     for i,line in enumerate(file_lines):
#         if 'edge id="Eastbound_'+boundary_edge+'"' in line:
#             edge_descriptor_line = i

#     # for line_num in lines_to_change:
#     #     new_line = file_lines[line_num][:183]+speed+file_lines[line_num][186:]
#     #     file_lines[line_num] = new_line

#     #New xml file that will be used in the sim:
#     new_fileName = os.path.join(sumo_templates_path,'i24_subnetwork.net.xml')

#     with open(new_fileName, 'w') as filehandle:
#         filehandle.writelines(file_lines)


#     return new_fileName

# NET_TEMPLATE = Set_i210_congestion(sumo_templates_path=sumo_templates_path,speed=downstream_speed)

# Make dict to pass to simulator:
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
        render=True,
        color_by_speed=False,
        use_ballistic=True
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

# edge_id = "119257908#1-AddedOnRampEdge"
# custom_callables = {
#     "avg_merge_speed": lambda env: np.nan_to_num(np.mean(
#         env.k.vehicle.get_speed(env.k.vehicle.get_ids_by_edge(edge_id)))),
#     "avg_outflow": lambda env: np.nan_to_num(
#         env.k.vehicle.get_outflow_rate(120)),
#     # we multiply by 5 to account for the vehicle length and by 1000 to convert
#     # into veh/km
#     "avg_density": lambda env: 5 * 1000 * len(env.k.vehicle.get_ids_by_edge(
#         edge_id)) / (env.k.network.edge_length(edge_id)
#                      * env.k.network.num_lanes(edge_id)),
# }