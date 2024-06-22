import numpy as np
import matplotlib.pyplot as pt
import os

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import IDMController
from flow.envs.merge import MergePOEnv, ADDITIONAL_ENV_PARAMS
from flow.networks import MergeNetwork

#For running a simulation:
from flow.core.experiment import Experiment

from Adversaries.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked,ACC_Benign





def sim_merge_with_attack(ACC_PR,
                          attack_PR,
                          RDA_params,
                          main_inflow,
                          ramp_inflow,
                          ACC_params=None,
                          attack_warmup_steps=1000,
                          display_attack_info=True,
                          want_render=True):
    
    
    vehicles = VehicleParams()

    # Human drivers:
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.1
        }),
        car_following_params=SumoCarFollowingParams(
            speed_mode=0,
        ),
        num_vehicles=5)


    # define ACC controllers:

    k_1 = ACC_params[0]
    k_2 = ACC_params[1]
    h = ACC_params[2]
    d_min = ACC_params[3]

    adversary_accel_controller = (ACC_Switched_Controller_Attacked,{
        'k_1':k_1,
        'k_2':k_2,
        'h':h,
        'd_min':d_min,
        'Total_Attack_Duration':RDA_params[0],
        'attack_decel_rate':RDA_params[1],
        'display_attack_info':display_attack_info,
        'SS_Threshold_min':30,
        'SS_Threshold_min':5,
        'want_multiple_attacks':True})


    benign_ACC_controller = (ACC_Benign,{
        'k_1':k_1,
        'k_2':k_2,
        'h':h,
        'd_min':d_min})


    vehicles.add(
        veh_id="benign_ACC",
        color="blue",
        acceleration_controller=benign_ACC_controller,
        car_following_params=SumoCarFollowingParams(
            speed_mode=0,
        ),
        )


    vehicles.add(
        veh_id="attacker_ACC",
        color="red",
        acceleration_controller=adversary_accel_controller,
        car_following_params=SumoCarFollowingParams(
            speed_mode=0,
        ),
        )


    inflow = InFlows()


    ACC_benign_inflows = ACC_PR*main_inflow*(1-attack_PR)
    ACC_attack_inflows = ACC_PR*main_inflow*(attack_PR)
    human_main_inflows = main_inflow*(1-ACC_PR)

    print('Human inflows: '+str(human_main_inflows))
    print('Benign ACC inflows: '+str(ACC_benign_inflows))
    print('Attacking ACC inflows: '+str(ACC_attack_inflows))


    
    # add humans
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=human_main_inflows,
        departLane="free",
        departSpeed=20)

    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=ramp_inflow,
        departLane="free",
        departSpeed=7.5)


    if(ACC_PR > 0.0):
        inflow.add(
            veh_type="benign_ACC",
            edge="inflow_highway",
            vehs_per_hour=ACC_benign_inflows,
            departLane="free",
            departSpeed=20)

    if(attack_PR > 0.0):
        inflow.add(
            veh_type="attacker_ACC",
            edge="inflow_highway",
            vehs_per_hour=ACC_attack_inflows,
            departLane="free",
            departSpeed=20)


    flow_params = dict(
        # name of the experiment
        exp_tag='merge-baseline',

        # name of the flow environment the experiment is running on
        env_name=MergePOEnv,

        # name of the network class the experiment is running on
        network=MergeNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            render=want_render,
            emission_path="./data/",
            sim_step=0.1,
            use_ballistic=True,
            restart_instance=False,
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=3600,
            additional_params=ADDITIONAL_ENV_PARAMS,
            sims_per_step=5,
            warmup_steps=0,
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "merge_length": 200,
                "pre_merge_length": 1500,
                "post_merge_length": 1000,
                "merge_lanes": 1,
                "highway_lanes": 2,
                "speed_limit": 30,
            },
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="uniform",
            perturbation=5.0,
        ),
    )

    exp = Experiment(flow_params)

    [info_dict,csv_path] = exp.run(num_runs=1,convert_to_csv=True)

    return os.path.join(emission_path,csv_path)