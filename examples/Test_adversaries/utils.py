import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import flow
reload(flow)

from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController #Human driving model
from flow.controllers.routing_controllers import ContinuousRouter #Router that keeps vehicles on the ring-road

#Lane change controllers:
# from flow.controllers.lane


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



import Detectors.Deep_Learning.AutoEncoders.utils
reload(Detectors.Deep_Learning.AutoEncoders.utils)
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model



from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder



import torch

# Anti-Flow specific functions for  detection:

from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

import os
import shutil

from Adversaries.controllers import car_following_adversarial

reload(car_following_adversarial)

import Adversaries.controllers.car_following_adversarial
from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact
from Adversaries.controllers.car_following_adversarial import ACC_Benign
from Adversaries.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked

from Adversaries.controllers import car_following_adversarial

from Adversaries.controllers.base_controller import BaseController

reload(car_following_adversarial)

import time

from copy import deepcopy


from cfm_functions import Bando_OVM_FTL


def make_mal_and_benign_driver_lists(Total_Attack_Duration,
    attack_decel_rate,
    SS_Threshold_min = 60):

    driver_controller_list_with_attack = []
    driver_controller_list_without_attack = []

    #handle human drivers:

    #cfm parameters:
    a_mean=0.666
    b_mean=21.6
    s0_mean=2.21
    s1_mean=2.82
    Vm_mean=8.94

    #lane-change parameters:

    left_delta_mean = 0.5
    right_delta_mean = 0.3
    left_beta_mean=1.5
    right_beta_mean=1.5
    switching_threshold_mean = 5.0

    num_human_drivers = 35

    for i in range(num_human_drivers):
        a = a_mean + np.random.normal(0,0.1)
        b = b_mean + np.random.normal(0,0.5)
        s0 = s0_mean + np.random.normal(0,0.2)
        s1 = s1_mean + np.random.normal(0,0.2)
        Vm = Vm_mean + np.random.normal(0,0.5)

        left_delta = left_delta_mean + np.random.normal(0,0.1)
        right_delta = right_delta_mean + np.random.normal(0,0.1)
        left_beta = left_beta_mean + np.random.normal(0,0.2)
        right_beta = right_beta_mean + np.random.normal(0,0.2)
        switching_threshold = switching_threshold_mean + np.random.normal(0,0.3)

        label = 'bando_ftl_ovm_a'+str(np.round(a,2))+'_b'+str(np.round(b,2))+'_s0'+str(np.round(s0,2))+'_s1'+str(np.round(s1,2))+'_Vm'+str(np.round(Vm,2))
        cfm_controller = (Bando_OVM_FTL,{'a':a,'b':b,'s0':s0,'s1':s1,'Vm':Vm,'noise':0.1})

        driver_controller_list_with_attack.append([label,cfm_controller,1])

        driver_controller_list_without_attack.append([label,cfm_controller,1])


    k_1_mean = 1.5
    k_2_mean = 0.3
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(3):
        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        
        driver_controller_list_with_attack.append([label,cfm_controller,1])    
        driver_controller_list_without_attack.append([label,cfm_controller,1])


    want_multiple_attacks=True
    warmup_steps = 500
    display_attack_info = True

    for i in range(2):

        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                        'want_multiple_attacks':want_multiple_attacks,
                                                        'Total_Attack_Duration':Total_Attack_Duration,
                                                        'attack_decel_rate':attack_decel_rate,
                                                        'warmup_steps':warmup_steps,
                                                        'SS_Threshold_min':SS_Threshold_min,
                                                        'display_attack_info':display_attack_info})

  
        ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

        label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

        label_adv = label_adv + ACC_label

        driver_controller_list_with_attack.append([label_adv,adversary,1])

        # Add same ACC to driver controller list with no attack:
        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list_without_attack.append([label,cfm_controller,1])

    return driver_controller_list_with_attack,driver_controller_list_without_attack


# For simulation:
def run_ring_sim_variable_cfm(ring_length=300,
    driver_controller_list=None,
    num_lanes=1,
    sim_time=500,
    want_render=False,
    emission_path='data'):
    

    #There is an updated version of this that

    #Simulation parameters:
    time_step = 0.1 #In seconds, how far each step of the simulation goes.
    sim_horizon = int(np.floor(sim_time/time_step)) #How many simulation steps will be taken -> Runs for 300 seconds

    #initialize the simulation using above parameters:
    traffic_lights = TrafficLightParams() #This is empty, so no traffic lights are used.
    initial_config = InitialConfig(shuffle=True,spacing="uniform", perturbation=1) #Vehicles start out evenly spaced.
    vehicles = VehicleParams() #The vehicles object will store different classes of drivers:
    sim_params = SumoParams(sim_step=time_step, render=want_render, emission_path=emission_path) #Sets the simulation time-step and where data will be recorded.
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)
    net_params = NetParams(additional_params={'length':ring_length,
                                              'lanes':num_lanes,
                                              'speed_limit': 30,
                                              'resolution': 40})

    if(driver_controller_list is None):
        print('Running IDM.')
        num_human_drivers = 40
        #Default to the IDM if otherwise controllers not specified:
        vehicles.add("idm_driver",
            acceleration_controller=(IDMController, {'noise':0.1}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(speed_mode=0),
            num_vehicles=num_human_drivers)

    else:
        print('Number of classes of driver: '+str(len(driver_controller_list)))
        for driver in driver_controller_list:

            if(len(driver)==3):
                label = driver[0]
                cfm_controller = driver[1]
                num_vehicles = driver[2]

                vehicles.add(label,
                    acceleration_controller = cfm_controller,
                    routing_controller=(ContinuousRouter, {}),
                    car_following_params=SumoCarFollowingParams(speed_mode=0),
                    num_vehicles=num_vehicles)

            else:
                label = driver[0]
                cfm_controller = driver[1]
                lc_controller = driver[2]
                num_vehicles = driver[3]

                vehicles.add(label,
                    acceleration_controller = cfm_controller,
                    lane_change_controller = lc_controller,
                    routing_controller=(ContinuousRouter, {}),
                    car_following_params=SumoCarFollowingParams(speed_mode=0),
                    num_vehicles=num_vehicles)


    #initialize the simulation:
    flow_params = dict(
        exp_tag='ring_variable_cfm',
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





def run_comparable_sims(Total_Attack_Duration,
    attack_decel_rate,
    emission_path,
    SS_Threshold_min = 60):

    driver_controller_list_with_attack,driver_controller_list_without_attack = \
    make_mal_and_benign_driver_lists(Total_Attack_Duration,attack_decel_rate,SS_Threshold_min)

    file_path_without_attack = sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_without_attack,
                                                     ring_length=600,
                                                     sim_time=300,
                                                     emission_path=emission_path)


    file_path_without_attack = os.path.join(os.getcwd(),file_path_without_attack[1])


    file_path_with_attack = sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=600,
                                                     sim_time=300,
                                                     emission_path=emission_path)

    file_path_with_attack = os.path.join(os.getcwd(),file_path_with_attack[1])

    return file_path_without_attack,file_path_with_attack




def augment_driver_list_with_attack(driver_controller_list_without_attack,
    Total_Attack_Duration,
    attack_decel_rate,
    SS_Threshold_min = 60):

    human_drivers_list = []

    



    return driver_controller_list_with_attack

def make_mal_driver_list(Total_Attack_Duration=3.0,attack_decel_rate = -.8):

    driver_controller_list_with_attack = []

    #cfm parameters:
    a_mean=0.666
    b_mean=21.6
    s0_mean=2.21
    s1_mean=2.82
    Vm_mean=8.94

    #lane-change parameters:

    left_delta_mean = 0.5
    right_delta_mean = 0.3
    left_beta_mean=1.5
    right_beta_mean=1.5
    switching_threshold_mean = 5.0

    num_human_drivers = 35

    for i in range(num_human_drivers):
        a = a_mean + np.random.normal(0,0.1)
        b = b_mean + np.random.normal(0,0.5)
        s0 = s0_mean + np.random.normal(0,0.2)
        s1 = s1_mean + np.random.normal(0,0.2)
        Vm = Vm_mean + np.random.normal(0,0.5)

        left_delta = left_delta_mean + np.random.normal(0,0.1)
        right_delta = right_delta_mean + np.random.normal(0,0.1)
        left_beta = left_beta_mean + np.random.normal(0,0.2)
        right_beta = right_beta_mean + np.random.normal(0,0.2)
        switching_threshold = switching_threshold_mean + np.random.normal(0,0.3)

        label = 'bando_ftl_ovm_a'+str(np.round(a,2))+'_b'+str(np.round(b,2))+'_s0'+str(np.round(s0,2))+'_s1'+str(np.round(s1,2))+'_Vm'+str(np.round(Vm,2))
        cfm_controller = (Bando_OVM_FTL,{'a':a,'b':b,'s0':s0,'s1':s1,'Vm':Vm,'noise':0.1})

        driver_controller_list_with_attack.append([label,cfm_controller,1])

    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(3):
        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list_with_attack.append([label,cfm_controller,1])    



    # v_des = 10.0
    # braking_period = 5.0
    # braking_rate = -3.0

    k_1 = k_1_mean + np.random.normal(0,0.2)
    k_2 = k_2_mean + np.random.normal(0,0.2)
    h = h_mean + np.random.normal(0,0.2)
    V_m = V_m_mean + np.random.normal(0,1.0)
    d_min = d_min_mean

    want_multiple_attacks=True

    warmup_steps = 500
    SS_Threshold_min = 60
    display_attack_info = True

    adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                    'want_multiple_attacks':want_multiple_attacks,
                                                    'Total_Attack_Duration':Total_Attack_Duration,
                                                    'attack_decel_rate':attack_decel_rate,
                                                    'warmup_steps':warmup_steps,
                                                    'SS_Threshold_min':SS_Threshold_min,
                                                    'display_attack_info':display_attack_info})

    print(adversary)
    
    ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    label_adv = label_adv + ACC_label

    driver_controller_list_with_attack.append([label_adv,adversary,1])


    k_1 = k_1_mean + np.random.normal(0,0.2)
    k_2 = k_2_mean + np.random.normal(0,0.2)
    h = h_mean + np.random.normal(0,0.2)
    V_m = V_m_mean + np.random.normal(0,1.0)
    d_min = d_min_mean

    adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                    'want_multiple_attacks':want_multiple_attacks,
                                                    'Total_Attack_Duration':Total_Attack_Duration,
                                                    'attack_decel_rate':attack_decel_rate,
                                                    'warmup_steps':warmup_steps,
                                                    'SS_Threshold_min':SS_Threshold_min,
                                                    'display_attack_info':display_attack_info})
    
    print(adversary)

    ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    label_adv = label_adv + ACC_label

    driver_controller_list_with_attack.append([label_adv,adversary,1])


    return driver_controller_list_with_attack



def run_sim_with_attack(Total_Attack_Duration,attack_decel_rate,emission_path):
    
    driver_controller_list_with_attack = make_mal_driver_list(Total_Attack_Duration,attack_decel_rate)
    
    sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=600,
                                                     sim_time=300,
                                                     emission_path=emission_path)
    
    file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
    
    return file_path 






from detector_dev.Process_RingRoad_Simulation.get_rec_errors_normalized_single_lane import \
get_trajectory_timeseries,filter_timeseries_dict_for_length,get_rec_errors_normalized

import sys


def get_rec_errors(emission_path,model,warmup_period=100):

#     timeseries_dict = visualize_ring.get_sim_timeseries(emission_path,warmup_period=warmup_period)
    
    
    timeseries_dict = get_trajectory_timeseries(csv_path=emission_path,
        warmup_period=warmup_period)

    timeseries_dict = filter_timeseries_dict_for_length(timeseries_dict,seq_len=100)
    

    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in veh_ids:
        timeseries_list = []
        
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

        timeseries_list = [speed,accel,head_way,rel_vel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

        
    return smoothed_losses


