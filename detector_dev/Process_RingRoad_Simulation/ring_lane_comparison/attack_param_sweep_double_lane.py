import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import flow
reload(flow)

import Detectors.Deep_Learning.AutoEncoders.utils
reload(Detectors.Deep_Learning.AutoEncoders.utils)
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model

import torch

# Anti-Flow specific functions for  detection:

from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

import utils
reload(utils)

from utils import Bando_OVM_FTL

import os
import shutil

from Adversaries.controllers import car_following_adversarial

reload(car_following_adversarial)

from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact
from Adversaries.controllers.car_following_adversarial import ACC_Benign
from Adversaries.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked

from flow.controllers.lane_change_controllers import AILaneChangeController

from utils import run_ring_sim_variable_cfm

from Adversaries.controllers import car_following_adversarial

reload(car_following_adversarial)

import time

from utils import run_ring_sim_variable_cfm

def get_losses(timeseries_dict,model,warmup_steps=500,want_timeseries_plot=True):
    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in veh_ids:
        timeseries_list = []
        
        speed = timeseries_dict[veh_id][warmup_steps:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][warmup_steps:,2]
        rel_vel = timeseries_dict[veh_id][warmup_steps:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

        timeseries_list = [speed,accel,head_way,rel_vel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][warmup_steps:,0]
    
    #Get smoothed loss values:
    for veh_id in veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

    
    if(want_timeseries_plot):
        plt.figure()
        
        for veh_id in veh_ids:
            smoothed_loss = smoothed_losses[veh_id]
            if('FStop' in veh_id):
                plt.plot(smoothed_loss,'r')
            else:
                plt.plot(smoothed_loss,'b')
        
    return smoothed_losses

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

    num_human_drivers = 70

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
        
        lc_controller = (AILaneChangeController,{'left_delta':left_delta,
                                                 'right_delta':right_delta,
                                                 'left_beta':left_beta,
                                                 'right_beta':right_beta,
                                                 'switching_threshold':switching_threshold})
        
        driver_controller_list_with_attack.append([label,cfm_controller,lc_controller,1])

    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(6):
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

    driver_controller_list_with_attack.append([label_adv,adversary,2])


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

    driver_controller_list_with_attack.append([label_adv,adversary,2])


    return driver_controller_list_with_attack

def make_benign_driver_list(Total_Attack_Duration=3.0,attack_decel_rate = -.8):

    driver_controller_list = []

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

    num_human_drivers = 70

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
        
        lc_controller = (AILaneChangeController,{'left_delta':left_delta,
                                                 'right_delta':right_delta,
                                                 'left_beta':left_beta,
                                                 'right_beta':right_beta,
                                                 'switching_threshold':switching_threshold})
        
        driver_controller_list.append([label,cfm_controller,lc_controller,1])

    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(10):
        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list.append([label,cfm_controller,1])    

    return driver_controller_list

def run_sim_with_attack(Total_Attack_Duration,attack_decel_rate,emission_path):
    
    driver_controller_list_with_attack = make_mal_driver_list(Total_Attack_Duration,attack_decel_rate)
    
    sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=600,
                                                     sim_time=300,
                                                     num_lanes=2,
                                                     emission_path=emission_path)
    
    file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
    
    return file_path 

def get_losses_from_attack(Total_Attack_Duration,attack_decel_rate,model,delete_file=False):
    
    emission_path = run_sim_with_attack(Total_Attack_Duration,attack_decel_rate)
    
    sim_timeseries_dict_with_attack = visualize_ring.get_sim_timeseries(emission_path)
    
    losses_dict = get_losses(sim_timeseries_dict_with_attack,model,warmup_steps=500,want_timeseries_plot=False)
    
    if(delete_file):
        os.remove(emission_path)

    return losses_dict


# Ray helper function:

import ray
@ray.remote
def run_sim_with_attack_ray(Total_Attack_Duration,attack_decel_rate,emission_path):
    return run_sim_with_attack(Total_Attack_Duration,attack_decel_rate,emission_path)


def run_batch_sim_ray(Total_Attack_Duration,attack_decel_rate,emission_path,num_runs=10):

    sim_info_ids = []

    for i in range(num_runs):
        sim_info_ids.append(
            run_sim_with_attack_ray.remote(Total_Attack_Duration,attack_decel_rate,emission_path=emission_path)
            )

    file_path_list = ray.get(sim_info_ids)

    return file_path_list


def rename_file(file_path,file_name_no_version,emission_path):

    existing_files = os.listdir(emission_path)

    existing_file_versions = 0

    for file in existing_files:
        if(file_name_no_version in file):
            existing_file_versions += 1

    existing_file_versions += 1

    new_file_name_with_version = file_name_no_version+'_ver_'+str(existing_file_versions)+'.csv'

    file_destination = os.path.join(emission_path,new_file_name_with_version)

    #maps from emission_path to 
    shutil.move(file_path,file_destination)


def get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length=600):
    file_name_no_version = 'ring_'+str(ring_length)+'m_double_lane_TAD_'+str(Total_Attack_Duration)+'_ADR_'+str(attack_decel_rate)
    return file_name_no_version


def get_number_run_sims(Total_Attack_Duration,attack_decel_rate,emission_path):
    file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate)

    existing_files = os.listdir(emission_path)

    existing_file_versions = 0

    for file in existing_files:
        if(file_name_no_version in file):
            existing_file_versions += 1

    return existing_file_versions



if __name__ == '__main__':
    emission_path = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'

    Total_Attack_Duration_list = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    attack_decel_rate_list = [0.0,-0.25,-0.5,-0.75,-1.0]

    num_runs_desired = 10

    ray.init(num_cpus=5)


    for Total_Attack_Duration in Total_Attack_Duration_list:
        for attack_decel_rate in attack_decel_rate_list:

            num_runs = num_runs_desired - get_number_run_sims(Total_Attack_Duration,attack_decel_rate,emission_path)

            file_path_list = run_batch_sim_ray(Total_Attack_Duration,
                attack_decel_rate,
                emission_path=emission_path,
                num_runs=num_runs)

            file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate)

            for file_path in file_path_list:

                rename_file(file_path,file_name_no_version,emission_path)


    print('All simulations finished.')
