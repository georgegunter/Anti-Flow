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

from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact
from Adversaries.controllers.car_following_adversarial import ACC_Benign
from Adversaries.controllers.car_following_adversarial import ACC_Switched_Controller_Attacked

from flow.controllers.lane_change_controllers import AILaneChangeController

from utils import run_ring_sim_variable_cfm

import time

from utils import run_ring_sim_variable_cfm

import ray


def make_benign_driver_list_single_lane():

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

        driver_controller_list.append([label,cfm_controller,1])

    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(5):
        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list.append([label,cfm_controller,1])    


    return driver_controller_list


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

def run_sim_with_attack(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length):
    
    driver_controller_list_with_attack = make_mal_driver_list(Total_Attack_Duration,attack_decel_rate)
    
    sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=ring_length,
                                                     sim_time=300,
                                                     emission_path=emission_path)
    
    file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
    

    file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length=ring_length)


    file_path_new = rename_file(file_path,file_name_no_version,emission_path)


    return file_path_new


@ray.remote
def run_sim_with_attack_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length):
    return run_sim_with_attack(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length)


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

    return file_destination


def get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length):
    file_name_no_version = 'ring_'+str(ring_length)+'m_single_lane_TAD_'+str(Total_Attack_Duration)+'_ADR_'+str(attack_decel_rate)
    return file_name_no_version


def get_number_run_sims(Total_Attack_Duration,attack_decel_rate,emission_path):
    file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate)

    existing_files = os.listdir(emission_path)

    existing_file_versions = 0

    for file in existing_files:
        if(file_name_no_version in file):
            existing_file_versions += 1

    return existing_file_versions





def run_batch_sim_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length,num_runs=10):

    sim_info_ids = []

    for i in range(num_runs):
        sim_info_ids.append(
            run_sim_with_attack_ray.remote(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length)
            )

    file_path_list = ray.get(sim_info_ids)

    return file_path_list





if __name__ == '__main__':
    emission_path = '/Volumes/My Passport for Mac/single_lane_ring_road_sweep_ring_length'

    # ray.init(num_cpus=3)

    ring_lengths = [900,1000]

    sim_result_ids = []


    # Total_Attack_Duration = 0.0
    # attack_decel_rate = 0.0

    Total_Attack_Duration = 10.0
    attack_decel_rate = -1.0

    print('Simulation: strong attack')

    for ring_length in ring_lengths:

        run_batch_sim_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length,num_runs=9)

    print('Simulations finished.')




    # print('Simulation: benign')

    # # for ring_length in ring_lengths:

    # #     file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length)

    # #     try:
    # #         print('Ring length: '+str(ring_length))
    # #         sim_result_ids.append(run_sim_with_attack_ray.remote(Total_Attack_Duration,
    # #             attack_decel_rate,
    # #             emission_path,
    # #             ring_length))
    # #     except:
    # #         print('Issue with simulation on ring length '+str(ring_length))

    # # sim_results = ray.get(sim_result_ids)


    # print('Benign simulations finished.')


    # Total_Attack_Duration = 5.0
    # attack_decel_rate = -.25

    # print('Simulation: weak attack')

    # for ring_length in ring_lengths:

    #     file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length)

    #     try:
    #         print('Ring length: '+str(ring_length))
    #         sim_result_ids.append(run_sim_with_attack_ray.remote(Total_Attack_Duration,
    #             attack_decel_rate,
    #             emission_path,
    #             ring_length))
    #     except:
    #         print('Issue with simulation on ring length '+str(ring_length))

    # sim_results = ray.get(sim_result_ids)


    # print('Weak attack simulations finished.')





    # Total_Attack_Duration = 10.0
    # attack_decel_rate = -1.0

    # print('Simulation: strong attack')

    # for ring_length in ring_lengths:

    #     file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_decel_rate,ring_length)

    #     try:
    #         print('Ring length: '+str(ring_length))
    #         sim_result_ids.append(run_sim_with_attack_ray.remote(Total_Attack_Duration,
    #             attack_decel_rate,
    #             emission_path,
    #             ring_length))
    #     except:
    #         print('Issue with simulation on ring length '+str(ring_length))

    # sim_results = ray.get(sim_result_ids)


    # print('Strong attack simulations finished.')















    