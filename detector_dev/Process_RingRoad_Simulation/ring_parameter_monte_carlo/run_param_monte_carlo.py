import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

import flow


import Detectors.Deep_Learning.AutoEncoders.utils
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model

import torch

# Anti-Flow specific functions for  detection:

from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

import utils

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

from copy import deepcopy


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
        b = b_mean + np.random.normal(0,0.1)
        s0 = s0_mean + np.random.normal(0,0.1)
        s1 = s1_mean + np.random.normal(0,0.1)
        Vm = Vm_mean + np.random.normal(0,0.1)

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
        k_1 = k_1_mean + np.random.normal(0,0.1)
        k_2 = k_2_mean + np.random.normal(0,0.1)
        h = h_mean + np.random.normal(0,0.1)
        V_m = V_m_mean + np.random.normal(0,0.1)
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
        a = a_mean*(1+np.random.normal(0,0.1))
        b = b_mean*(1+np.random.normal(0,0.1))
        s0 = s0_mean*(1+np.random.normal(0,0.1))
        s1 = s1_mean*(1+np.random.normal(0,0.1))
        Vm = Vm_mean*(1+np.random.normal(0,0.1))

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
        k_1 = k_1_mean*(1+np.random.normal(0,0.1))
        k_2 = k_2_mean*(1+np.random.normal(0,0.1))
        h = h_mean*(1+np.random.normal(0,0.1))
        V_m = V_m_mean*(1+np.random.normal(0,0.1))
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list_with_attack.append([label,cfm_controller,1]) 



    # v_des = 10.0
    # braking_period = 5.0
    # braking_rate = -3.0

    k_1 = k_1_mean*(1+np.random.normal(0,0.1))
    k_2 = k_2_mean*(1+np.random.normal(0,0.1))
    h = h_mean*(1+np.random.normal(0,0.1))
    V_m = V_m_mean*(1+np.random.normal(0,0.1))
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
    
    ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    label_adv = label_adv + ACC_label

    driver_controller_list_with_attack.append([label_adv,adversary,1])


    k_1 = k_1_mean*(1+np.random.normal(0,0.1))
    k_2 = k_2_mean*(1+np.random.normal(0,0.1))
    h = h_mean*(1+np.random.normal(0,0.1))
    V_m = V_m_mean*(1+np.random.normal(0,0.1))
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


    return driver_controller_list_with_attack

def make_driver_list_benign(driver_controller_list_with_attack):
    benign_driver_list = deepcopy(driver_controller_list_with_attack)

    for i in range(len(benign_driver_list)):
        if('RDA' in benign_driver_list[i][0]):
            benign_driver_list[i][1][1]['Total_Attack_Duration'] = 0.0
            benign_driver_list[i][1][1]['attack_decel_rate'] = 0.0

            # rename the cfm id:
            temp_str = deepcopy(benign_driver_list[i][0])

            j = 12
            while(temp_str[j] != '_'): j+=1
            temp_str = temp_str[:12] + '0.0' + temp_str[j:]

            j = 12
            while(temp_str[j:j+3] != 'ADR'): j+=1
            j += 4
            k = j
            while(temp_str[k]!='_'):k+=1
            temp_str = temp_str[:j] + '0.0' + temp_str[k:]
            benign_driver_list[i][0] = temp_str


    return benign_driver_list




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




def run_sims(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length):
    
    # Make sims with parameters the same, except for no attack in one:
    driver_controller_list_with_attack = make_mal_driver_list(Total_Attack_Duration,attack_decel_rate)
    
    driver_controller_list_benign = make_driver_list_benign(driver_controller_list_with_attack)

    print('Running attacked sim.')
    sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=ring_length,
                                                     sim_time=300,
                                                     emission_path=emission_path)
    print('Running non attacked sim.')
    sim_res_list_benign = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_benign,
                                                     ring_length=ring_length,
                                                     sim_time=300,
                                                     emission_path=emission_path)

    attack_file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
    attack_file_name_no_version = get_file_name_attack_no_version(Total_Attack_Duration,attack_decel_rate,ring_length=ring_length)
    attack_file_path_new = rename_file(attack_file_path,attack_file_name_no_version,emission_path)


    benign_file_path = os.path.join(os.getcwd(),sim_res_list_benign[1])
    benign_file_name_no_version = get_file_name_benign_no_version(Total_Attack_Duration,attack_decel_rate,ring_length=ring_length)
    benign_file_path_new = rename_file(benign_file_path,benign_file_name_no_version,emission_path)

    return attack_file_path_new,benign_file_path_new


@ray.remote
def run_sims_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length):
    return run_sims(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length)


def get_file_name_attack_no_version(Total_Attack_Duration,attack_decel_rate,ring_length):
    file_name_no_version = 'ring_'+str(ring_length)+'m_single_lane_TAD_'+str(Total_Attack_Duration)+'_ADR_'+str(attack_decel_rate)
    return file_name_no_version

def get_file_name_benign_no_version(Total_Attack_Duration,attack_decel_rate,ring_length):
    file_name_no_version = 'ring_'+str(ring_length)+'m_single_lane_benign'
    return file_name_no_version


def get_number_run_attack_sims(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length=600):
    file_name_no_version = get_file_name_attack_no_version(Total_Attack_Duration,attack_decel_rate,ring_length)

    existing_files = os.listdir(emission_path)

    existing_file_versions = 0

    for file in existing_files:
        if(file_name_no_version in file):
            existing_file_versions += 1

    return existing_file_versions



def run_batch_sims_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length,num_runs=10):

    sim_info_ids = []

    number_sim_runs_existing = get_number_run_attack_sims(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length)

    num_runs = num_runs - number_sim_runs_existing

    for i in range(num_runs):
        sim_info_ids.append(
            run_sims_ray.remote(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length)
            )

    file_path_list = ray.get(sim_info_ids)

    return file_path_list



if __name__ == '__main__':
    num_runs = 100

    want_run_strong_attack = False
    if(want_run_strong_attack):
        emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/strong_attack'
        Total_Attack_Duration=10.0
        attack_decel_rate=-.5
        ring_length=600


        ray.init(num_cpus=4)

        res = run_batch_sims_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length,num_runs)

        print('Finished with simulation batch.')


    want_run_medium_attack = False
    if(want_run_medium_attack):
        emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/medium_attack'
        Total_Attack_Duration=5.0
        attack_decel_rate=-.25
        ring_length=600


        ray.init(num_cpus=4)

        res = run_batch_sims_ray(Total_Attack_Duration,attack_decel_rate,emission_path,ring_length,num_runs)

        print('Finished with simulation batch.')




