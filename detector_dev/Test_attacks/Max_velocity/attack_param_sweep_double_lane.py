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


from flow.controllers.lane_change_controllers import AILaneChangeController


# For simulation:
from detector_dev.Process_RingRoad_Simulation.utils import run_ring_sim_variable_cfm,Bando_OVM_FTL



import Detectors.Deep_Learning.AutoEncoders.utils
reload(Detectors.Deep_Learning.AutoEncoders.utils)
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model

import torch

# Anti-Flow specific functions for  detection:

from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

import os
import shutil

import Adversaries.controllers.car_following_adversarial
from Adversaries.controllers.car_following_adversarial import *


import time




class Bando_OVM_FTL(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 delay=0.0,
                 noise=0.0,
                 fail_safe=None,
                 a=0.8,
                 b=20.0,
                 s0=1.0,
                 s1=2.0,
                 Vm=15.0):
        #Inherit the base controller:
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=delay,
            fail_safe=fail_safe,
            noise=noise)
        
        # Model parameters, which can be changed at initialization:
        self.Vm = Vm
        self.s0 = s0
        self.s1 = s1
        self.a = a
        self.b = b
        
    def get_accel(self, env):
        """This function is queried during simulation
           to acquire an acceleration value:"""
        # env contains all information on the simulation, and 
        # can be queried to get the state of different vehicles.
        # We assume this vehicle has access only to its own state,
        # and the position/speed of the vehicle ahead of it. 
        lead_id = env.k.vehicle.get_leader(self.veh_id) #Who is the leader
        v_l = env.k.vehicle.get_speed(lead_id) #Leader speed
        v = env.k.vehicle.get_speed(self.veh_id) #vehicle's own speed
        s = env.k.vehicle.get_headway(self.veh_id) #inter-vehicle spacing to leader

        # We build this model off the popular Bando OV-FTL model:
        v_opt = self.OV(s)
        ftl = self.FTL(v,v_l,s)
        u = self.a*(v_opt-v) + self.b*ftl
        
        return u #return the acceleration that is set above.
        
    def get_custom_accel(self,this_vel, lead_vel, h):
        """This function can be queried at any time,
           and is useful for analyzing controller
           behavior outside of a sim."""

        v = this_vel
        v_l = lead_vel
        s = h

        v_opt = self.OV(s)
        ftl = self.FTL(v,v_l,s)
        u = self.a*(v_opt-v) + self.b*ftl
        return u
    
    def OV(self,s):
        return self.Vm*((np.tanh(s/self.s0-self.s1)+np.tanh(self.s1))/(1+np.tanh(self.s1)))
    
    def FTL(self,v,v_l,s):
        return (v_l-v)/(s**2)


def make_mal_driver_list(Total_Attack_Duration=3.0,attack_vmax = -.8):

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

    num_benign_ACCs = 6

    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0

    for i in range(num_benign_ACCs):
        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list_with_attack.append([label,cfm_controller,1])  

    ########## ADVERSARIES: ##########

    V_m_comp = attack_vmax

    want_multiple_attacks=True
    warmup_steps = 500
    SS_Threshold_min = 60
    display_attack_info = True

    num_adversries = 4

    for i in range(num_adversries):

        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean



        adversary = (ACC_comp_overwrite_Vm, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                        'want_multiple_attacks':want_multiple_attacks,
                                                        'Total_Attack_Duration':Total_Attack_Duration,
                                                        'V_m_comp':V_m_comp,
                                                        'warmup_steps':warmup_steps,
                                                        'SS_Threshold_min':SS_Threshold_min,
                                                        'display_attack_info':display_attack_info})

        
        ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

        label_adv = 'V_max_comp_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_vmaxcomp_'+str(np.round(attack_vmax,2))

        label_adv = label_adv + ACC_label

        driver_controller_list_with_attack.append([label_adv,adversary,1])

    return driver_controller_list_with_attack

def run_sim_with_attack(Total_Attack_Duration,attack_vmax,emission_path):
    
    driver_controller_list_with_attack = make_mal_driver_list(Total_Attack_Duration,attack_vmax)
    
    sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
                                                     ring_length=600,
                                                     sim_time=300,
                                                     num_lanes=2,
                                                     emission_path=emission_path)
    
    file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
    
    return file_path 

# Ray helper function:

import ray
@ray.remote
def run_sim_with_attack_ray(Total_Attack_Duration,attack_vmax,emission_path):
    return run_sim_with_attack(Total_Attack_Duration,attack_vmax,emission_path)


def run_batch_sim_ray(Total_Attack_Duration,attack_vmax,emission_path,num_runs=10):

    sim_info_ids = []

    for i in range(num_runs):
        sim_info_ids.append(
            run_sim_with_attack_ray.remote(Total_Attack_Duration,attack_vmax,emission_path=emission_path)
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


def get_file_name_no_version(Total_Attack_Duration,attack_vmax,ring_length=600):
    file_name_no_version = 'ring_'+str(ring_length)+'m_single_lane_TAD_'+str(Total_Attack_Duration)+'_vmcomp_'+str(attack_vmax)
    return file_name_no_version


def get_number_run_sims(Total_Attack_Duration,attack_vmax,emission_path):

    existing_files = os.listdir(emission_path)

    existing_file_versions = 0

    for file in existing_files:
        if(file_name_no_version in file):
            existing_file_versions += 1

    return existing_file_versions



if __name__ == '__main__':
    emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/max_velocity/double_lane_ring_road_attack_parameter_sweep'

    Total_Attack_Duration_list = [2.0,4.0,6.0,8.0,10.0]
    attack_vmax_list = [1.0,3.0,5.0,7.0,10.0]


    want_single_run = False
        
    if(want_single_run):

        print('RUNNING SINGLE SIMULATION.')

        file_path = run_sim_with_attack(Total_Attack_Duration_list[0],attack_vmax_list[0],emission_path)

        file_name_no_version = get_file_name_no_version(Total_Attack_Duration_list[0],attack_vmax_list[0])

        rename_file(file_path,file_name_no_version,emission_path)


    want_run_sweep = True

    if(want_run_sweep):

        print('RUNNING PARAMETER SWEEP.')

        num_runs_desired = 5

        ray.init(num_cpus=4)


        for Total_Attack_Duration in Total_Attack_Duration_list:
            for attack_vmax in attack_vmax_list:

                num_runs = num_runs_desired

                file_path_list = run_batch_sim_ray(Total_Attack_Duration,
                    attack_vmax,
                    emission_path=emission_path,
                    num_runs=num_runs)

                file_name_no_version = get_file_name_no_version(Total_Attack_Duration,attack_vmax)

                for file_path in file_path_list:

                    rename_file(file_path,file_name_no_version,emission_path)


        print('All simulations finished.')
