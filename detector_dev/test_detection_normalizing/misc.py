import os

import numpy as np
import matplotlib.pyplot as plt

from detector_dev.utils import run_ring_sim_variable_cfm,Bando_OVM_FTL
from Adversaries.controllers.car_following_adversarial import ACC_Benign,ACC_Switched_Controller_Attacked
from flow.controllers.velocity_controllers import FollowerStopper

from flow.controllers.lane_change_controllers import AILaneChangeController,StaticLaneChanger


import detector_dev.utils as utils
import Data_Processing.sim_processing_utils as sim_processing_utils
from Data_Processing.sim_processing_utils import get_trajectory_timeseries
from importlib import reload


import torch

# Anti-Flow specific functions for  detection:
from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder
from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth
from Detectors.Deep_Learning.AutoEncoders.utils import sliding_window_mult_feat
from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X

import sys



def sim_attacker_follows_smoother(v_des):
    driver_controller_list = []

    #human drivers:
    num_human_drivers = 38

    a=0.666
    b=21.6
    s0=2.21
    s1=2.82
    Vm=18.94

    label = 'bando_ftl_ovm_a'+str(a)+'_b'+str(b)+'_s0'+str(s0)+'_s1'+str(s1)+'_Vm'+str(Vm)
    cfm_controller = (Bando_OVM_FTL,{'a':a,'b':b,'s0':s0,'s1':s1,'Vm':Vm,'noise':0.1})
    driver_controller_list.append([label,cfm_controller,num_human_drivers])

    #attack vehicle:
    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0
    k_1 = k_1_mean# + np.random.normal(0,0.2)
    k_2 = k_2_mean# + np.random.normal(0,0.2)
    h = h_mean# + np.random.normal(0,0.2)
    V_m = V_m_mean# + np.random.normal(0,1.0)
    d_min = d_min_mean


    want_multiple_attacks = True
    Total_Attack_Duration = 10
    attack_decel_rate = -1.0
    warmup_steps = 0
    SS_Threshold_min = 30.0
    SS_Threshold_range = 0.0
    display_attack_info = False

    adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                    'want_multiple_attacks':want_multiple_attacks,
                                                    'Total_Attack_Duration':Total_Attack_Duration,
                                                    'attack_decel_rate':attack_decel_rate,
                                                    'warmup_steps':warmup_steps,
                                                    'SS_Threshold_min':SS_Threshold_min,
                                                    'SS_Threshold_range':SS_Threshold_range,
                                                    'display_attack_info':display_attack_info})

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    driver_controller_list.append([label_adv,adversary,1])


    # follower-stopper
    fstop_controller = (FollowerStopper,{'v_des':v_des})
    label = 'FollowerStopper_Vdes_'+str(v_des)
    driver_controller_list.append([label,fstop_controller,1])



    # Simulation components:

    ring_length = 500
    sim_time = 200


    want_render = False
    want_shuffle = False


    sim_res_list_1_smoother_1_attacker = utils.run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list,
                                                                         want_render=want_render,
                                                                         want_shuffle=want_shuffle,
                                                                         ring_length=ring_length,
                                                                         sim_time=sim_time)


    emission_path = os.path.join(os.getcwd(),sim_res_list_1_smoother_1_attacker[1])

    trajectory_dict_1_smoother_1_attacker_ordered = get_trajectory_timeseries(csv_path = emission_path)

    return trajectory_dict_1_smoother_1_attacker_ordered


def sim_attacker_precedes_smoother(v_des):
    driver_controller_list = []

    #human drivers:
    num_human_drivers = 38

    a=0.666
    b=21.6
    s0=2.21
    s1=2.82
    Vm=18.94

    label = 'bando_ftl_ovm_a'+str(a)+'_b'+str(b)+'_s0'+str(s0)+'_s1'+str(s1)+'_Vm'+str(Vm)
    cfm_controller = (Bando_OVM_FTL,{'a':a,'b':b,'s0':s0,'s1':s1,'Vm':Vm,'noise':0.1})
    driver_controller_list.append([label,cfm_controller,num_human_drivers])


    # follower-stopper
    fstop_controller = (FollowerStopper,{'v_des':v_des})
    label = 'FollowerStopper_Vdes_'+str(v_des)
    driver_controller_list.append([label,fstop_controller,1])

    #attack vehicle:
    k_1_mean = 1.5
    k_2_mean = 0.2
    h_mean = 1.8
    V_m_mean = 15.0
    d_min_mean = 10.0
    k_1 = k_1_mean# + np.random.normal(0,0.2)
    k_2 = k_2_mean# + np.random.normal(0,0.2)
    h = h_mean# + np.random.normal(0,0.2)
    V_m = V_m_mean# + np.random.normal(0,1.0)
    d_min = d_min_mean


    want_multiple_attacks = True
    Total_Attack_Duration = 10
    attack_decel_rate = -1.0
    warmup_steps = 0
    SS_Threshold_min = 30.0
    SS_Threshold_range = 0.0
    display_attack_info = False

    adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                    'want_multiple_attacks':want_multiple_attacks,
                                                    'Total_Attack_Duration':Total_Attack_Duration,
                                                    'attack_decel_rate':attack_decel_rate,
                                                    'warmup_steps':warmup_steps,
                                                    'SS_Threshold_min':SS_Threshold_min,
                                                    'SS_Threshold_range':SS_Threshold_range,
                                                    'display_attack_info':display_attack_info})

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    driver_controller_list.append([label_adv,adversary,1])


    # Simulation components:

    ring_length = 500
    sim_time = 200


    want_render = False
    want_shuffle = False


    sim_res_list_1_smoother_1_attacker = utils.run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list,
                                                                         want_render=want_render,
                                                                         want_shuffle=want_shuffle,
                                                                         ring_length=ring_length,
                                                                         sim_time=sim_time)


    emission_path = os.path.join(os.getcwd(),sim_res_list_1_smoother_1_attacker[1])

    trajectory_dict_1_smoother_1_attacker_ordered = get_trajectory_timeseries(csv_path = emission_path)

    return trajectory_dict_1_smoother_1_attacker_ordered


def sim_only_HVs(num_lanes=1):

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

    num_human_drivers = int(40*num_lanes)

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


    ring_length = 500
    sim_time = 200


    want_render = False
    want_shuffle = False


    sim_res_list = utils.run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list,
                                                                         want_render=want_render,
                                                                         want_shuffle=want_shuffle,
                                                                         ring_length=ring_length,
                                                                         sim_time=sim_time)


    emission_path = os.path.join(os.getcwd(),sim_res_list[1])

    trajectory_dict = get_trajectory_timeseries(csv_path = emission_path)

    return trajectory_dict, emission_path



def sim_benign_mixed(num_lanes=1):

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

    num_human_drivers = int(35*num_lanes)

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


    lc_controller = (StaticLaneChanger,{})


    for i in range(5):

        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list.append([label,cfm_controller,lc_controller,1])


    ring_length = 500
    sim_time = 200


    want_render = False
    want_shuffle = False


    sim_res_list = utils.run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list,
                                                                         want_render=want_render,
                                                                         want_shuffle=want_shuffle,
                                                                         ring_length=ring_length,
                                                                         sim_time=sim_time)


    emission_path = os.path.join(os.getcwd(),sim_res_list[1])

    trajectory_dict = get_trajectory_timeseries(csv_path = emission_path)

    return trajectory_dict, emission_path



def sim_attacked_mixed(Total_Attack_Duration,attack_decel_rate):

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

    num_human_drivers = int(35)

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


    lc_controller = (StaticLaneChanger,{})


    for i in range(3):

        k_1 = k_1_mean + np.random.normal(0,0.2)
        k_2 = k_2_mean + np.random.normal(0,0.2)
        h = h_mean + np.random.normal(0,0.2)
        V_m = V_m_mean + np.random.normal(0,1.0)
        d_min = d_min_mean

        label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
        cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
        driver_controller_list.append([label,cfm_controller,lc_controller,1])



    k_1 = k_1_mean + np.random.normal(0,0.2)
    k_2 = k_2_mean + np.random.normal(0,0.2)
    h = h_mean + np.random.normal(0,0.2)
    V_m = V_m_mean + np.random.normal(0,1.0)
    d_min = d_min_mean


    want_multiple_attacks = True
    # Total_Attack_Duration = 10
    # attack_decel_rate = -1.0
    warmup_steps = 0
    SS_Threshold_min = 30.0
    SS_Threshold_range = 0.0
    display_attack_info = True

    adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
                                                    'want_multiple_attacks':want_multiple_attacks,
                                                    'Total_Attack_Duration':Total_Attack_Duration,
                                                    'attack_decel_rate':attack_decel_rate,
                                                    'warmup_steps':warmup_steps,
                                                    'SS_Threshold_min':SS_Threshold_min,
                                                    'SS_Threshold_range':SS_Threshold_range,
                                                    'display_attack_info':display_attack_info})

    label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

    driver_controller_list.append([label_adv,adversary,lc_controller,1])



    # Add two compromised ACCs:

    # for i in range(2):

    #     k_1 = k_1_mean + np.random.normal(0,0.2)
    #     k_2 = k_2_mean + np.random.normal(0,0.2)
    #     h = h_mean + np.random.normal(0,0.2)
    #     V_m = V_m_mean + np.random.normal(0,1.0)
    #     d_min = d_min_mean

    #     want_multiple_attacks = True
    #     # Total_Attack_Duration = 10 # These are passed to the function
    #     # attack_decel_rate = -1.0
    #     warmup_steps = 0
    #     SS_Threshold_min = 30.0
    #     SS_Threshold_range = 30.0
    #     display_attack_info = False

    #     adversary = (ACC_Switched_Controller_Attacked, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
    #                                                     'want_multiple_attacks':want_multiple_attacks,
    #                                                     'Total_Attack_Duration':Total_Attack_Duration,
    #                                                     'attack_decel_rate':attack_decel_rate,
    #                                                     'warmup_steps':warmup_steps,
    #                                                     'SS_Threshold_min':SS_Threshold_min,
    #                                                     'SS_Threshold_range':SS_Threshold_range,
    #                                                     'display_attack_info':display_attack_info})

    #     label_adv = 'RDA_adv_TDA_'+str(np.round(Total_Attack_Duration,2))+'_ADR_'+str(np.round(attack_decel_rate,2))


    #     driver_controller_list.append([label_adv,adversary,lc_controller,1])


    ring_length = 500
    sim_time = 200


    want_render = False
    want_shuffle = False


    sim_res_list = utils.run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list,
                                                                         want_render=want_render,
                                                                         want_shuffle=want_shuffle,
                                                                         ring_length=ring_length,
                                                                         sim_time=sim_time)


    emission_path = os.path.join(os.getcwd(),sim_res_list[1])

    trajectory_dict = get_trajectory_timeseries(csv_path = emission_path)

    return trajectory_dict, emission_path














def make_timeseries_list(trajectory_dict):
    timeseries_list = []
    for veh_id in trajectory_dict:
        trajectory_samples = []
        trajectory_data = trajectory_dict[veh_id]
        
        speed = trajectory_data[:,1]
        accel = np.gradient(speed,.1)
        head_way = trajectory_data[:,2]
        rel_vel = trajectory_data[:,3]
        
        trajectory_samples.append(speed)
        trajectory_samples.append(accel)
        trajectory_samples.append(head_way)
        trajectory_samples.append(rel_vel)
        
        timeseries_list.append(trajectory_samples)
    return timeseries_list







# Detection:



def make_timeseries_list(trajectory_dict):
    timeseries_list = []
    for veh_id in trajectory_dict:
        trajectory_samples = []
        trajectory_data = trajectory_dict[veh_id]
        
        speed = trajectory_data[:,1]
        accel = np.gradient(speed,.1)
        head_way = trajectory_data[:,2]
        rel_vel = trajectory_data[:,3]
        
        trajectory_samples.append(speed)
        trajectory_samples.append(accel)
        trajectory_samples.append(head_way)
        trajectory_samples.append(rel_vel)
        
        timeseries_list.append(trajectory_samples)
    return timeseries_list

def make_timeseries_list_normalized(trajectory_dict,normalizing_factors=None):
    timeseries_list_non_normalized = make_timeseries_list(trajectory_dict)
    num_timeseries = len(timeseries_list_non_normalized[0])

    if(normalizing_factors is None):
        normalizing_factors = np.zeros(num_timeseries,)
        
        for t_list in timeseries_list_non_normalized:
            for i in range(num_timeseries):
                max_abs_val = np.max(np.abs(t_list[i]))
                if(max_abs_val > normalizing_factors[i]): normalizing_factors[i] = max_abs_val
    
    normalized_timeseries_list = []
    
    for t_list in timeseries_list_non_normalized:
        t_list_normed = deepcopy(t_list)
        for i in range(num_timeseries):
            t_list_normed[i] = t_list_normed[i]/normalizing_factors[i]
        normalized_timeseries_list.append(t_list_normed)
            
    return normalized_timeseries_list,normalizing_factors




def get_losses_per_vehicle(timeseries_list,trajectory_dict,time,model):

    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(trajectory_dict.keys())
    
    # timeseries_list = make_timeseries_list(trajectory_dict)
    
    for veh_id in trajectory_dict:
        timeseries_sample = timeseries_list[num_veh_processed]

        _,loss = sliding_window_mult_feat(model,timeseries_sample)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(trajectory_dict.keys())

    #Get smoothed loss values:
    for veh_id in trajectory_dict:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

    return smoothed_losses


