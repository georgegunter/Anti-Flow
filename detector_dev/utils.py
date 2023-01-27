#Import different needed quantities from Flow:
import flow
from importlib import reload
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

from sklearn.cluster import KMeans


# from Adversaries.controllers.car_following_adversarial import FollowerStopper_Overreact

from Adversaries.controllers.base_controller import BaseController


import os
import numpy as np

def run_ring_sim_variable_cfm(ring_length=300,
    driver_controller_list=None,
    num_lanes=1,
    sim_time=500,
    want_render=False,
    want_shuffle=True,
    emission_path='data'):
    

    #There is an updated version of this that

    #Simulation parameters:
    time_step = 0.1 #In seconds, how far each step of the simulation goes.
    sim_horizon = int(np.floor(sim_time/time_step)) #How many simulation steps will be taken -> Runs for 300 seconds

    #initialize the simulation using above parameters:
    traffic_lights = TrafficLightParams() #This is empty, so no traffic lights are used.
    initial_config = InitialConfig(shuffle=want_shuffle,spacing="uniform", perturbation=1) #Vehicles start out evenly spaced.
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

def Model_Based_Ring_CFM_SysID(csv_path):
    '''TO DO'''
    return None


import numpy as np

import time

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

import flow.visualize.visualize_ring as visualize_ring

from flow.visualize.visualize_ring import get_measured_leader,get_rel_dist_to_measured_leader,get_vel_of_measured_leader

from copy import deepcopy

import sys


def train_ring_relative_detector(GPS_penetration_rate,ring_length,emission_path,n_epoch=200):
    # This is specific to training on a ring, it's slightly more optimized than the version
    # below.

    warmup_period = 500 #Wait until there's a well developed wave

    timeseries_dict = visualize_ring.get_sim_timeseries(csv_path=emission_path,warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())

    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)

    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    ring_sim_dict = visualize_ring.get_sim_data_dict_ring(csv_path=emission_path,warmup_period=warmup_period)

    timeseries_list = []

    for veh_id in measured_veh_ids:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        measured_leader = get_measured_leader(ring_sim_dict,veh_id,measured_veh_ids)
        leader_dist = get_rel_dist_to_measured_leader(ring_sim_dict,veh_id,measured_leader)
        leader_vel = get_vel_of_measured_leader(ring_sim_dict,veh_id,measured_leader)
        
        timeseries_list.append([speed,accel,leader_dist,leader_vel])

    train_X = make_train_X(timeseries_list)

    model = get_cnn_lstm_ae_model(n_features=4)

    model_file_name = 'ringlength'+str(ring_length)+'_'+str(GPS_penetration_rate)+'percentGPS'

    print('Model: '+model_file_name)

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model

def train_ring_nonrelative_detector(GPS_penetration_rate,ring_length,emission_path,n_epoch=200):

    warmup_period = 500 #Wait until there's a well developed wave

    timeseries_dict = visualize_ring.get_sim_timeseries(csv_path=emission_path,warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())

    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)

    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    ring_sim_dict = visualize_ring.get_sim_data_dict_ring(csv_path=emission_path,warmup_period=warmup_period)

    timeseries_list = []

    for veh_id in measured_veh_ids:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        # measured_leader = get_measured_leader(ring_sim_dict,veh_id,measured_veh_ids)
        # leader_dist = get_rel_dist_to_measured_leader(ring_sim_dict,veh_id,measured_leader)
        # leader_vel = get_vel_of_measured_leader(ring_sim_dict,veh_id,measured_leader)
        
        timeseries_list.append([speed,accel])

    train_X = make_train_X(timeseries_list)

    model = get_cnn_lstm_ae_model(n_features=2)

    model_file_name = 'ringlength'+str(ring_length)+'_'+str(GPS_penetration_rate)+'percentGPS_nonrelative'

    print('Model: '+model_file_name)

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model

def assess_relative_model_on_attack(emission_path,GPS_penetration_rate,model,want_timeseries_plot=False,warmup_period=500.0):
    print('Finding losses on attack simulation.')

    timeseries_dict = visualize_ring.get_sim_timeseries(csv_path=emission_path,
                                                                  warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())

    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)
    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    ring_sim_dict = visualize_ring.get_sim_data_dict_ring(csv_path=emission_path,
                                                           warmup_period=warmup_period)
    adv_label = 'FStop' #
    for veh_id in veh_ids:
        if(adv_label in veh_id and veh_id not in measured_veh_ids):
            measured_veh_ids.append(veh_id)
            
    print('Number of vehicles measured: '+str(num_measured_vehicle_ids))
        
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in measured_veh_ids:

        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        measured_leader = get_measured_leader(ring_sim_dict,veh_id,measured_veh_ids)
        leader_dist = get_rel_dist_to_measured_leader(ring_sim_dict,veh_id,measured_leader)
        leader_vel = get_vel_of_measured_leader(ring_sim_dict,veh_id,measured_leader)

        timeseries_list = [speed,accel,leader_dist,leader_vel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(measured_veh_ids)
    time = timeseries_dict[measured_veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in measured_veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

    
    if(want_timeseries_plot):
        plt.figure()
        
        for veh_id in measured_veh_ids:
            smoothed_loss = smoothed_losses[veh_id]
            if(adv_label in veh_id):
                plt.plot(smoothed_loss,'r')
            else:
                plt.plot(smoothed_loss,'b')
        
    return smoothed_losses

def assess_nonrelative_model_on_attack(emission_path,GPS_penetration_rate,model,want_timeseries_plot=True):
    print('Finding losses on attack simulation.')
    timeseries_dict = visualize_ring.get_sim_timeseries(csv_path=emission_path,
                                                                  warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())

    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)
    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    ring_sim_dict = visualize_ring.get_sim_data_dict_ring(csv_path=emission_path,
                                                           warmup_period=warmup_period)
    adv_label = 'FStop' #
    for veh_id in veh_ids:
        if(adv_label in veh_id and veh_id not in measured_veh_ids):
            measured_veh_ids.append(veh_id)
            
    print('Number of vehicles measured: '+str(num_measured_vehicle_ids))
        
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in measured_veh_ids:

        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)

        timeseries_list = [speed,accel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1
        print('Vehicles processed: '+str(num_veh_processed))

    print('Finished.')
    
    smoothed_losses = dict.fromkeys(measured_veh_ids)
    time = timeseries_dict[measured_veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in measured_veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

    print('Smoothed losses found.')
    
    if(want_timeseries_plot):
        plt.figure()
        
        for veh_id in measured_veh_ids:
            smoothed_loss = smoothed_losses[veh_id]
            if(adv_label in veh_id):
                plt.plot(smoothed_loss,'r')
            else:
                plt.plot(smoothed_loss,'b')
        
    return smoothed_losses



import csv

# Very general scripts for training detectors which use relative info:
def load_timeseries_dict_for_relative(csv_path,warmup_period=0.0):

    row_num = 1
    curr_veh_id = 'id'
    sim_dict = {}
    curr_veh_data = []

    with open(csv_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if(row_num > 1):
                # Don't read header
                if(curr_veh_id != row[1]):
                    #Add in new data to the dictionary:
                    
                    #Store old data:
                    if(len(curr_veh_data)>0):
                        sim_dict[curr_veh_id] = curr_veh_data
                    #Rest where data is being stashed:
                    curr_veh_data = []
                    curr_veh_id = row[1] # Set new veh id
                    #Allocate space for storing:
                    sim_dict[curr_veh_id] = []

                curr_veh_id = row[1]
                time = float(row[0])
                if(time > warmup_period):
                    # data = [time,speed,headway,leader_rel_speed,leader_id]
                    data = [row[0],row[4],row[5],row[8],row[6]]
                    curr_veh_data.append(data)
            row_num += 1

        #Add the very last vehicle's information:
        sim_dict[curr_veh_id] = curr_veh_data
        print('Data loaded.')
    return sim_dict

def get_time_indexed_dict(timeseries_dict,step_size=0.1):
    time_indexed_dict = {}

    for veh_id in timeseries_dict:
        data = timeseries_dict[veh_id]
        for i in range(len(data)):
            t = float(data[i][0])

            step_number = int(np.round(t,2)/step_size)

            if(step_number not in time_indexed_dict):
                time_indexed_dict[step_number] = {}
                time_indexed_dict[step_number][veh_id] = data[i]
            else:
                time_indexed_dict[step_number][veh_id] = data[i]

    return time_indexed_dict

def get_measured_veh_ids_at_GPS_rate(veh_ids,GPS_penetration_rate):
    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)

    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    return measured_veh_ids

def get_relative_vehicle_data(timeseries_dict,time_indexed_dict,measured_veh_ids,step_size=0.1):
    relative_vehicle_data_dict = dict.fromkeys(measured_veh_ids)

    for veh_id in measured_veh_ids:

        rel_data_list = []

        all_veh_data = timeseries_dict[veh_id]
        for i in range(len(all_veh_data)):

            data = all_veh_data[i]

            t = float(data[0])
            step_number = int(np.round(t,2)/step_size)
            # Find the effective leader:
            curr_leader_id = data[-1]

            time_indexed_data = time_indexed_dict[step_number] #indexed by vehicle ids

            rel_distance = 0.0

            while(curr_leader_id not in measured_veh_ids and curr_leader_id is not None):
                # Add up additional headway:
                rel_distance += float(time_indexed_data[curr_leader_id][2])

                # Look at where the next vehicle in front points:
                curr_leader_id = time_indexed_data[curr_leader_id][-1]

                # If there is no leader it means the vehicle is at the end of the road.

            if(curr_leader_id is None):
                rel_distance = 1000.0
                rel_velocity = 0.0

            else:
                rel_velocity = float(time_indexed_data[curr_leader_id][1]) - float(data[1]) 
                rel_data_list.append([rel_distance,rel_velocity])

        relative_vehicle_data_dict[veh_id] = rel_data_list

    return relative_vehicle_data_dict


def train_relative_detector(GPS_penetration_rate,emission_path,model_name='partial_detector',warmup_period=100,n_epoch=200):

    timeseries_dict = load_timeseries_dict_for_relative(csv_path=emission_path,warmup_period=warmup_period)

    # Should I infer this or just assume it's set?
    step_size = 0.1

    time_indexed_dict = get_time_indexed_dict(timeseries_dict,step_size)

    veh_ids = list(timeseries_dict.keys())

    measured_veh_ids = get_measured_veh_ids_at_GPS_rate(veh_ids)

    relative_vehicle_data_dict = get_relative_vehicle_data(timeseries_dict,time_indexed_dict,measured_veh_ids,step_size)

    #Organize into a list of timeseries data:
    timeseries_detection_data_list = []

    for veh_id in measured_veh_ids:
        veh_data = timeseries_dict[veh_id]
        relative_vehicle_data = relative_vehicle_data_dict[veh_id]

        speeds = []
        rel_headways = []
        rel_speeds = []

        for i in range(len(veh_data)):
            speeds.append(float(veh_data[i][1]))
            rel_headways.append(relative_vehicle_data[i][0])
            rel_speeds.append(relative_vehicle_data[i][1])

        accels = np.gradient(speeds,step_size)

        speeds = np.array(speeds)
        rel_headways = np.array(rel_headways)
        rel_speeds = np.array(rel_speeds)

    timeseries_detection_data_list.append([speeds,accels,rel_headways,rel_speeds])

    # organize into a training data set and train:
    train_X = make_train_X(timeseries_detection_data_list)

    model = get_cnn_lstm_ae_model(n_features=4)

    model_file_name = model_name+'_'+str(GPS_penetration_rate)+'percentGPS'

    print('Model: '+model_file_name)

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model










# RELATED TO CLASSIFICATION:


def k_means_classify(max_losses,cluster_diff=0.1):

    min_l = np.min(max_losses)
    max_l = np.max(max_losses)
    normalize_losses = (max_losses-min_l)/(max_l-min_l)
    X = normalize_losses
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X.reshape(-1,1))
    labels = kmeans.labels_
    cluster_centroids = kmeans.cluster_centers_

    positive_labels = []
    negative_labels = []

    for i in range(len(X)):
        l = X[i]
        label = labels[i]
        if(label==0):negative_labels.append(l)
        else:positive_labels.append(l) 

    if(np.min(positive_labels)-cluster_diff > np.max(negative_labels)):
        return labels,cluster_centroids
    else:
        return np.zeros_like(labels),cluster_centroids


def threshold_classification(max_losses,threshold):
    labels = np.zeros(len(max_losses))

    for i in range(len(max_losses)):
        if(max_losses[i] > threshold):
            labels[i] = 1

    return labels





