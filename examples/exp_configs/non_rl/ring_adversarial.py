from flow.networks.ring import RingNetwork
from flow.core.params import VehicleParams
from flow.controllers.car_following_models import IDMController #Human driving model
from flow.controllers.routing_controllers import ContinuousRouter #Router that keeps vehicles on the ring-road

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

import numpy as np


# Custom for this sim:
from flow.controllers.lane_change_controllers import AILaneChangeController


from flow.controllers.base_controller import BaseController

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
        
    def get_custom_accel(self, this_vel, lead_vel, h):
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



sim_time = 1000
ring_length = 300
num_lanes = 2


#Make a number of unique drivers to all be put on road:

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
    
    lc_controller = (AILaneChangeController,{'left_delta':left_delta,
                                             'right_delta':right_delta,
                                             'left_beta':left_beta,
                                             'right_beta':right_beta,
                                             'switching_threshold':switching_threshold,
                                             'want_print_LC_info':True})
    
    driver_controller_list.append([label,cfm_controller,lc_controller,1])




#Simulation parameters:
time_step = 0.1 #In seconds, how far each step of the simulation goes.
emission_path = 'data' #Where csv is stored
want_render = True #If we want SUMO to render the environment and display the simulation.
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

