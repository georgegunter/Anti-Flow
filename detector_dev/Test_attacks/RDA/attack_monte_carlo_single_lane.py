import numpy as np
import os
import shutil
import time
from copy import deepcopy
import ray
from tqdm import tqdm




import flow
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




from Adversaries.controllers.base_controller import BaseController
from Adversaries.controllers.car_following_adversarial import ACC_Benign
from Adversaries.controllers.car_following_adversarial import ACC_comp_overwrite_Vm
from Adversaries.controllers.car_following_adversarial import ACC_comp_inject_radar
from Adversaries.controllers.car_following_adversarial import ACC_comp_RDA











# For simulation:
def run_ring_sim_variable_cfm(ring_length=600,
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



def make_benign_driver_list(num_human_drivers=35,num_ACC_drivers = 3):
	driver_controller_list = []

	#cfm parameters:
	a_mean=0.666
	b_mean=21.6
	s0_mean=2.21
	s1_mean=2.82
	Vm_mean=10.0

	for i in range(num_human_drivers):
		a = a_mean*(1+np.random.normal(0,0.1))
		b = b_mean*(1+np.random.normal(0,0.1))
		s0 = s0_mean*(1+np.random.normal(0,0.1))
		s1 = s1_mean*(1+np.random.normal(0,0.1))
		Vm = Vm_mean*(1+np.random.normal(0,0.02))

		label = 'bando_ftl_ovm_a'+str(np.round(a,2))+'_b'+str(np.round(b,2))+'_s0'+str(np.round(s0,2))+'_s1'+str(np.round(s1,2))+'_Vm'+str(np.round(Vm,2))
		cfm_controller = (Bando_OVM_FTL,{'a':a,'b':b,'s0':s0,'s1':s1,'Vm':Vm,'noise':0.1})

		driver_controller_list.append([label,cfm_controller,1])

	k_1_mean = 1.5
	k_2_mean = 0.2
	h_mean = 1.8
	V_m_mean = 15.0
	d_min_mean = 10.0

	for i in range(num_ACC_drivers):
		k_1 = k_1_mean*(1+np.random.normal(0,0.1))
		k_2 = k_2_mean*(1+np.random.normal(0,0.1))
		h = h_mean*(1+np.random.normal(0,0.1))
		V_m = V_m_mean*(1+np.random.normal(0,0.1))
		d_min = d_min_mean

		label = 'ACC_k_1'+str(np.round(k_1,2))+'_k_2'+str(np.round(k_2,2))+'_h'+str(np.round(h,2))+'_V_m'+str(np.round(V_m,2))+'d_m'+str(np.round(d_min,2))
		cfm_controller = (ACC_Benign,{'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min})
		driver_controller_list.append([label,cfm_controller,1]) 

	return driver_controller_list



def make_mal_driver_list(Attack_duration,Attack_frequency,attack_decel_rate,num_comped_ACCs=2,num_human_drivers=35,num_ACC_drivers = 3):

	driver_controller_list = make_benign_driver_list(num_human_drivers,num_ACC_drivers)


	for i in range(num_comped_ACCs):

		k_1_mean = 1.5
		k_2_mean = 0.2
		h_mean = 1.8
		V_m_mean = 15.0
		d_min_mean = 10.0

		k_1 = k_1_mean*(1+np.random.normal(0,0.1))
		k_2 = k_2_mean*(1+np.random.normal(0,0.1))
		h = h_mean*(1+np.random.normal(0,0.1))
		V_m = V_m_mean*(1+np.random.normal(0,0.1))
		d_min = d_min_mean

		want_multiple_attacks=True

		warmup_steps = 500 + np.random.rand()*20
		SS_Threshold_min = Attack_frequency

		display_attack_info = True

		adversary = (ACC_comp_RDA, {'k_1':k_1,'k_2':k_2,'h':h,'V_m':V_m,'d_min':d_min,
														'want_multiple_attacks':want_multiple_attacks,
														'Total_Attack_Duration':Attack_duration,
														'attack_decel_rate':attack_decel_rate,
														'warmup_steps':warmup_steps,
														'SS_Threshold_min':Attack_frequency,
														'SS_Threshold_range':0,
														'display_attack_info':display_attack_info})
		
		ACC_label = '_k1_'+str(np.round(k_1,2))+'_k2_'+str(np.round(k_2,2))+'_h_'+str(np.round(h,2))+'_Vm_'+str(np.round(V_m,2))+'_dm_'+str(np.round(d_min,2))

		label_adv = 'RDA_adv_DA_'+str(np.round(Attack_duration,2))+'_AF_'+str(np.round(Attack_frequency,2))+'_ADR_'+str(np.round(attack_decel_rate,2))

		label_adv = label_adv + ACC_label

		driver_controller_list.append([label_adv,adversary,1])

	return driver_controller_list



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


def get_file_name_no_version(attack_params):
	Attack_duration = attack_params[0]
	Attack_frequency = attack_params[1]
	attack_decel_rate = attack_params[2]
	
	file_name_no_version = 'RDA_AD_'+str(Attack_duration)+'_AF_'+str(Attack_frequency)+'_DR_'+str(attack_decel_rate)

	return file_name_no_version





def run_sim_with_attack(attack_params,emission_path,ring_length=600):

	Attack_duration = attack_params[0]
	Attack_frequency = attack_params[1]
	attack_decel_rate = attack_params[2]
	
	driver_controller_list_with_attack = make_mal_driver_list(Attack_duration,Attack_frequency,attack_decel_rate)
	
	sim_res_list_with_attack = run_ring_sim_variable_cfm(driver_controller_list = driver_controller_list_with_attack,
													 ring_length=ring_length,
													 sim_time=400,
													 emission_path=emission_path)
	
	file_path = os.path.join(os.getcwd(),sim_res_list_with_attack[1])
	

	file_name_no_version = get_file_name_no_version(attack_params)


	file_path_new = rename_file(file_path,file_name_no_version,emission_path)


	return file_path_new


@ray.remote
def run_sim_with_attack_ray(attack_params,emission_path,ring_length=600):
	return run_sim_with_attack(attack_params,emission_path,ring_length=600)



def get_number_run_sims(emission_path,attack_params):

	file_name_no_version = get_file_name_no_version(attack_params)

	existing_files = os.listdir(emission_path)

	existing_file_versions = 0

	for file in existing_files:
		if(file_name_no_version in file):
			existing_file_versions += 1

	return existing_file_versions


def run_sim_list_ray(attack_params_list,emission_path,ring_length=600):

	sim_info_ids = []

	for attack_params in tqdm(attack_params_list):

		sim_info_ids.append(
			run_sim_with_attack_ray.remote(attack_params,emission_path,ring_length=600)
			)

	file_path_list = ray.get(sim_info_ids)

	return file_path_list


if __name__ == '__main__':
	
	emission_path = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_attack_monte_carlo'


	want_test_sim = True

	if(want_test_sim):

		attack_params = [np.random.uniform()*20,np.random.uniform()*60,np.random.uniform()*(-2)]

		run_sim_with_attack(attack_params,emission_path)


	want_run_all_sims = False

	if(want_run_all_sims):
		num_samples = 200

		ray.init(num_cpus=3)

		attack_params_list = np.zeros([num_samples,3])

		for i in range(num_samples):
			attack_params_list[i,:] = np.array([np.random.uniform()*20,np.random.uniform()*60,np.random.uniform()*(-2)])


		run_sim_list_ray(attack_params_list,emission_path)

		print('Finished with simulations.')






