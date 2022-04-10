import math
import numpy as np

from Adversaries.controllers.base_controller import BaseController

class ACC_Benign(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = 0.5
        self.d_min = d_min
        self.V_m = V_m
        self.h = h
        self.a = 0.0

    def normal_ACC_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L

        self.a = self.accel_func(v, v_l, s)

    def accel_func(self,v,v_l,s):

        max_follow_dist = self.h*self.V_m

        if(s > max_follow_dist):
            # Switch to speed cotnrol if leader too far away, and max speed at V_m:
            u_des = self.Cruise_Control_accel(v)
            # u_des = np.min([0.0,self.ACC_accel(v,v_l,s)])
        else:
            u_des = self.ACC_accel(v,v_l,s)

        u_act = np.min([1.0,u_des])

        return u_act

    def Cruise_Control_accel(self,v):
        return self.k_3*(self.V_m - v)

    def ACC_accel(self,v,v_l,s):
        ex = s - v*self.h - self.d_min
        ev = v_l - v
        return self.k_1*ex+self.k_2*ev

    def get_accel(self, env):
        """See parent class."""
        self.normal_ACC_accel(env)
        return self.a #return the acceleration that is set above.
        
    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

class ACC_Switched_Controller_Attacked(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 SS_Threshold_min=60,
                 SS_Threshold_range=20,
                 want_multiple_attacks=False,
                 Total_Attack_Duration = 3.0,
                 attack_decel_rate = -.8,
                 display_attack_info = False,
                 warmup_steps = 1000,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = 0.5
        self.d_min = d_min
        self.V_m = V_m
        self.h = h
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.want_multiple_attacks = want_multiple_attacks
        self.initial_attack_occurred = False
        self.is_malicious = True

        self.SS_Threshold = SS_Threshold_min + np.random.rand()*SS_Threshold_range #number seconds at SS to initiate attack

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0 
        self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
        self.a = 0.0
        self.display_attack_info = display_attack_info
        self.warmup_steps = warmup_steps



    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:

        self.a = self.attack_decel_rate

        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id) 

        if(s < (v*(self.h-.2))):
            #If vehicle in front is getting too close, break from disturbance
            self.Reset_After_Attack(env)

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.initial_attack_occurred = True
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.Curr_Attack_Duration = 0.0
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        if(self.display_attack_info):
            print('Attack info:'+str(self.veh_id)+', '+str(self.attack_decel_rate)+', '+str(self.Total_Attack_Duration)+', '+str(env.step_counter*env.sim_step))

    def Check_For_Steady_State(self):
        self.numSteps_Steady_State += 1

    def normal_ACC_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L


        u = self.accel_func(v, v_l, s)

        self.a = u

    def accel_func(self,v,v_l,s):

        max_follow_dist = self.h*self.V_m

        if(s > max_follow_dist):
            # Switch to speed cotnrol if leader too far away, and max speed at V_m:
            u_des = self.Cruise_Control_accel(v)
            # u_des = np.min([0.0,self.ACC_accel(v,v_l,s)])
        else:
            u_des = self.ACC_accel(v,v_l,s)

        u_act = np.min([1.0,u_des])

        return u_act

    def Cruise_Control_accel(self,v):
        return self.k_3*(self.V_m - v)

    def ACC_accel(self,v,v_l,s):
        ex = s - v*self.h - self.d_min
        ev = v_l - v
        return self.k_1*ex+self.k_2*ev

    def Check_Start_Attack(self,env):

        step_size = env.sim_step
        SS_length = step_size * self.numSteps_Steady_State

        if(self.want_multiple_attacks):
            #If want multiple starts then let the time for attacking recycle:
            if(SS_length >= self.SS_Threshold):
                if(not self.isUnderAttack):
                    if(self.display_attack_info):
                        print('Beginning attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
                self.isUnderAttack = True
            else:
                self.isUnderAttack = False

        else:
            #If I don't want multiple attacks then only wait until initial wait period is up:
            if(not self.initial_attack_occurred):
                #Haven't attacked yet:
                if(SS_length >= self.SS_Threshold):
                    if(not self.isUnderAttack):
                        if(self.display_attack_info):
                            print('Beginning attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
                    self.isUnderAttack = True
                else:
                    self.isUnderAttack = False

            else:
                self.isUnderAttack = False

    def get_accel(self, env):
        """See parent class."""

        is_passed_warmup = env.step_counter > self.warmup_steps #Has the simulation progressed far enough

        perform_attack = self.isUnderAttack and is_passed_warmup #Should perform the attack if waited long enough and the random wait is over

        if(perform_attack):
            #Attack under way:
            self.Attack_accel(env) #Sets the vehicles acceleration, which is self.a
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:   
            # No attack currently happening:
            self.normal_ACC_accel(env) #Sets vehicles acceleration in self.a
            # Check to see if need to initiate attack:
            self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)


        return self.a #return the acceleration that is set above.

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

class ACC_Switched_Controller_Attacked_Single(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 distance_threshold_min = 900,
                 distance_threshold_max = 2600,
                 Total_Attack_Duration = 3.0,
                 attack_decel_rate = -.8,
                 display_attack_info = False,
                 warmup_steps = 1000,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None):
        """Instantiate a Switched Adaptive Cruise controller with Cruise Control."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.veh_id = veh_id
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = 0.1
        self.d_min = d_min
        self.V_m = V_m
        self.h = h

        # Hyper-params needed for tracking attack:
        self.isUnderAttack = False
        
        #Traverse a random distance before attacking, but not too close to edges of sim:
        self.distance_to_attack = distance_threshold_min + np.random.rand()*(distance_threshold_max-distance_threshold_min)

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0 
        self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
        self.a = 0.0
        self.display_attack_info = display_attack_info
        self.warmup_steps = warmup_steps
        self.attack_executed = False

        print('Attack ACC spawned : '+veh_id+' warmup steps: '+str(self.warmup_steps)+' attack distance: '+str(self.distance_to_attack))

    def Attack_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        u_ACC = self.accel_func(v, v_l, s) 
        # Select the minimum of the two, so that collisions are avoided
        if(self.attack_decel_rate<u_ACC):
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)

        commanded_accel = np.min([self.attack_decel_rate,u_ACC])

        self.a = commanded_accel
        

    def Reset_After_Attack(self,env):
        self.isUnderAttack = False
        self.Curr_Attack_Duration = 0.0
        self.attack_executed = True
        if(self.display_attack_info):
            print('Ended attack:'+str(self.veh_id)+', '+str(self.attack_decel_rate)+', '+str(self.Total_Attack_Duration)+', '+str(env.step_counter*env.sim_step))

    def normal_ACC_accel(self,env):
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        v = env.k.vehicle.get_speed(self.veh_id)
        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L

        self.a = self.accel_func(v, v_l, s)

    def accel_func(self,v,v_l,s):

        max_follow_dist = self.h*self.V_m

        if(s > max_follow_dist):
            # Switch to speed cotnrol if leader too far away, and max speed at V_m:
            u_des = self.Cruise_Control_accel(v)
            # u_des = np.min([0.0,self.ACC_accel(v,v_l,s)])
        else:
            u_des = self.ACC_accel(v,v_l,s)

        u_act = np.min([1.0,u_des])

        return u_act

    def Cruise_Control_accel(self,v):
        return self.k_3*(self.V_m - v)

    def ACC_accel(self,v,v_l,s):
        ex = s - v*self.h - self.d_min
        ev = v_l - v
        return self.k_1*ex+self.k_2*ev

    def Check_Start_Attack(self,env):

        is_passed_warmup = env.step_counter > self.warmup_steps
        #Traveled a long enough distance:
        is_passed_wait_period = env.k.vehicle.get_distance(self.veh_id) > self.distance_to_attack

        if(is_passed_wait_period and is_passed_warmup):
            self.isUnderAttack = True
            if(self.display_attack_info):
                print('Beginning attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
        else:
            self.isUnderAttack = False


    def get_accel(self, env):
        """See parent class."""

        #Should the vehicle execute an attack:
        performing_attack = self.isUnderAttack 

        if(performing_attack):
            #Attack under way:
            self.Attack_accel(env) #Sets the vehicles acceleration, which is self.a
            #Imcrement how long the attack has been happening for:
            self.Curr_Attack_Duration += env.sim_step
            #Attack is completed:
            if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
                #Attack has progressed for long enough and should be reset
                self.Reset_After_Attack(env)
                self.attack_executed = True

        else:
            #Attack hasn't been executed yet:
            if(not self.attack_executed):
                self.Check_Start_Attack(env)
                is_passed_warmup = env.step_counter > self.warmup_steps

            #Normal acceleration:
            self.normal_ACC_accel(env) 
            # Specificy that no attack is being executed:
            env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)

        return self.a #return the acceleration that is set above.

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

class FollowerStopper_Overreact(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 delay=0.0,
                 noise=0.0,
                 fail_safe=None,
                 v_des=10.0,
                 braking_period = 5.0,
                 braking_rate = -2.0):
        #Inherit the base controller:
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=delay,
            fail_safe=fail_safe,
            noise=noise)
        
        self.braking_period = braking_period #How long the mAV brakes for
        self.braking_rate = braking_rate #How hard the mAV brakes
        self.curr_braking_period = 0.0 #Keeps track of braking
        self.is_braking = False #Whether or not engaged in braking
        self.v_des = v_des #If not braking, what speed the mAV tries to drive at
                
    def get_accel(self, env):
        lead_id = env.k.vehicle.get_leader(self.veh_id) #Who is the leader
        v_l = env.k.vehicle.get_speed(lead_id) #Leader speed
        v = env.k.vehicle.get_speed(self.veh_id) #vehicle's own speed
        s = env.k.vehicle.get_headway(self.veh_id) #inter-vehicle spacing to leader
        u = 0.0
        
        #If the vehicle gets too close it brakes for a long period of time:
        
        timegap = s/(v+.01)
        

        if(timegap < 2.0 and not self.is_braking):
            print('Braking engaged, time '+str(env.sim_step*env.step_counter))
            self.is_braking = True

        #Engaged in braking:    
        if(self.is_braking):
            u = self.braking_rate
            self.curr_braking_period += env.sim_step

            if(self.curr_braking_period>=self.braking_period):
                self.curr_braking_period = 0.0
                self.is_braking = False
                
        #Managing speed:
        else:
            u = 0.1*(self.v_des - v) #Simple proportional speed control
            
        return u #return the acceleration that is set above.
        
    def get_custom_accel(self, v, v_l, s):
        """Leave as 0.0 since behavior has memory"""
        return 0.0

