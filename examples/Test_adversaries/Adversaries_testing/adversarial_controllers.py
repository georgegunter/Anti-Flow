import numpy as npa
from copy import deepcopy
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

        # max_follow_dist = self.h*self.V_m

        # if(s > max_follow_dist):
        #   # Switch to speed cotnrol if leader too far away, and max speed at V_m:
        #   u_des = self.Cruise_Control_accel(v)
        #   # u_des = np.min([0.0,self.ACC_accel(v,v_l,s)])
        # else:
        #   u_des = self.ACC_accel(v,v_l,s)

        # u_act = np.min([1.0,u_des])


        u_ACC = self.ACC_accel(v,v_l,s)
        u_CC = u_des = self.Cruise_Control_accel(v)

        u_act = np.min([u_ACC,u_CC])

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

class ACC_comp_overwrite_Vm(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 V_m_comp, #maximum speed from compromise
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 SS_Threshold_min=60,
                 SS_Threshold_range=40,
                 want_multiple_attacks = True,
                 Total_Attack_Duration = 3.0,
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
        self.V_m_comp = V_m_comp


        self.V_m_current = V_m
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.want_multiple_attacks = want_multiple_attacks
        self.initial_attack_occurred = False
        self.is_malicious = True


        #Timing related things:
        self.SS_Threshold = SS_Threshold_min + np.random.rand()*SS_Threshold_range #number seconds at SS to initiate attack
        self.warmup_steps = warmup_steps

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0


        self.a = 0.0
        self.display_attack_info = display_attack_info


        if(self.display_attack_info):
            print('Spawning radar inject compromised ACC, attack frequency: '+str(self.SS_Threshold))
        if(self.want_multiple_attacks):
            print('Will engage in multiple attacks.')
        else:
            print('Will engage in a single attack.')


    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:
        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        ds_dt = v_l - v

        self.V_m_current = self.V_m_comp

        self.a = selfaccel_func(v,v_l,s)

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.initial_attack_occurred = True
        self.isUnderAttack = False
        self.Curr_Attack_Duration = 0.0
        self.numSteps_Steady_State = 0
        self.V_m_current = self.V_m
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        if(self.display_attack_info):
            print('Radar warp attack finished. veh_id: '+str(self.veh_id)+', slope: '+str(self.g)+', duration: '+str(self.Total_Attack_Duration)+', time: '+str(env.step_counter*env.sim_step))

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

        # u_act = np.min([1.0,u_des])

        return u_act

    def Cruise_Control_accel(self,v):
        return self.k_3*(self.V_m_current - v)

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
                        print('Beginning radar warp attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
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
                            print('Beginning radar warp attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
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
            
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:   
            # No attack currently happening:
            self.normal_ACC_accel(env) #Sets vehicles acceleration in self.a
            # Check to see if need to initiate attack:
            self.numSteps_Steady_State += 1
            if(env.step_counter >= self.warmup_steps):
                self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)


        self.a = np.max([self.a,-5.0])
        self.a = np.min([self.a,5.0])

        return self.a #return the acceleration that is set above.

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

class ACC_comp_inject_radar(BaseController):
    def __init__(self,
                 veh_id,
                 car_following_params,
                 g, #slope of attack
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 SS_Threshold_min=60,
                 SS_Threshold_range=40,
                 want_multiple_attacks = True,
                 Total_Attack_Duration = 3.0,
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
        self.g = g
        self.isUnderAttack = False
        self.numSteps_Steady_State = 0
        self.want_multiple_attacks = want_multiple_attacks
        self.initial_attack_occurred = False
        self.is_malicious = True


        #Timing related things:
        self.SS_Threshold = SS_Threshold_min + np.random.rand()*SS_Threshold_range #number seconds at SS to initiate attack
        self.warmup_steps = warmup_steps

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0


        self.a = 0.0
        self.display_attack_info = display_attack_info


        if(self.display_attack_info):
            print('Spawning radar inject compromised ACC, attack frequency: '+str(self.SS_Threshold))
        if(self.want_multiple_attacks):
            print('Will engage in multiple attacks.')
        else:
            print('Will engage in a single attack.')


    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:
        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)
        ds_dt = v_l - v

        s_attacked = self.g*s

        ds_dt_attacked = self.g * ds_dt

        v_l_attacked = v + ds_dt_attacked


        a_attacked = self.accel_func(v,v_l_attacked,s_attacked)

        self.a = a_attacked

        # a_nom = self.accel_func(v,v_l,s)

        # # If need to brake harder to avoid a collision:
        # self.a = np.min([self.attack_decel_rate,a_nom])

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.initial_attack_occurred = True
        self.isUnderAttack = False
        self.Curr_Attack_Duration = 0.0
        self.numSteps_Steady_State = 0
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        if(self.display_attack_info):
            print('Radar warp attack finished. veh_id: '+str(self.veh_id)+', slope: '+str(self.g)+', duration: '+str(self.Total_Attack_Duration)+', time: '+str(env.step_counter*env.sim_step))

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
                        print('Beginning radar warp attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
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
                            print('Beginning radar warp attack: '+self.veh_id+' Time: '+str(env.step_counter*env.sim_step))
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
            
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:   
            # No attack currently happening:
            self.normal_ACC_accel(env) #Sets vehicles acceleration in self.a
            # Check to see if need to initiate attack:
            self.numSteps_Steady_State += 1
            if(env.step_counter >= self.warmup_steps):
                self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)


        self.a = np.max([self.a,-5.0])
        self.a = np.min([self.a,5.0])

        return self.a #return the acceleration that is set above.

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

class ACC_comp_RDA(BaseController):

    def __init__(self,
                 veh_id,
                 car_following_params,
                 k_1=1.0,
                 k_2=1.0,
                 V_m=30,
                 h=1.2,
                 d_min=8.0,
                 SS_Threshold_min=60,
                 SS_Threshold_range=40,
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


        #Timing related things:
        self.SS_Threshold = SS_Threshold_min + np.random.rand()*SS_Threshold_range #number seconds at SS to initiate attack
        self.warmup_steps = warmup_steps

        self.Total_Attack_Duration = Total_Attack_Duration #How long attack lasts for
        self.Curr_Attack_Duration = 0.0 
        self.attack_decel_rate = attack_decel_rate #Rate at which ACC decelerates
        self.a = 0.0
        self.display_attack_info = display_attack_info


        if(self.display_attack_info):
            print('Spawning compromised ACC, attack frequency: '+str(self.SS_Threshold))
        if(self.want_multiple_attacks):
            print('Will engage in multiple attacks.')
        else:
            print('Will engage in a single attack.')


    def Attack_accel(self,env):
        #Declerates the car for a set period at a set rate:
        self.Curr_Attack_Duration += env.sim_step

        s = env.k.vehicle.get_headway(self.veh_id)
        L = env.k.vehicle.get_length(self.veh_id)
        s = s - L
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        v_l = env.k.vehicle.get_speed(lead_id)

        a_nom = self.accel_func(v,v_l,s)

        # If need to brake harder to avoid a collision:
        self.a = np.min([self.attack_decel_rate,a_nom])

        if(self.Curr_Attack_Duration >= self.Total_Attack_Duration):
            self.Reset_After_Attack(env)

    def Reset_After_Attack(self,env):
        self.initial_attack_occurred = True
        self.isUnderAttack = False
        self.Curr_Attack_Duration = 0.0
        self.numSteps_Steady_State = 0
        pos  = env.k.vehicle.get_position(self.veh_id)
        lane = env.k.vehicle.get_lane(self.veh_id)
        if(self.display_attack_info):
            print('Attack finished:'+str(self.veh_id)+', '+str(self.attack_decel_rate)+', '+str(self.Total_Attack_Duration)+', '+str(env.step_counter*env.sim_step))

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
            
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=1)
        else:   
            # No attack currently happening:
            self.normal_ACC_accel(env) #Sets vehicles acceleration in self.a
            # Check to see if need to initiate attack:
            self.numSteps_Steady_State += 1
            if(env.step_counter >= self.warmup_steps):
                self.Check_Start_Attack(env)
            # Specificy that no attack is being executed:
            #Commenting out temporarily, since needs to be re-implemented:
            # env.k.vehicle.set_malicious(veh_id=self.veh_id,is_malicious=0)


        return self.a #return the acceleration that is set above.

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # Not implemented...
        return self.a

    