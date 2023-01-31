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


class FollowerStopper_ACC_switch_on_time(BaseController):
    def __init__(self,veh_id,car_following_params,
        k_1=1.0,
        k_2=1.0,
        V_m=30,
        h=1.2,
        d_min=8.0,
        v_des=25.0,
        warmup_time=0.0,
        print_is_follower_stopper=True,
        time_delay=0.0,
        noise=0,
        fail_safe=['obey_speed_limit', 'safe_velocity', 'feasible_accel'],
        danger_edges=None,
        control_length=None,
        no_control_edges=None):

        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)

        self.warmup_time = warmup_time

        self.print_is_follower_stopper = print_is_follower_stopper


        self.check_initiilization_time = False

        self.is_Follower_Stopper = False

        """Instantiate FollowerStopper."""
        if fail_safe:
            BaseController.__init__(
                self, veh_id, car_following_params, delay=0.0,
                fail_safe=fail_safe)
        else:
            BaseController.__init__(
                self, veh_id, car_following_params, delay=0.0,
                fail_safe='safe_velocity')

        # follower-stopper parameters:
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        self.danger_edges = danger_edges if danger_edges else {}
        self.control_length = control_length
        self.no_control_edges = no_control_edges

        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = 0.5
        self.d_min = d_min
        self.V_m = V_m
        self.h = h

    def find_intersection_dist(self, env):
        """Find distance to intersection.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.k.vehicle.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.network.edge_length(edge_id)
        relative_pos = env.k.vehicle.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel_follower_stopper(self, env):
        """See parent class."""
        if env.time_counter < env.env_params.warmup_steps * env.env_params.sims_per_step:
            if self.default_controller:
                return self.default_controller.get_accel(env)
            else:
                return None
        else:
            lead_id = env.k.vehicle.get_leader(self.veh_id)
            this_vel = env.k.vehicle.get_speed(self.veh_id)
            lead_vel = env.k.vehicle.get_speed(lead_id)

            if self.v_des is None:
                return None

            if lead_id is None:
                v_cmd = self.v_des
            else:
                dx = env.k.vehicle.get_headway(self.veh_id)
                dv_minus = min(lead_vel - this_vel, 0)

                dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
                dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
                dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2
                v = min(max(lead_vel, 0), self.v_des)
                # compute the desired velocity
                if dx <= dx_1:
                    v_cmd = 0
                elif dx <= dx_2:
                    v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
                elif dx <= dx_3:
                    v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                            / (dx_3 - dx_2)
                else:
                    v_cmd = self.v_des

            edge = env.k.vehicle.get_edge(self.veh_id)

            if edge == "":
                return None

            if (self.find_intersection_dist(env) <= 10 and
                    env.k.vehicle.get_edge(self.veh_id) in self.danger_edges) or \
                    env.k.vehicle.get_edge(self.veh_id)[0] == ":":
                return None
            else:
                # compute the acceleration from the desired velocity
                return np.clip((v_cmd - this_vel) / env.sim_step, -np.abs(self.max_deaccel), self.max_accel)

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

    def get_accel_ACC(self, env):
        return self.normal_ACC_accel(env)


    def get_accel(self, env):
        # should 
        if(not self.check_initiilization_time):
            initialization_time = env.sim_step*env.step_counter
            self.is_Follower_Stopper = initialization_time >= self.warmup_time
            self.check_initiilization_time = True

            if(self.print_is_follower_stopper and self.is_Follower_Stopper):
                print(str(self.veh_id)+' initialized as a Follower Stopper at time '+str(initialization_time) + 'with warmup time '+str(self.warmup_time))

        if(self.is_Follower_Stopper):
            return self.get_accel_follower_stopper(env)
        else:
            return self.get_accel_ACC(env)


    def get_custom_accel(self,env):
        return 0


class FollowerStopper(BaseController):
    """Inspired by Dan Work's... work.

    Dissipation of stop-and-go waves via control of autonomous vehicles:
    Field experiments https://arxiv.org/abs/1705.01693

    Usage
    -----
    See base class for example.

    Parameters
    ----------
    veh_id : str
        unique vehicle identifier
    v_des : float, optional
        desired speed of the vehicles (m/s)
    no_control_edges : [str]
        list of edges that we should not apply control on
    """

    def __init__(self,
                 veh_id,
                 car_following_params,
                 v_des=15,
                 fail_safe=None,
                 danger_edges=None,
                 control_length=None,
                 no_control_edges=None):
        """Instantiate FollowerStopper."""
        if fail_safe:
            BaseController.__init__(
                self, veh_id, car_following_params, delay=0.0,
                fail_safe=fail_safe)
        else:
            BaseController.__init__(
                self, veh_id, car_following_params, delay=0.0,
                fail_safe='safe_velocity')

        # desired speed of the vehicle
        self.v_des = v_des

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # other parameters
        self.dx_1_0 = 4.5
        self.dx_2_0 = 5.25
        self.dx_3_0 = 6.0
        self.d_1 = 1.5
        self.d_2 = 1.0
        self.d_3 = 0.5

        self.danger_edges = danger_edges if danger_edges else {}
        self.control_length = control_length
        self.no_control_edges = no_control_edges

    def find_intersection_dist(self, env):
        """Find distance to intersection.

        Parameters
        ----------
        env : flow.envs.Env
            see flow/envs/base.py

        Returns
        -------
        float
            distance from the vehicle's current position to the position of the
            node it is heading toward.
        """
        edge_id = env.k.vehicle.get_edge(self.veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = env.k.network.edge_length(edge_id)
        relative_pos = env.k.vehicle.get_position(self.veh_id)
        dist = edge_len - relative_pos
        return dist

    def get_accel(self, env):
        """See parent class."""
        if env.time_counter < env.env_params.warmup_steps * env.env_params.sims_per_step:
            if self.default_controller:
                return self.default_controller.get_accel(env)
            else:
                return None
        else:
            lead_id = env.k.vehicle.get_leader(self.veh_id)
            this_vel = env.k.vehicle.get_speed(self.veh_id)
            lead_vel = env.k.vehicle.get_speed(lead_id)

            if self.v_des is None:
                return None

            if lead_id is None:
                v_cmd = self.v_des
            else:
                dx = env.k.vehicle.get_headway(self.veh_id)
                dv_minus = min(lead_vel - this_vel, 0)

                dx_1 = self.dx_1_0 + 1 / (2 * self.d_1) * dv_minus**2
                dx_2 = self.dx_2_0 + 1 / (2 * self.d_2) * dv_minus**2
                dx_3 = self.dx_3_0 + 1 / (2 * self.d_3) * dv_minus**2
                v = min(max(lead_vel, 0), self.v_des)
                # compute the desired velocity
                if dx <= dx_1:
                    v_cmd = 0
                elif dx <= dx_2:
                    v_cmd = v * (dx - dx_1) / (dx_2 - dx_1)
                elif dx <= dx_3:
                    v_cmd = v + (self.v_des - this_vel) * (dx - dx_2) \
                            / (dx_3 - dx_2)
                else:
                    v_cmd = self.v_des

            edge = env.k.vehicle.get_edge(self.veh_id)

            if edge == "":
                return None

            if (self.find_intersection_dist(env) <= 10 and
                    env.k.vehicle.get_edge(self.veh_id) in self.danger_edges) or \
                    env.k.vehicle.get_edge(self.veh_id)[0] == ":":
                return None
            else:
                # compute the acceleration from the desired velocity
                return np.clip((v_cmd - this_vel) / env.sim_step, -np.abs(self.max_deaccel), self.max_accel)

    def get_custom_accel(self, this_vel, lead_vel, h):
        """See parent class."""
        # raise NotImplementedError
        return 0

