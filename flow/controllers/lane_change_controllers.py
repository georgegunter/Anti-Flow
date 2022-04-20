"""Contains a list of custom lane change controllers."""

from flow.controllers.base_lane_changing_controller import \
    BaseLaneChangeController


class SimLaneChangeController(BaseLaneChangeController):
    """A controller used to enforce sumo lane-change dynamics on a vehicle.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return None


class StaticLaneChanger(BaseLaneChangeController):
    """A lane-changing model used to keep a vehicle in the same lane.

    Usage: See base class for usage example.
    """

    def get_lane_change_action(self, env):
        """See parent class."""
        return 0


class AILaneChangeController(BaseLaneChangeController):
    """A lane-changing controller based on acceleration incentive model.

    Usage
    -----
    See base class for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO/Aimsun identification
    lane_change_params : flow.core.param.SumoLaneChangeParams
        see parent class
    left_delta : float
        used for the incentive criterion for left lane change (default: .5)
    right_delta : float
        used for the incentive criterion for right lane change (default: .5)
    left_beta : float
        used for the incentive criterion for left lane change (default: 1.5)
    right_beta : float
        used for the incentive criterion for right lane change (default: 1.5)
    """

    def __init__(self,
                 veh_id,
                 lane_change_params=None,
                 left_delta=0.5,
                 right_delta=0.5,
                 left_beta=1.5,
                 right_beta=1.5,
                 switching_threshold = 5.0,
                 want_print_LC_info=False):
        """Instantiate an AI lane-change controller."""
        BaseLaneChangeController.__init__(
            self,
            veh_id,
            lane_change_params,
        )

        self.veh_id = veh_id
        self.left_delta = left_delta
        self.right_delta = right_delta
        self.left_beta = left_beta
        self.right_beta = right_beta
        self.want_print_LC_info = want_print_LC_info # Will print info about LC when happens
        self.switching_threshold = switching_threshold
        self.cooldown_period = switching_threshold


    def get_lane_change_action(self, env):
        """See parent class."""
        # Increment how long since last lane-change:
        self.cooldown_period += env.sim_step

        if(self.cooldown_period >= self.switching_threshold):
            # acceleration if the ego vehicle remains in current lane.
            ego_accel_controller = env.k.vehicle.get_acc_controller(self.veh_id)
            acc_in_present_lane = ego_accel_controller.get_accel(env)

            # get ego vehicle lane number, and velocity
            ego_lane = env.k.vehicle.get_lane(self.veh_id)
            ego_vel = env.k.vehicle.get_speed(self.veh_id)

            # get lane leaders, followers, headways, and tailways
            lane_leaders = env.k.vehicle.get_lane_leaders(self.veh_id)
            lane_followers = env.k.vehicle.get_lane_followers(self.veh_id)
            lane_headways = env.k.vehicle.get_lane_headways(self.veh_id)
            lane_tailways = env.k.vehicle.get_lane_tailways(self.veh_id)

            # determine left and right lane number
            this_edge = env.k.vehicle.get_edge(self.veh_id)
            num_lanes = env.k.network.num_lanes(this_edge)

            r_lane = ego_lane - 1 if ego_lane > 0 else None
            l_lane = ego_lane + 1 if ego_lane < num_lanes-1 else None

            # if(self.want_print_LC_info):
            #     print('Vehicle: '+self.veh_id+' Current Lane: '+str(ego_lane)+' Left  Lane: '+str(l_lane)+' Right  Lane: '+str(r_lane))

            # compute ego and new follower accelerations if moving to left lane
            if l_lane is not None:
                # get left leader and follower vehicle ID
                l_l = lane_leaders[l_lane]
                l_f = lane_followers[l_lane]

                # ego acceleration if the ego vehicle is in the lane to the left
                if l_l not in ['', None]:
                    # left leader velocity and headway
                    l_l_vel = env.k.vehicle.get_speed(l_l)
                    l_l_headway = lane_headways[l_lane]
                    acc_in_left_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=l_l_vel,
                        h=l_l_headway)
                else:  # if left lane exists but left leader does not exist
                    # in this case we assign None to the leader velocity and
                    # large number to headway
                    l_l_vel = None
                    l_l_headway = 1000
                    acc_in_left_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=l_l_vel,
                        h=l_l_headway)

                # follower acceleration if the ego vehicle is in the left lane
                if l_f not in ['', None]:
                    # left follower velocity and headway
                    l_f_vel = env.k.vehicle.get_speed(l_f)
                    l_f_tailway = lane_tailways[l_lane]
                    l_f_accel_controller = env.k.vehicle.get_acc_controller(l_f)
                    left_lane_follower_acc = l_f_accel_controller.get_custom_accel(
                        this_vel=l_f_vel,
                        lead_vel=ego_vel,
                        h=l_f_tailway)
                else:  # if left lane exists but left follower does not exist
                    # in this case we assign maximum acceleration
                    left_lane_follower_acc = ego_accel_controller.max_accel
            else:
                acc_in_left_lane = None
                left_lane_follower_acc = None

            # compute ego and new follower accelerations if moving to right lane
            if r_lane is not None:
                # get right leader and follower vehicle ID
                r_l = lane_leaders[r_lane]
                r_f = lane_followers[r_lane]

                # ego acceleration if the ego vehicle is in the lane to the right
                if r_l not in ['', None]:
                    # right leader velocity and headway
                    r_l_vel = env.k.vehicle.get_speed(r_l)
                    r_l_headway = lane_headways[r_lane]
                    acc_in_right_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=r_l_vel,
                        h=r_l_headway)
                else:  # if right lane exists but right leader does not exist
                    # in this case we assign None to the leader velocity and
                    # large number to headway
                    r_l_vel = None
                    r_l_headway = 1000
                    acc_in_right_lane = ego_accel_controller.get_custom_accel(
                        this_vel=ego_vel,
                        lead_vel=r_l_vel,
                        h=r_l_headway)

                # follower acceleration if the ego vehicle is in the right lane
                if r_f not in ['', None]:
                    # right follower velocity and headway
                    r_f_vel = env.k.vehicle.get_speed(r_f)
                    r_f_headway = lane_tailways[r_lane]
                    r_f_accel_controller = env.k.vehicle.get_acc_controller(r_f)
                    right_lane_follower_acc = r_f_accel_controller.get_custom_accel(
                        this_vel=r_f_vel,
                        lead_vel=ego_vel,
                        h=r_f_headway)
                else:  # if right lane exists but right follower does not exist
                    # assign maximum acceleration
                    right_lane_follower_acc = ego_accel_controller.max_accel
            else:
                acc_in_right_lane = None
                right_lane_follower_acc = None

            #Calculate safety and insentives for LC:
            if(acc_in_left_lane is None):# no left lane
                left_safety = False
                left_insentive = False
            elif(left_lane_follower_acc is None):# no followers in left lane
                left_safety = (acc_in_left_lane >= -self.left_beta)
                left_insentive = (acc_in_left_lane >= acc_in_present_lane + self.left_delta)
            else:# left lane with follower
                left_safety = (acc_in_left_lane >= -self.left_beta) and \
                    (left_lane_follower_acc >= -self.left_beta)
                left_insentive = (acc_in_left_lane >= acc_in_present_lane + self.left_delta)
            
            if(acc_in_right_lane is None):# no right lane
                right_safety = False
                right_insentive = False
            elif(right_lane_follower_acc is None):# no followers in right lane
                right_safety = (acc_in_right_lane >= -self.right_beta)
                right_insentive = (acc_in_right_lane >= acc_in_present_lane + self.right_delta)
            else:# right lane with follower
                right_safety = (acc_in_right_lane >= -self.right_beta) and \
                    (right_lane_follower_acc >= -self.right_beta)
                right_insentive = (acc_in_right_lane >= acc_in_present_lane + self.right_delta)



            if l_lane is not None and left_safety and left_insentive:
                #Left lane is safe and want to merge:
                action = 1
                self.cooldown_period = 0.0
                if(self.want_print_LC_info):
                    print('Vehicle: '+str(self.veh_id)+' merged left from lane '+str(ego_lane)+' into lane '+str(l_lane))

            elif r_lane is not None and right_safety and right_insentive:
                #Right lane is safe and want to merge:
                action = -1
                self.cooldown_period = 0.0
                if(self.want_print_LC_info):
                    print('Vehicle: '+str(self.veh_id)+' merged right from lane '+str(ego_lane)+' into lane '+str(r_lane))
            else:
                #Don't want any action:
                action = 0
        else: action = 0
        return action
