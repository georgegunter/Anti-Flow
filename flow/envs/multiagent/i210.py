"""Environment for training vehicles to reduce congestion in the I210."""

from gym.spaces import Box
import numpy as np

from flow.core.rewards import instantaneous_mpg
from flow.envs.multiagent.base import MultiEnv

# largest number of lanes on any given edge in the network
MAX_LANES = 6
SPEED_SCALE = 50
HEADWAY_SCALE = 1000

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 1,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 1,
    # whether we use an obs space that contains adjacent lane info or just the
    # lead obs
    "lead_obs": True,
    # whether the reward should come from local vehicles instead of global
    # rewards
    "local_reward": True,
    # desired velocity
    "target_velocity": 25
}


class I210MultiEnv(MultiEnv):
    """Partially observable multi-agent environment for the I-210 subnetworks.

    The policy is shared among the agents, so there can be a non-constant
    number of RL vehicles throughout the simulation.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2

    The following states, actions and rewards are considered for one autonomous
    vehicle only, as they will be computed in the same way for each of them.

    States
        The observation consists of the speeds and bumper-to-bumper headways of
        the vehicles immediately preceding and following autonomous vehicles in
        all of the preceding lanes as well, a binary value indicating which of
        these vehicles is autonomous, and the speed of the autonomous vehicle.
        Missing vehicles are padded with zeros.

    Actions
        The action consists of an acceleration, bound according to the
        environment parameters, as well as three values that will be converted
        into probabilities via softmax to decide of a lane change (left, none
        or right). NOTE: lane changing is currently not enabled. It's a TODO.

    Rewards
        The reward function encourages proximity of the system-level velocity
        to a desired velocity specified in the environment parameters, while
        slightly penalizing small time headways among autonomous vehicles.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)
        self.lead_obs = env_params.additional_params.get("lead_obs")
        self.reroute_on_exit = env_params.additional_params.get("reroute_on_exit")
        self.max_lanes = MAX_LANES
        self.num_enter_lanes = 5
        self.entrance_edge = "ghost0"
        self.exit_edge = "119257908#3"
        self.control_range = env_params.additional_params.get('control_range', None)
        self.no_control_edges = env_params.additional_params.get('no_control_edges', [])
        self.mpg_reward = env_params.additional_params["mpg_reward"]
        self.look_back_length = env_params.additional_params["look_back_length"]

        # dynamics controller for uncontrolled RL vehicles (mimics humans)
        controller = self.k.vehicle.type_parameters["human"][
            "acceleration_controller"]
        self._human_controller = controller[0](
            veh_id="av",
            car_following_params=self.k.vehicle.type_parameters["human"][
                "car_following_params"],
            **controller[1]
        )

        # list of all RL vehicles (even out of network) after reroute_on_exit starts
        self.reroute_rl_ids = set()

        # whether to add a slight reward for opening up a gap that will be annealed out N iterations in
        self.headway_curriculum = env_params.additional_params["headway_curriculum"]
        # how many timesteps to anneal the headway curriculum over
        self.headway_curriculum_iters = env_params.additional_params["headway_curriculum_iters"]
        self.headway_reward_gain = env_params.additional_params["headway_reward_gain"]
        self.min_time_headway = env_params.additional_params["min_time_headway"]

        # whether to add a slight reward for opening up a gap that will be annealed out N iterations in
        self.speed_curriculum = env_params.additional_params["speed_curriculum"]
        # how many timesteps to anneal the headway curriculum over
        self.speed_curriculum_iters = env_params.additional_params["speed_curriculum_iters"]
        self.speed_reward_gain = env_params.additional_params["speed_reward_gain"]
        self.leader = []

        # penalize stops
        self.penalize_stops = env_params.additional_params["penalize_stops"]
        self.stop_penalty = env_params.additional_params["stop_penalty"]

        # penalize accel
        self.penalize_accel = env_params.additional_params.get("penalize_accel", False)
        self.accel_penalty = env_params.additional_params["accel_penalty"]

    @property
    def observation_space(self):
        """See class definition."""
        # speed, speed of leader, headway
        if self.lead_obs:
            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(3,),
                dtype=np.float32
            )
        # speed, dist to ego vehicle, binary value which is 1 if the vehicle is
        # an AV
        else:
            leading_obs = 3 * self.max_lanes
            follow_obs = 3 * self.max_lanes

            # speed and lane
            self_obs = 2

            return Box(
                low=-float('inf'),
                high=float('inf'),
                shape=(leading_obs + follow_obs + self_obs,),
                dtype=np.float32
            )

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-np.abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1,),  # (4,),
            dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            rl_ids = []
            accels = []
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                controller = self.k.vehicle.get_acc_controller(rl_id)
                accel = controller.compute_failsafe(accel, self)
                accels.append(accel)
                rl_ids.append(rl_id)

            # prevent the AV from blocking the entrance
            self.k.vehicle.apply_acceleration(rl_ids, accels)

    def in_control_range(self, veh_id):
        """Return if a veh_id is on an edge that is allowed to be controlled.

        If control range is defined it uses control range, otherwise it searches over a set of edges
        """
        return (self.control_range and self.control_range[1] >
                self.k.vehicle.get_x_by_id(veh_id) > self.control_range[0]) or \
               (len(self.no_control_edges) > 0 and self.k.vehicle.get_edge(veh_id) not in
                self.no_control_edges)

    def get_state(self):
        """See class definition."""
        valid_ids = [rl_id for rl_id in self.k.vehicle.get_rl_ids() if self.in_control_range(rl_id)]
        if self.lead_obs:
            veh_info = {}
            for rl_id in valid_ids:
                speed = self.k.vehicle.get_speed(rl_id)
                lead_id = self.k.vehicle.get_leader(rl_id)
                if lead_id in ["", None]:
                    # in case leader is not visible
                    lead_speed = SPEED_SCALE
                    headway = HEADWAY_SCALE
                else:
                    lead_speed = self.k.vehicle.get_speed(lead_id)
                    headway = self.k.vehicle.get_headway(rl_id)
                veh_info.update({rl_id: np.array([speed / SPEED_SCALE, headway / HEADWAY_SCALE,
                                                  lead_speed / SPEED_SCALE])})
        else:
            veh_info = {rl_id: np.concatenate((self.state_util(rl_id),
                                               self.veh_statistics(rl_id)))
                        for rl_id in valid_ids}
        return veh_info

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # in the warmup steps
        if rl_actions is None:
            return {}

        rewards = {}
        valid_ids = [rl_id for rl_id in self.k.vehicle.get_rl_ids() if self.in_control_range(rl_id)]
        valid_human_ids = [veh_id for veh_id in self.k.vehicle.get_ids() if self.in_control_range(veh_id)]

        if self.env_params.additional_params["local_reward"]:
            des_speed = self.env_params.additional_params["target_velocity"]
            for rl_id in valid_ids:
                rewards[rl_id] = 0
                if self.mpg_reward:
                    rewards[rl_id] = instantaneous_mpg(self, rl_id, gain=1.0) / 100.0
                    follow_id = rl_id
                    for i in range(self.look_back_length):
                        follow_id = self.k.vehicle.get_follower(follow_id)
                        if follow_id not in ["", None]:
                            rewards[rl_id] += instantaneous_mpg(self, follow_id, gain=1.0) / 100.0
                        else:
                            break
                else:
                    follow_id = rl_id
                    for i in range(self.look_back_length + 1):
                        if follow_id not in ["", None]:
                            follow_speed = self.k.vehicle.get_speed(self.k.vehicle.get_follower(follow_id))
                            reward = (des_speed - min(np.abs(follow_speed - des_speed), des_speed)) ** 2
                            reward /= ((des_speed ** 2) * self.look_back_length)
                            rewards[rl_id] += reward
                        else:
                            break
                        follow_id = self.k.vehicle.get_follower(follow_id)

        else:
            if self.mpg_reward:
                reward = np.nan_to_num(instantaneous_mpg(self, valid_human_ids, gain=1.0)) / 100.0
            else:
                speeds = self.k.vehicle.get_speed(valid_human_ids)
                des_speed = self.env_params.additional_params["target_velocity"]
                # rescale so the critic can estimate it quickly
                if self.reroute_on_exit:
                    reward = np.nan_to_num(np.mean([(des_speed - np.abs(speed - des_speed))
                                                    for speed in speeds]) / des_speed)
                else:
                    reward = np.nan_to_num(np.mean([(des_speed - np.abs(speed - des_speed)) ** 2
                                                    for speed in speeds]) / (des_speed ** 2))
            rewards = {rl_id: reward for rl_id in valid_ids}

        # curriculum over time-gaps
        if self.headway_curriculum and self._num_training_iters <= self.headway_curriculum_iters:
            t_min = self.min_time_headway  # smallest acceptable time headway
            for veh_id, rew in rewards.items():
                lead_id = self.k.vehicle.get_leader(veh_id)
                penalty = 0
                if lead_id not in ["", None] \
                        and self.k.vehicle.get_speed(veh_id) > 0:
                    t_headway = max(
                        self.k.vehicle.get_headway(veh_id) /
                        self.k.vehicle.get_speed(veh_id), 0)
                    scaling_factor = max(0, 1 - self._num_training_iters / self.headway_curriculum_iters)
                    penalty += scaling_factor * self.headway_reward_gain * min((t_headway - t_min) / t_min, 0)

                rewards[veh_id] += penalty

        if self.speed_curriculum and self._num_training_iters <= self.speed_curriculum_iters:
            des_speed = self.env_params.additional_params["target_velocity"]

            for veh_id, rew in rewards.items():
                speed = self.k.vehicle.get_speed(veh_id)
                speed_reward = 0.0
                follow_id = veh_id
                for i in range(self.look_back_length):
                    follow_id = self.k.vehicle.get_follower(follow_id)
                    if follow_id not in ["", None]:
                        if self.reroute_on_exit:
                            speed_reward += (des_speed - np.abs(speed - des_speed)) / des_speed
                        else:
                            speed_reward += ((des_speed - np.abs(speed - des_speed)) ** 2) / (des_speed ** 2)
                    else:
                        break
                scaling_factor = max(0, 1 - self._num_training_iters / self.speed_curriculum_iters)

                rewards[veh_id] += speed_reward * scaling_factor * self.speed_reward_gain

        for veh_id in rewards.keys():
            speed = self.k.vehicle.get_speed(veh_id)
            if self.penalize_stops:
                if speed < 1.0:
                    rewards[veh_id] -= self.stop_penalty
            if self.penalize_accel and veh_id in self.k.vehicle.previous_speeds:
                prev_speed = self.k.vehicle.get_previous_speed(veh_id)
                abs_accel = abs(speed - prev_speed) / self.sim_step
                rewards[veh_id] -= abs_accel * self.accel_penalty

        # print('time to get reward is ', time() - t)
        return rewards

    def additional_command(self):
        """See parent class.

        Define which vehicles are observed for visualization purposes. Additionally, optionally reroute vehicles
        back once they have exited.
        """
        # In the warmup period all vehicles act as humans.
        if self.time_counter < \
                self.env_params.warmup_steps * self.env_params.sims_per_step:
            uncontrolled_veh_ids = [veh_id for veh_id in self.k.vehicle.get_ids() if
                                    "human" not in self.k.vehicle.get_type(veh_id)]

        # If no control range is specified, all vehicles are controlled.
        elif len(self.no_control_edges) == 0:
            uncontrolled_veh_ids = []

        # Vehicles in the control_range are controlled, others act as humans.
        else:
            uncontrolled_veh_ids = []
            for veh_id in self.k.vehicle.get_ids():
                if not self.in_control_range(veh_id) and "human" not in self.k.vehicle.get_type(veh_id):
                    uncontrolled_veh_ids.append(veh_id)

        # Assign accelerations to uncontrolled vehicles.
        for veh_id in uncontrolled_veh_ids:
            self._human_controller.veh_id = veh_id
            acceleration = self._human_controller.get_action(self)
            self.k.vehicle.apply_acceleration(veh_id, acceleration)

        super().additional_command()
        # specify observed vehicles
        for rl_id in self.k.vehicle.get_rl_ids():
            # leader
            lead_id = self.k.vehicle.get_leader(rl_id)
            if lead_id:
                self.k.vehicle.set_observed(lead_id)

        if self.reroute_on_exit and self.time_counter >= self.env_params.sims_per_step * self.env_params.warmup_steps \
                and not self.env_params.evaluate:
            veh_ids = list(self.k.vehicle.get_ids())
            edges = self.k.vehicle.get_edge(veh_ids)
            valid_lanes = list(range(self.num_enter_lanes))
            for veh_id, edge in zip(veh_ids, edges):
                if edge == "":
                    continue
                if edge[0] == ":":  # center edge
                    continue
                # on the exit edge, near the end, and is the vehicle furthest along
                if edge == self.exit_edge and \
                        (self.k.vehicle.get_position(veh_id) > self.k.network.edge_length(self.exit_edge) - 100) \
                        and self.k.vehicle.get_leader(veh_id) is None:
                    type_id = self.k.vehicle.get_type(veh_id)
                    # remove the vehicle
                    self.k.vehicle.remove(veh_id)
                    index = np.random.randint(low=0, high=len(valid_lanes))
                    lane = valid_lanes[index]
                    del valid_lanes[index]
                    # reintroduce it at the start of the network
                    # Note, the position is 20 so you are not overlapping with the inflow car that is being removed.
                    # this allows the vehicle to be immediately inserted.
                    try:
                        self.k.vehicle.add(
                            veh_id=veh_id,
                            edge=self.entrance_edge,
                            type_id=str(type_id),
                            lane=str(lane),
                            pos="20.0",
                            speed="23.0")
                    except Exception as e:
                        print(e)
                    if len(valid_lanes) == 0:
                        break

            departed_ids = list(self.k.vehicle.get_departed_ids())
            if isinstance(departed_ids, tuple) and len(departed_ids) > 0:
                for veh_id in departed_ids:
                    if veh_id not in self._observed_ids:
                        self.k.vehicle.remove(veh_id)

            # update set of all reroute RL vehicles
            self.reroute_rl_ids = set(self.k.vehicle.get_rl_ids()) | self.reroute_rl_ids
        else:
            # reset
            self.reroute_rl_ids = set()

    def state_util(self, rl_id):
        """Return an array of headway, tailway, leader speed, follower speed.

        Also return a 1 if leader is rl 0 otherwise, a 1 if follower is rl 0 otherwise.
        If there are fewer than MAX_LANES the extra
        entries are filled with -1 to disambiguate from zeros.
        """
        veh = self.k.vehicle
        lane_headways = veh.get_lane_headways(rl_id).copy()
        lane_tailways = veh.get_lane_tailways(rl_id).copy()
        lane_leader_speed = veh.get_lane_leaders_speed(rl_id).copy()
        lane_follower_speed = veh.get_lane_followers_speed(rl_id).copy()
        leader_ids = veh.get_lane_leaders(rl_id).copy()
        follower_ids = veh.get_lane_followers(rl_id).copy()
        rl_ids = self.k.vehicle.get_rl_ids()
        is_leader_rl = [1 if l_id in rl_ids else 0 for l_id in leader_ids]
        is_follow_rl = [1 if f_id in rl_ids else 0 for f_id in follower_ids]
        diff = MAX_LANES - len(is_leader_rl)
        if diff > 0:
            # the minus 1 disambiguates missing cars from missing lanes
            lane_headways += diff * [-1]
            lane_tailways += diff * [-1]
            lane_leader_speed += diff * [-1]
            lane_follower_speed += diff * [-1]
            is_leader_rl += diff * [-1]
            is_follow_rl += diff * [-1]
        lane_headways = np.asarray(lane_headways) / 1000
        lane_tailways = np.asarray(lane_tailways) / 1000
        lane_leader_speed = np.asarray(lane_leader_speed) / 100
        lane_follower_speed = np.asarray(lane_follower_speed) / 100
        return np.concatenate((lane_headways, lane_tailways, lane_leader_speed,
                               lane_follower_speed, is_leader_rl,
                               is_follow_rl))

    def veh_statistics(self, rl_id):
        """Return speed, edge information, and x, y about the vehicle itself."""
        speed = self.k.vehicle.get_speed(rl_id) / 100.0
        lane = (self.k.vehicle.get_lane(rl_id) + 1) / 10.0
        return np.array([speed, lane])

    def step(self, rl_actions):
        """See parent class for more details; add option to reroute vehicles."""
        state, reward, done, info = super().step(rl_actions)
        if done['__all__']:
            # handle the edge case where a vehicle hasn't been put back when the rollout terminates
            if self.reroute_on_exit:
                for rl_id in self.reroute_rl_ids:
                    if rl_id not in state.keys():
                        done[rl_id] = True
                        reward[rl_id] = 0
                        state[rl_id] = -1 * np.ones(self.observation_space.shape[0])
            # you have to catch the vehicles on the exit edge, they have not yet
            # recieved a done when the env terminates
            on_exit_edge = [rl_id for rl_id in self.k.vehicle.get_rl_ids()
                            if self.k.vehicle.get_edge(rl_id) == self.exit_edge]
            for rl_id in on_exit_edge:
                done[rl_id] = True
                reward[rl_id] = 0
                state[rl_id] = -1 * np.ones(self.observation_space.shape[0])

        return state, reward, done, info

    def reset(self, new_inflow_rate=None):
        """Reset the environment."""
        state = super().reset(new_inflow_rate=new_inflow_rate)
        # update the network to set the downstream edge speed
        if self.env_params.additional_params["randomize_downstream_speed"]:
            min_speed = self.env_params.additional_params["min_downstream_speed"]
            max_speed = self.env_params.additional_params["max_downstream_speed"]
            downstream_speed = np.random.uniform(low=min_speed, high=max_speed)
            self.k.network.set_max_speed(self.exit_edge, downstream_speed)

        return state


class I210TestEnv(I210MultiEnv):
    # TODO(@evinitsky) clean this up, this shouldn't actually subclass MultiEnv
    """Version of I210 that overrides non-essential methods to speed up run-time for the non-RL case."""

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        return 0

    def get_state(self, **kwargs):
        """See class definition."""
        return {'fake': np.array([])}


class MultiStraightRoad(I210MultiEnv):
    """Partially observable multi-agent environment for a straight road. Look at superclass for more information."""

    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator)
        self.num_enter_lanes = 1
        self.entrance_edge = self.network.routes['highway_0'][0][0][0]
        self.exit_edge = self.network.routes['highway_0'][0][0][-1]

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # in the warmup steps, rl_actions is None
        if rl_actions:
            rl_ids = []
            accels = []
            for rl_id, actions in rl_actions.items():
                accel = actions[0]
                controller = self.k.vehicle.get_acc_controller(rl_id)
                accel = controller.compute_failsafe(accel, self)
                accels.append(accel)
                rl_ids.append(rl_id)

            # prevent the AV from blocking the entrance
            self.k.vehicle.apply_acceleration(rl_ids, accels)


class StraightRoadTestEnv(MultiStraightRoad):
    # TODO(@evinitsky) clean this up, this shouldn't actually subclass MultiEnv
    """Version of I210 that overrides non-essential methods to speed up run-time for the non-RL case."""

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=0, high=0, shape=(0,), dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        return

    def compute_reward(self, rl_actions, **kwargs):
        """See parent class."""
        return 0

    def get_state(self, **kwargs):
        """See class definition."""
        return {'fake': np.array([])}
