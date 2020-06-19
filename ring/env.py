from flow.envs.base import Env
from gym.spaces.box import Box
from gym.spaces.multi_binary import MultiBinary
import numpy as np
from math import log
import math


class SensorRange:
    def __init__(self, veh_kernel, ego_id, max_range=30):
        self.__kernel = veh_kernel
        self.__ego_id = ego_id
        self.__max_range = max_range

    def SimpleFilter(self):
        ego_x = self.__kernel.get_orientation(self.__ego_id)[0]
        ego_y = self.__kernel.get_orientation(self.__ego_id)[1]
        # print( "ego: ", ego_x, ", ", ego_y)

        filtered_veh = []
        for veh_id in self.__kernel.get_ids():
            dx = self.__kernel.get_orientation(veh_id)[0] - ego_x
            dy = self.__kernel.get_orientation(veh_id)[1] - ego_y
            distance = math.sqrt(dx*dx+dy*dy)
            if distance < self.__max_range:
                filtered_veh.append([veh_id,self.__kernel.get_orientation(veh_id)[0],self.__kernel.get_orientation(veh_id)[1]])

        return np.delete(sorted(filtered_veh),0,1)

    def __call__(self):
        return np.array(self.SimpleFilter(),dtype=np.float).ravel()

    # SensorRange(self.k.vehicle, self.k.vehicle.get_rl_ids()[0])


class lcEnv(Env):

    @property
    def action_space(self):
        return MultiBinary(1)

    @property
    def observation_space(self):
        return Box(
            low=0,
            high=float("inf"),
            shape=(30,),
        )

    def step(self, rl_actions):
        """Advance the environment by one step.

        Assigns actions to autonomous and human-driven agents (i.e. vehicles,
        traffic lights, etc...). Actions that are not assigned are left to the
        control of the simulator. The actions are then used to advance the
        simulator by the number of time steps requested per environment step.

        Results from the simulations are processed through various classes,
        such as the Vehicle and TrafficLight kernels, to produce standardized
        methods for identifying specific network state features. Finally,
        results from the simulator are used to generate appropriate
        observations.

        Parameters
        ----------
        rl_actions : array_like
            an list of actions provided by the rl algorithm

        Returns
        -------
        observation : array_like
            agent's observation of the current environment
        reward : float
            amount of reward associated with the previous state/action pair
        done : bool
            indicates whether the episode has ended
        info : dict
            contains other diagnostic information from the previous action
        """
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash encodes whether the simulator experienced a collision
            crash = self.k.simulation.check_collision()

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.warmup_steps +
                self.env_params.horizon)  # or crash

        # compute the info for each agent
        infos = {}

        infos['crash'] = crash
        infos['arrived'] = self.k.vehicle.get_arrived_ids()

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash)
        else:
            reward = self.compute_reward(rl_actions, fail=crash)

        return next_observation, reward, done, infos

    def ready(self):
        for _ in range(1000):
            self.step(None)



    def _apply_rl_actions(self, rl_actions):
        if rl_actions == None:
            return
        else:
            rl_ids = self.k.vehicle.get_rl_ids()
            for rl_v in rl_ids:
                self.k.vehicle.apply_lane_change(rl_v, -rl_actions)


    def get_state(self, **kwargs):
        ret = np.zeros(30)
        if len(self.k.vehicle.get_rl_ids()) > 0:
            x = SensorRange(self.k.vehicle, self.k.vehicle.get_rl_ids()[0])
            for i, k in enumerate(x()):
                if i >= 30:
                    break
                ret[i] = k

        return ret


    def compute_reward(self, rl_actions, **kwargs):
        r = 0
        rl_ids = self.k.vehicle.get_rl_ids()

        try:
            rl_ids = self.k.vehicle.get_rl_ids()[0]

            # if self.k.vehicle.get_lane(rl_ids) == 0:
            #     r -= self.k.vehicle.get_position(rl_ids) / 100

            if self.k.simulation.check_collision():
                r += -100
            r += log(self.k.vehicle.get_speed(rl_ids))
        except:
            None
        return r

    def run(self):
        rl_ids = self.k.vehicle.get_rl_ids()[0]
        r = 0
        state, reward, done, info = self.step(1)
        r+= reward
        i=0
        while True:
            if self.k.simulation.check_collision():
                r += -500
                break
            elif i > 200:
                break
            _, reward, done, info = self.step(0)
            r += reward
            i+=1
        if self.k.vehicle.get_lane(rl_ids) == 1:
            r -= 500
        return state, r, True, info
