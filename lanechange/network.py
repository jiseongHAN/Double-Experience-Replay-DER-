from flow.networks import RingNetwork
network_name = RingNetwork

# input parameter classes to the network class
from flow.core.params import NetParams, InitialConfig

from env import *

# name of the network
name = "training_example"

# network-specific parameters
from flow.networks.ring import ADDITIONAL_NET_PARAMS
net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
net_params.additional_params['lanes'] = 2

# initial configuration to vehicles
initial_config = InitialConfig(spacing="random")

# vehicles class
from flow.core.params import VehicleParams

# vehicles dynamics models
from flow.controllers import IDMController, ContinuousRouter
from flow.core.params import SumoCarFollowingParams

vehicles = VehicleParams()
vehicles.add("human",
             acceleration_controller=(IDMController, {}),
             car_following_params=SumoCarFollowingParams(
             speed_mode='aggressive'
             ),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=13)

from flow.controllers import RLController
from flow.core.params import SumoLaneChangeParams
from flow.controllers.base_lane_changing_controller import BaseLaneChangeController


class RLLaneChangeController(BaseLaneChangeController):
    """A lane-changing model used to move vehicles into lane 0."""
    def __init__(self, veh_id, lane_change_params=SumoLaneChangeParams(lane_change_mode='aggressive')):
        """Instantiate an RL Controller."""
        BaseLaneChangeController.__init__(
            self,
            veh_id,
            lane_change_params)


vehicles.add(veh_id="rl",
            car_following_params=SumoCarFollowingParams(
            speed_mode='aggressive'
            ),
             acceleration_controller=(RLController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=1,
             lane_change_controller=(RLLaneChangeController, {}),
             lane_change_params=SumoLaneChangeParams(lane_change_mode='aggressive')
             )


from flow.core.params import SumoParams

sumo_params = SumoParams(sim_step=0.1, render=False, restart_instance=True)

from flow.core.params import EnvParams

# Define horizon as a variable to ensure consistent use across notebook
HORIZON= 1500

env_params = EnvParams(
    # length of one rollout
    horizon=HORIZON,

    additional_params={
        # maximum acceleration of autonomous vehicles
        "max_accel": 3,
        # maximum deceleration of autonomous vehicles
        "max_decel": 3,
        # bounds on the ranges of ring road lengths the autonomous vehicle
        # is trained on
        "ring_length": [500, 540],
        "lane_change_duration": 5,
    },
)



flow_params = dict(
    # name of the experiment
    exp_tag=name,
    # name of the flow environment the experiment is running on
    env_name=lcEnv,
    # name of the network class the experiment uses
    network=network_name,
    # simulator that is used by the experiment
    simulator='traci',
    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=sumo_params,
    # environment related parameters (see flow.core.params.EnvParams)
    env=env_params,
    # network-related parameters (see flow.core.params.NetParams and
    # the network's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,
    # vehicles to be placed in the network at the start of a rollout
    # (see flow.core.vehicles.Vehicles)
    veh=vehicles,
    # (optional) parameters affecting the positioning of vehicles upon
    # initialization/reset (see flow.core.params.InitialConfig)
    initial=initial_config
)

from flow.utils.registry import make_create_env

create_env, gym_name = make_create_env(params=flow_params, version=0)
