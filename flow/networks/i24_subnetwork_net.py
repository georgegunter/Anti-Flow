"""Contains the I-24 sub-network class."""
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

ADDITIONAL_NET_PARAMS = {
    # whether to include vehicle on the on-ramp
    "on_ramp": False,
    # whether to include the downstream slow-down edge in the network
    "ghost_edge": True,
    # Percentage of mainline-flow that takes off-ramp:
    "offramp_percent": .05,
}

# Adding mainline, on-ramp and off-ramp edges for Westbound simulation:
EDGES_DISTRIBUTION = ["Westbound_1", "Westbound_2.0", "Westbound_2", "Westbound_3",
                      "Westbound_4", "Westbound_5", "Westbound_6", "Westbound_7",
                      "Westbound_On_1", "Westbound_Off_2", ":202177572_1", ":202185444_1",
                      ":202185444_0", ":202192541_0", ":202192541_1", ":306535021_0", ":306535022_0"
                      ]

# we connect the on-ramp edges with their corresponding edged from the mainline
ON_RAMP_CONNECTIONS = [('Westbound_On_1', 'Westbound_4')]
# we connect the off-ramp edges with their corresponding edged from the mainline
OFF_RAMP_CONNECTIONS = [('Westbound_4', 'Westbound_Off_2')]

# Adding mainline, on-ramp and off-ramp edges for Eastbound simulation:
NUM_EASTBOUND_MAIN = 8

for i in range(4, NUM_EASTBOUND_MAIN+1):
    EDGES_DISTRIBUTION.append('Eastbound_'+str(i))

# Add off-ramps:
EDGES_DISTRIBUTION.append('Eastbound_Off_2')

# Add on-ramps:
EDGES_DISTRIBUTION.append('Eastbound_On_1')

# we connect the on-ramp edges with their corresponding edged from the mainline
ON_RAMP_CONNECTIONS.append([('Eastbound_On_1', 'Eastbound_7')])
# we connect the off-ramp edges with their corresponding edged from the mainline
OFF_RAMP_CONNECTIONS.append([('Eastbound_7', 'Eastbound_Off_2')])


class I24SubNetwork(Network):
    """A network used to simulate the I-24 subnetwork.

    Requires from net_params:

    * **on_ramp** : whether to include vehicle on the on-ramp
    * **ghost_edge** : whether to include the downstream slow-down edge in the
      network

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import I24SubNetwork
    >>>
    >>> network = I210SubNetwork(
    >>>     name='I-24_subnetwork',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams()
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize the I210 sub-network scenario."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super(I24SubNetwork, self).__init__(
            name=name,
            vehicles=vehicles,
            net_params=net_params,
            initial_config=initial_config,
            traffic_lights=traffic_lights,
        )

    def specify_routes(self, net_params):
        """See parent class."""
        # specifying the routs for the west flow
        main_rt_west = EDGES_DISTRIBUTION[2:8]
        on_ramp_rt_west = ['Westbound_On_1']
        for rt in EDGES_DISTRIBUTION[4:8]:
            on_ramp_rt_west.append(rt)

        off_ramp_rt_west = EDGES_DISTRIBUTION[2:5]
        off_ramp_rt_west.append('Westbound_Off_2')

        # specifying the routs for the east flow
        main_rt_east = EDGES_DISTRIBUTION[10:15]
        on_ramp_rt_east = ['Eastbound_On_1']
        for rt in EDGES_DISTRIBUTION[13:15]:
            on_ramp_rt_east.append(rt)
        off_ramp_rt_east = EDGES_DISTRIBUTION[10:14]
        off_ramp_rt_east.append('Eastbound_Off_2')

        # Final/overall route(including eastbound nd westbound flow)
        rts = {
            'main_route_west': [
                (main_rt_west, 1-net_params.additional_params["offramp_percent"]),
                (off_ramp_rt_west, net_params.additional_params["offramp_percent"])
            ],
            'Westbound_On_1': [(on_ramp_rt_west, 1)],
            'Westbound_4': [
                (['Westbound_4', 'Westbound_Off_2'], net_params.additional_params["on_offramp_percent"]),
                (['Westbound_4', 'Westbound_5', 'Westbound_6', 'Westbound_7'],
                 1-net_params.additional_params["on_offramp_percent"])],
            'main_route_east': [
                (main_rt_east, 1-net_params.additional_params["offramp_percent"]),
                (off_ramp_rt_east, net_params.additional_params["offramp_percent"])
            ],
            'Eastbound_On_1': [(on_ramp_rt_east, 1)],
            'Eastbound_7': [(['Eastbound_7', 'Eastbound_Off_2'], 1)]
        }

        return rts
