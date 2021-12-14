"""Contains the I-210 sub-network class."""
from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams

ADDITIONAL_NET_PARAMS = {
    # whether to include vehicle on the on-ramp
    "on_ramp": False,
    # whether to include the downstream slow-down edge in the network
    "ghost_edge": False,
}

EDGES_DISTRIBUTION = []
# ADD mainline edges:
NUM_EASTBOUND_MAIN = 24
EASTBOUND_MAIN = []
for i in range(1,NUM_EASTBOUND_MAIN+1):
  EASTBOUND_MAIN.append('Eastbound_'+str(i))
  EDGES_DISTRIBUTION.append('Eastbound_'+str(i))
# Add off-ramps:
NUM_EASTBOUND_OFF = 7
EASTBOUND_OFF = []
for i in range(1,NUM_EASTBOUND_OFF+1):
  EASTBOUND_MAIN.append('Eastbound_Off_'+str(i))
  EDGES_DISTRIBUTION.append('Eastbound_Off_'+str(i))
# Add on-ramps:
NUM_EASTBOUND_ON = 6
EASTBOUND_ON = []
for i in range(1,NUM_EASTBOUND_ON+1):
  EASTBOUND_MAIN.append('Eastbound_On_'+str(i))
  EDGES_DISTRIBUTION.append('Eastbound_On_'+str(i))

ON_RAMP_CONNECTIONS = [('Eastbound_On_1','Eastbound_7'),
                        ('Eastbound_On_2','Eastbound_9'),
                        ('Eastbound_On_3','Eastbound_11'),
                        ('Eastbound_On_4','Eastbound_14'),
                        ('Eastbound_On_5','Eastbound_17'),
                        ('Eastbound_On_6','Eastbound_23')
]

OFF_RAMP_CONNECTIONS = [('Eastbound_3','Eastbound_Off_1'),
                        ('Eastbound_7','Eastbound_Off_2'),
                        ('Eastbound_9','Eastbound_Off_3'),
                        ('Eastbound_12','Eastbound_Off_4'),
                        ('Eastbound_14','Eastbound_Off_5'),
                        ('Eastbound_15','Eastbound_Off_6'),
                        ('Eastbound_19','Eastbound_Off_7')
]


class I24SubNetwork(Network):
    """A network used to simulate the I-24 sub-network.

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
        main_rt = EASTBOUND_MAIN[3:10]
        on_ramp_rt = ['Eastbound_On_1']
        for rt in EASTBOUND_MAIN[6:10]:
            on_ramp_rt.append(rt)

        rts = {'Eastbound_3':[(main_rt, 1.0),],
                'Eastbound_On_1':[(on_ramp_rt, 1.0)],
                'Eastbound_7':[(['Eastbound_7','Eastbound_Off_2'],1.0)]
        }

        return rts

    # def specify_edge_starts(self):
    #     """See parent class."""
    #     if self.net_params.additional_params["ghost_edge"]:
    #         # Collect the names of all the edges.
    #         edge_names = [
    #             e[0] for e in self.length_with_ghost_edge
    #             if not e[0].startswith(":")
    #         ]

    #         edge_starts = []
    #         for edge in edge_names:
    #             # Find the position of the edge in the list of tuples.
    #             edge_pos = next(
    #                 i for i in range(len(self.length_with_ghost_edge))
    #                 if self.length_with_ghost_edge[i][0] == edge
    #             )

    #             # Sum of lengths until the edge is reached to compute the
    #             # starting position of the edge.
    #             edge_starts.append((
    #                 edge,
    #                 sum(e[1] for e in self.length_with_ghost_edge[:edge_pos])
    #             ))

    #     elif self.net_params.additional_params["on_ramp"]:
    #         # TODO: this will incorporated in the future, if needed.
    #         edge_starts = []

    #     else:
    #         # TODO: this will incorporated in the future, if needed.
    #         edge_starts = []

    #     return edge_starts

    # def specify_internal_edge_starts(self):
    #     """See parent class."""
    #     if self.net_params.additional_params["ghost_edge"]:
    #         # Collect the names of all the junctions.
    #         edge_names = [
    #             e[0] for e in self.length_with_ghost_edge
    #             if e[0].startswith(":")
    #         ]

    #         edge_starts = []
    #         for edge in edge_names:
    #             # Find the position of the edge in the list of tuples.
    #             edge_pos = next(
    #                 i for i in range(len(self.length_with_ghost_edge))
    #                 if self.length_with_ghost_edge[i][0] == edge
    #             )

    #             # Sum of lengths until the edge is reached to compute the
    #             # starting position of the edge.
    #             edge_starts.append((
    #                 edge,
    #                 sum(e[1] for e in self.length_with_ghost_edge[:edge_pos])
    #             ))

    #     elif self.net_params.additional_params["on_ramp"]:
    #         # TODO: this will incorporated in the future, if needed.
    #         edge_starts = []

    #     else:
    #         # TODO: this will incorporated in the future, if needed.
    #         edge_starts = []

    #     return edge_starts
