# Flow software:

This holds code coming from the Flow repository which is used for running the different traffic simulations in Atnti-Flow.

## Overview of architecture:
A brief description of a few of the most important portions of the flow architecture are given here:


- \controllers : Where different algorithms are stored which are subsequently used for controlling the motion of simulated vehicles.
- \networks : Where details on how vehicles navigate a given network must be specified. Includes elements such as possible routes, and edge connections.
- \core : How interfacing between Flow and the SUMO simulation environment is done.

