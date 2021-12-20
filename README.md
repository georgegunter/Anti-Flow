# Anti-Flow
This repository is built on and extends the [Flow software package](https://github.com/flow-project/flow) to explore cyber-security attacks on intelligent transportation systems. Where Flow is focused on how sparesely adopted AVs can improve traffic, this repository explores the opposite question: How can sparsely cyber-compromised AVs be used to degrade traffic flow?

## Getting started:
Anti-Flow relies on the SUMO simulation environment. We recommend using the installation instructions [here](https://sumo.dlr.de/docs/Installing/). Test the installation via the following commands:
- `which sumo`
- `sumo --version`
- `sumo-gui`
If succesful this will print the version of sumo being run, and then open the sumo graphical user interface. It is very important that SUMO be installed correctly as the rest of Anti-Flow/Flow relies on SUMO for performing simulation. 

We recommend setting up an anaconda environemnt to run Anti-Flow, which can be done using the following commands:
- `git clone https://github.com/georgegunter/Anti-Flow.git`
- `cd flow`
- `conda env create -f environment.yml`
- `conda activate anti_flow`
- `python setup.py develop`

To test whether the installation is succesfful try running:
- `conda activate anti_flow`
- `python examples/simulate.py ring`
This should open the sumo-gui and present the user with an ability to run a ring-road simulation environment.

<!-- ## Recreating research results:

To recreate simulations from "Compromised ACC vehicles can degrade current mixed-autonomy traffic performance while remaining stealthy against detection." run \examples\full_network_attack.py to create attacked traffic.
 -->

## Tutorials:

For a guide on how to set up an adversarial simulation environment see \tutorials. This covers running an initial simulation, adding a compromised AV, and subseuqently using an anomaly detection technique on 

## Citations:

Please be sure to cite not just Anti-Flow, but also Flow when using for academic reseach. 

cite Anit-Flow here:

FILL-IN


Cite the original flow repository using these papers:

Wu, C., Kreidieh, A. R., Parvate, K., Vinitsky, E., & Bayen, A. M. (2021). Flow: A modular learning framework for mixed autonomy traffic. IEEE Transactions on Robotics.

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M,  Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409). Available: http://proceedings.mlr.press/v87/vinitsky18a.html



