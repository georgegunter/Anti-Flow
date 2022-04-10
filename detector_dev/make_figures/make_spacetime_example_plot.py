import os
import matplotlib.pyplot as plt
from flow.visualize.visualize_ring import get_sim_data_dict_ring,get_ring_positions,stack_data_for_spacetime

from copy import deepcopy

# csv_path = os.path.join(os.getcwd(),sim_res_no_attack[1])

csv_path = '/Users/vanderbilt/Desktop/Research_2021/Anti-Flow/detector_dev/data/ring_variable_cfm_20220112-2037411642041461.872571-0_emission.csv'

sim_data_dict_full = get_sim_data_dict_ring(csv_path,warmup_period=900)

ring_length = 600

GPS_penetration_rate = 0.5

veh_ids = list(sim_data_dict_full.keys())

num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
measured_veh_ids = deepcopy(veh_ids)

for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
    rand_int = np.random.randint(0,len(measured_veh_ids))
    del measured_veh_ids[rand_int]

sim_data_dict = dict.fromkeys(measured_veh_ids)

for veh_id in measured_veh_ids:
    sim_data_dict[veh_id] = sim_data_dict_full[veh_id]



veh_ids = list(sim_data_dict.keys())
ring_positions = get_ring_positions(sim_data_dict,ring_length)
times,positions,speeds = stack_data_for_spacetime(sim_data_dict,ring_positions)

positions_mod_ring_length = np.mod(positions,ring_length)

fontsize = 25
dot_size = 8.0

pt.figure(figsize=[15,9])
# pt.title('Space time plot, ring length: '+str(ring_length),fontsize=fontsize)
pt.scatter(times,positions_mod_ring_length,c=speeds,s=dot_size)
pt.ylabel('Position [m]',fontsize=fontsize)
pt.xlabel('Time [s]',fontsize=fontsize)


# pt.plot([950,950],[0,ring_length],'k--',linewidth=3.0,label='Snapshot of simulation.')
# pt.legend(fontsize=fontsize,loc='upper left')

pt.xlim([900,1000])
pt.ylim([0,ring_length])

pt.xticks(fontsize=fontsize)
pt.yticks(fontsize=fontsize)

cbar = pt.colorbar(label='Speed [m/s]')
cbar.ax.tick_params(labelsize=fontsize)
pt.show()