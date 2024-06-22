import numpy as np
import matplotlib.pyplot as plt

import os

from detector_dev.Process_I24_simulations.utils_classification import *

from detector_dev.Process_I24_simulations.i24_utils import get_sim_timeseries as get_sim_timeseries_i24
from detector_dev.Process_I24_simulations.i24_utils import get_attack_params as get_attack_params_i24


if __name__ == '__main__':
	timeseries_dict_extreme_attack = get_sim_timeseries_i24('/Volumes/My Passport for Mac/Misc/Dur_300_Mag_-0.5_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.05.csv')
	timeseries_dict_benign = get_sim_timeseries_i24('/Volumes/My Passport for Mac/i24_random_sample/simulations/1800_inflow/Dur_0.0027839584656419447_Mag_-1.1173175105601907_Inflow_1800_ACCPenetration_0.2_AttackPenetration_0.1_ver_1.csv')

	speeds_by_time_extreme_attack = get_speeds_by_time(timeseries_dict=timeseries_dict_extreme_attack)

	speed_by_time_benign = get_speeds_by_time(timeseries_dict=timeseries_dict_benign)

	plt.figure()

	times_extreme = []
	mean_speeds_extreme = []

	times_benign = []
	mean_speeds_benign = []


	for time in speeds_by_time_extreme_attack:
		times_extreme.append(time)
		mean_speeds_extreme.append(np.mean(speeds_by_time_extreme_attack[time]))

	for time in speed_by_time_benign:
		times_benign.append(time)
		mean_speeds_benign.append(np.mean(speed_by_time_benign[time]))



	plt.figure(figsize=[15,5])

	plt.plot(times_extreme,mean_speeds_extreme,label='Attacked',linewidth=5.0)
	plt.plot(times_benign,mean_speeds_benign,label='No attack',linewidth=5.0)
	plt.ylabel('MTS [m/s]',fontsize=20)
	plt.xlabel('Time [s]',fontsize=20)
	plt.xlim([600,1200])
	plt.legend(fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.title('Effect of standstill attack',fontsize=25)

	plt.savefig("strong_attack_on_MLFW_MTS.png",bbox_inches='tight')