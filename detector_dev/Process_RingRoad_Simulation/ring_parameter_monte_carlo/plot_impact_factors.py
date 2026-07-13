import numpy as np
import matplotlib.pyplot as plt


#related to ring road:
from flow.visualize.visualize_ring import get_sim_timeseries as get_sim_timeseries_ring
from detector_dev.Process_RingRoad_Simulation.process_ring_attacks import get_attack_params as get_attack_params_ring


from Data_Processing.sim_processing_utils import get_trajectory_timeseries


from detector_dev.Process_RingRoad_Simulation.get_performance_metrics_all import *
import os
import ray
import csv
from tqdm import tqdm

def get_impact_difference(attack_file_path,benign_file_path):

    attack_impacts = get_relevant_data(file_path=attack_file_path)

    benign_impacts = get_relevant_data(file_path=benign_file_path)

    return [attack_impacts[1],benign_impacts[1],attack_impacts[2],benign_impacts[2]]

def get_all_impact_differences(emission_repo,num_runs,default_attack_name,default_benign_name):
    all_impact_differences = []

    for i in tqdm(range(num_runs)):
        attack_file_name = default_attack_name+'_ver_'+str(i+1)+'.csv'
        benign_file_name = default_benign_name+'_ver_'+str(i+1)+'.csv'

        attack_file_path = os.path.join(emission_repo,attack_file_name)
        benign_file_path = os.path.join(emission_repo,benign_file_name)


        all_impact_differences.append(get_impact_difference(attack_file_path,benign_file_path))

    return all_impact_differences


# For parallelization:

@ray.remote
def get_impact_difference_ray(attack_file_path,benign_file_path):
    return get_impact_difference(attack_file_path,benign_file_path)


def get_all_impact_differences_ray(emission_repo,num_runs,default_attack_name,default_benign_name,file_name='impact_differences.csv'):
    all_impact_differences_ids = []

    for i in range(num_runs):
        attack_file_name = default_attack_name+'_ver_'+str(i+1)+'.csv'
        benign_file_name = default_benign_name+'_ver_'+str(i+1)+'.csv'

        attack_file_path = os.path.join(emission_repo,attack_file_name)
        benign_file_path = os.path.join(emission_repo,benign_file_name)


        all_impact_differences_ids.append(
            get_impact_difference_ray.remote(attack_file_path,benign_file_path))

    all_impact_differences = ray.get(all_impact_differences_ids)

    all_impact_differences_np_array = np.array(all_impact_differences)

    np.savetxt(file_name,all_impact_differences_np_array,delimiter=',')

    print('Impact factors saved.')

    return all_impact_differences_np_array


if __name__ == '__main__':

    ray.init(num_cpus=4)

    if(False):
        emission_repo = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/strong_attack'

        num_runs = 100

        default_attack_name = 'ring_600m_single_lane_TAD_10.0_ADR_-0.5'

        default_benign_name = 'ring_600m_single_lane_benign'

        file_name = 'impact_factors_strong.csv'

        all_impact_differences = get_all_impact_differences_ray(emission_repo,num_runs,default_attack_name,default_benign_name,file_name=file_name)


    if(False):

        emission_repo = '/Volumes/My Passport for Mac/Traffic_attack_sim_results/RDA/single_lane_ring_road_parameter_monte_carlo/medium_attack'

        num_runs = 100

        default_attack_name = 'ring_600m_single_lane_TAD_5.0_ADR_-0.25'

        default_benign_name = 'ring_600m_single_lane_benign'

        file_name = 'impact_factors_medium.csv'

        all_impact_differences = get_all_impact_differences_ray(emission_repo,num_runs,default_attack_name,default_benign_name,file_name=file_name)

    if(True):
        IF_medium = np.loadtxt('impact_factors_medium.csv',delimiter=',',dtype=float)
        IF_strong = np.loadtxt('impact_factors_strong.csv',delimiter=',',dtype=float)
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(IF_medium[:,0]-IF_medium[:,1],IF_medium[:,2]-IF_medium[:,3],'.')
        plt.title('Medium attack')
        plt.xlim([-2.0,0.5])
        plt.ylim([-1.5,6.0])

        plt.subplot(1,2,2)
        plt.plot(IF_strong[:,0]-IF_strong[:,1],IF_strong[:,2]-IF_strong[:,3],'.')
        plt.title('Strong attack')
        plt.xlim([-2.0,0.5])
        plt.ylim([-1.5,6.0])
        plt.show()



    # all_impact_differences = get_all_impact_differences(emission_repo,num_runs,default_attack_name,default_benign_name)

    # attack_file_path = os.path.join(medium_attack_emission_path,'ring_600m_single_lane_TAD_5.0_ADR_-0.25_ver_1.csv')

    # benign_file_path = os.path.join(medium_attack_emission_path,'ring_600m_single_lane_benign_ver_1.csv')

    # attack_impacts = get_relevant_data(file_path=attack_file_path)

    # benign_impacts = get_relevant_data(file_path=benign_file_path)