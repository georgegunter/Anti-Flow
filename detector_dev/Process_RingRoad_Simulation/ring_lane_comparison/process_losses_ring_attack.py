import numpy as np

import torch

import time

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat

from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth

from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder


import flow.visualize.visualize_ring as visualize_ring

from flow.visualize.visualize_ring import get_measured_leader,get_rel_dist_to_measured_leader,get_vel_of_measured_leader

from copy import deepcopy

import sys

import utils

from utils import assess_relative_model_on_attack

import ray

import os

import csv

from sklearn.metrics import roc_curve,auc

def train_ring_relative_detector(GPS_penetration_rate,ring_length,emission_path,n_epoch=200):

    warmup_period = 50 #Wait until there's a well developed wave

    timeseries_dict = visualize_ring.get_sim_timeseries(csv_path=emission_path,warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())

    num_measured_vehicle_ids = int(np.floor(len(veh_ids)*GPS_penetration_rate))
    measured_veh_ids = deepcopy(veh_ids)

    for i in range(len(measured_veh_ids)-num_measured_vehicle_ids):
        rand_int = np.random.randint(0,len(measured_veh_ids))
        del measured_veh_ids[rand_int]

    ring_sim_dict = visualize_ring.get_sim_data_dict_ring(csv_path=emission_path,warmup_period=warmup_period)

    timeseries_list = []

    for veh_id in measured_veh_ids:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

    train_X = make_train_X(timeseries_list)

    model = get_cnn_lstm_ae_model(n_features=4)

    ring_length = 600

    model_file_name = 'ringlength'+str(ring_length)+'_2lanes_'+'_'+str(GPS_penetration_rate)+'percentGPS'

    print('Model: '+model_file_name)

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model

def get_losses_complete_obs(emission_path,model,warmup_period=100):

    timeseries_dict = visualize_ring.get_sim_timeseries(emission_path,warmup_period=warmup_period)

    veh_ids = list(timeseries_dict.keys())
   
    num_veh_processed = 0

    testing_losses_dict = dict.fromkeys(veh_ids)

    for veh_id in veh_ids:
        timeseries_list = []
        
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

        timeseries_list = [speed,accel,head_way,rel_vel]

        _,loss = sliding_window_mult_feat(model,timeseries_list)

        testing_losses_dict[veh_id]=loss

        num_veh_processed+=1

        sys.stdout.write('\r'+'Vehicles processed: '+str(num_veh_processed)+'\r')

    print('\n')
    
    smoothed_losses = dict.fromkeys(veh_ids)
    time = timeseries_dict[veh_ids[0]][:,0]
    
    #Get smoothed loss values:
    for veh_id in veh_ids:
        loss = testing_losses_dict[veh_id]
        smoothed_loss = loss_smooth(time,loss)
            
        smoothed_losses[veh_id] =  loss_smooth(time,loss)

        
    return smoothed_losses

@ray.remote
def get_max_losses_complete_obs_ray(emission_path,model):
    smoothed_losses = get_losses_complete_obs(emission_path,model)
    max_losses = []
    for veh_id in smoothed_losses:
        max_losses.append([veh_id,np.max(smoothed_losses[veh_id])])

    return [get_attack_params(emission_path),max_losses]

def get_attack_params(emission_path):
    num_chars = len(emission_path)

    i = 0
    while(emission_path[i:i+3] != 'TAD' and i < num_chars):i += 1
    i += 4
    j = i
    while(emission_path[j] != '_' and j < num_chars): j+= 1

    TAD = float(emission_path[i:j])

    i=j
    while(emission_path[i:i+3] != 'ADR' and i < num_chars):i += 1
    i += 4

    j = i
    while(emission_path[j] != '_' and j < num_chars): j+= 1

    ADR = float(emission_path[i:j])

    return [TAD,ADR]

def get_attack_identifier(TAD,ADR):
    return 'TAD_'+str(TAD)+'_ADR_'+str(ADR)+'_'

def get_all_max_losses_repo(emission_repo,model):
    emission_files = os.listdir(emission_repo)

    loss_ids = []  

    for file in emission_files:
        if(file[-3:] == 'csv'):
            emission_path = os.path.join(emission_repo,file)

            loss_ids.append(get_max_losses_complete_obs_ray.remote(emission_path,model))

    loss_results = ray.get(loss_ids)

    return loss_results

def get_loss_result_ver(attack_params,loss_results_repo):
    file_name_no_ver = 'TAD_'+str(attack_params[0])+'_ADR_'+str(attack_params[1])

    ver_num = 1

    all_files = os.listdir(loss_results_repo)

    for file in all_files:
        if(file_name_no_ver in file):ver_num +=1

    file_name = file_name_no_ver + '_ver_'+str(ver_num)+'.csv'

    return file_name

def write_loss_result(loss_result,loss_results_repo):

    attack_params = loss_result[0]

    file_name = get_loss_result_ver(attack_params,loss_results_repo)

    file_name = os.path.join(loss_results_repo,file_name)

    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')

        data = loss_result[1]

        for i in range(len(data)):
            csv_writer.writerow(data[i])

def write_all_loss_results(loss_results,loss_results_repo):

    for loss_result in loss_results:
        write_loss_result(loss_result,loss_results_repo)

def get_labeled_losses(loss_path_file):
    ben_losses = []
    mal_losses = []

    with open(loss_path_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            cfm_label = row[0]
            loss = row[1]
            if('adv' in cfm_label):
                mal_losses.append(float(loss))
            else:
                ben_losses.append(float(loss))

    return [ben_losses,mal_losses]


def get_all_loss_results_by_sim(loss_results_repo):

    losses_dict = {}

    all_files = os.listdir(loss_results_repo)

    result_files = []

    for file in all_files:
        if('csv' in file):
            file_path = os.path.join(loss_results_repo,file)
            result_files.append(file_path)

    for loss_path_file in result_files:

        [TAD,ADR] = get_attack_params(loss_path_file)
        key = get_attack_identifier(TAD,ADR)

        if(key not in losses_dict.keys()):
            losses_dict[key] = dict.fromkeys(['ben','mal'])
            losses_dict[key]['ben'] = []
            losses_dict[key]['mal'] = []

        [ben_losses,mal_losses] = get_labeled_losses(loss_path_file)


        losses_dict[key]['ben'].append(ben_losses)
        losses_dict[key]['mal'].append(mal_losses)

    return losses_dict



def get_all_loss_results(loss_results_repo):

    losses_dict = {}

    all_files = os.listdir(loss_results_repo)

    result_files = []

    for file in all_files:
        if('csv' in file):
            file_path = os.path.join(loss_results_repo,file)
            result_files.append(file_path)

    for loss_path_file in result_files:

        [TAD,ADR] = get_attack_params(loss_path_file)
        key = get_attack_identifier(TAD,ADR)

        if(key not in losses_dict.keys()):
            losses_dict[key] = dict.fromkeys(['ben','mal'])
            losses_dict[key]['ben'] = []
            losses_dict[key]['mal'] = []

        [ben_losses,mal_losses] = get_labeled_losses(loss_path_file)

        for loss in ben_losses:
            losses_dict[key]['ben'].append(loss)
        for loss in mal_losses:
            losses_dict[key]['mal'].append(loss)

    return losses_dict

def get_all_losses_AUC(losses_dict):

    attack_param_keys = list(losses_dict.keys())

    auc_results = []

    for key in attack_param_keys:
        ben_losses = losses_dict[key]['ben']
        mal_losses = losses_dict[key]['mal']

        labels = []
        losses = []

        for loss in ben_losses:
            labels.append(0)
            losses.append(loss)
        for loss in mal_losses:
            labels.append(1)
            losses.append(loss)

        fpr, tpr, thresholds = roc_curve(labels, losses, pos_label=1)

        auc_val = auc(fpr,tpr)

        temp =  get_attack_params(key)

        auc_results.append([temp[0],temp[1],auc_val])

    return auc_results

def scatter_plot_AUC(auc_results):
    auc_results_np_arr = np.array(auc_results)
    plt.scatter(auc_results_np_arr[:,0],auc_results_np_arr[:,1],c=auc_results_np_arr[:,2],s=150)
    plt.ylabel('Braking rate [m/s^2]',fontsize=20)
    plt.xlabel('Braking duration [s]',fontsize=20)
    plt.colorbar()

# def get_classifiction_results(loss_path_file,classifier):
#     [ben_losses,mal_losses] = get_labeled_losses(loss_path_file)

#     all_losses = []

#     for loss in ben_losses:
#         all_losses.append(loss)
#     for loss in mal_losses:
#         all_losses.append(loss)

#     labels = 





# def get_all_loss_results():
#     model = get_cnn_lstm_ae_model(n_features=4)

#     # Load in a trained model:
#     MODEL_PATH = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/models/cnn_lstm_ae_ringlength600_1lane__1.0percentGPS.pt'
#     model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

#     emission_repo = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep'

#     loss_results_repo = os.path.join(emission_repo,'loss_results_1.0_GPS')

#     # loss_results_repo = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep/loss_results_1.0_GPS'

#     loss_results_single_lane = get_all_max_losses_repo(emission_repo,model)

#     write_all_loss_results(loss_results_single_lane,loss_results_repo)


#     #initialize a model for loading in:
#     model = get_cnn_lstm_ae_model(n_features=4)

#     # Load in a trained model:
#     MODEL_PATH = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/models/cnn_lstm_ae_ringlength600_1.0percentGPS.pt'
#     model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

#     emission_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'

#     loss_results_repo = os.path.join(emission_repo,'loss_results_1.0_GPS')

#     # loss_results_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep/loss_results_1.0_GPS'

#     loss_results_double_lane = get_all_max_losses_repo(emission_repo,model)

#     write_all_loss_results(loss_results_double_lane,loss_results_repo)

#     return([loss_results_single_lane,loss_results_double_lane])


if __name__ == '__main__':

    # # FOR ONE LANE:

    # model = get_cnn_lstm_ae_model(n_features=4)

    # # Load in a trained model:
    # MODEL_PATH = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/models/cnn_lstm_ae_ringlength600_1lane__1.0percentGPS.pt'
    # model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

    # # emission_repo = '/Volumes/My Passport for Mac/single_lane_ring_road_attack_parameter_sweep'

    # # loss_results_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep/loss_results_1.0_GPS'

    # emission_repo = '/Volumes/My Passport for Mac/test'

    # loss_results_repo = os.path.join(emission_repo,'/loss_results_1.0_GPS')


    # loss_results = get_all_max_losses_repo(emission_repo,model)



    # # FOR TWO LANES:

    # initialize a model for loading in:
    # model = get_cnn_lstm_ae_model(n_features=4)

    # # Load in a trained model:
    # MODEL_PATH = '/Users/vanderbilt/Desktop/Research_2022/Anti-Flow/detector_dev/models/cnn_lstm_ae_ringlength600_1.0percentGPS.pt'
    # model.load_state_dict(torch.load(MODEL_PATH,map_location=torch.device('cpu')))

    # emission_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep'

    # loss_results_repo = '/Volumes/My Passport for Mac/double_lane_ring_road_attack_parameter_sweep/loss_results_1.0_GPS'

    [loss_results_single_lane,loss_results_double_lane] = get_all_loss_results()















