import numpy as np
import time
from time import time as timer_start

from Detectors.Deep_Learning.AutoEncoders.utils import SeqDataset,train_epoch,eval_data,train_model,get_cnn_lstm_ae_model,make_train_X,sliding_window_mult_feat
from Detectors.Deep_Learning.AutoEncoders.utils import get_loss_filter_indiv as loss_smooth
from Detectors.Deep_Learning.AutoEncoders.cnn_lstm_ae import CNNRecurrentAutoencoder

from process_i24_losses import get_sim_timeseries


def train_i24_detection_model(timeseries_data,warmup_period,n_epoch=50):

	timeseries_list = []

    for veh_id in timeseries_data:
        #[time,speed,headway,accel,leader_speed,fuel_consumption]
        speed = timeseries_dict[veh_id][:,1]
        accel = np.gradient(speed,.1)
        head_way = timeseries_dict[veh_id][:,2]
        rel_vel = timeseries_dict[veh_id][:,3]
        
        timeseries_list.append([speed,accel,head_way,rel_vel])

    train_X = make_train_X(timeseries_list)

    model = get_cnn_lstm_ae_model(n_features=4)

    model_file_name = 'i24_cnn_lstm_ae_detection_model'

    print('Beginning training...')
    begin_time = time.time()
    model = train_model(model,train_X,model_file_name,n_epoch=n_epoch)
    finish_time = time.time()
    print('Finished training, total time: '+str(finish_time-begin_time))

    return model



if __name__ == '__main__':
	training_data_path = '/Volumes/My Passport for Mac/benign_initial_i24/I-24_benign.csv'
	warmup_period = 1200
	timeseries_data = get_sim_timeseries(csv_path=training_data_path,warmup_period=warmup_period)

	i24_detection_model = 









