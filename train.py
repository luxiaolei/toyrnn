import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import trange

from model import LSTMModel

tr_data = pd.read_csv('occupancy_data/datatraining.txt')
val_data = pd.read_csv('occupancy_data/datatest.txt')
te_data = pd.read_csv('occupancy_data/datatest2.txt')

w_s, n_l, h_s, lr, l1_coe, l2_coe, clip, b_s = 5, 3, 500, 1e-3, 0.,0.,1, 1
val_step = 200

speed_test = 1

def reshape_data(df, w_s, b_s=1):
    y = df.Occupancy.values
    arr_x = df[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values

    # Slice time steps
    arr_x_sliced   = []

    for s in range(len(arr_x) - w_s + 1):
        e = s+w_s
        arr_x_sliced   += [arr_x[s:e, :].tolist()]
    arr_y_sliced = y[w_s-1:]
    
    return np.array(arr_x_sliced), arr_y_sliced
    



lstmModel = LSTMModel(w_s, n_l, h_s, lr, l1_coe=0., l2_coe=0., clip=1)

tr_x, tr_y = reshape_data(tr_data, 5)
val_x, val_y = reshape_data(tr_data, 5)


sess = tf.Session()
sess.run(lstmModel.init_op)

state = np.zeros([n_l, 2, b_s, h_s])
tr_loss , val_loss = [], []


for e in range(50):
    for i in trange(tr_x.shape[0]):
        fd = {
            lstmModel.x : [tr_x[i]],
            lstmModel.y : [tr_y[i]],
            lstmModel.drop_out: 1.,
            lstmModel.state_placeholder: state        
        }

        state, loss, prob, _ = sess.run([
                lstmModel.final_state,
                lstmModel.loss, 
                lstmModel.pre_probs, 
                lstmModel.train_op], feed_dict=fd)
        tr_loss += [loss]

        if not speed_test:
            if i % val_step == 0:
                state_val = np.zeros([n_l, 2, b_s, h_s]) 

                for i in trange(val_x.shape[0]):
                    fd = {
                        lstmModel.x : [val_x[i]],
                        lstmModel.y : [val_y[i]],
                        lstmModel.drop_out: 1.,
                        lstmModel.state_placeholder: state_val        
                    }

                    state_val, loss, prob = sess.run([
                            lstmModel.final_state,
                            lstmModel.lossX, 
                            lstmModel.pre_probs], feed_dict=fd)
                    val_loss += [loss]

