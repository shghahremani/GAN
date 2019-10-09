import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.layers.recurrent import GRU, LSTM
from keras.layers.merge import concatenate
from keras.layers import Input, multiply, Lambda
from keras.layers.core import *
from keras import backend as K
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

train_df = pd.read_csv('dataset/train_dataset_F_ROCOF_tieline_150delay.csv')
test_df = pd.read_csv('dataset/test_dataset_F_ROCOF_tieline_150delay.csv')

# 3 col for LogDemand
#3 col for LogDexport
#8 col for tieline
#9 col for genfrequency
#1 for frequency
##### 24 *(50/4)= 288
### 288+6(delaystarta + delayend ,...)=294 cpl totally

from sklearn.utils import shuffle
test_df = shuffle(test_df)
train_df = shuffle(train_df)

test_SafeZone=test_df[test_df.unsafety_state==0]
test_UnSafeZone=test_df[(test_df.unsafety_state!=0) & (test_df.sysDown_state==0)]
#test_UnStableZone=test_df[(test_df.unsafety_state!=0) & (test_df.sysDown_state!=0)]

class DataLoader():
    def __init__(self, X, y, batch_size, seq_length, input_size):
        self.batch_size = batch_size
        self.seq_length = seq_length
        # -1 MEANS LAST INDEX
        X_shape = list(X.shape)
        X_shape[-1] = int(X_shape[-1] / input_size)
        step = int(X_shape[-1] / seq_length)
        lengh = step * seq_length
        # like image we have 3 dimention of T,P,PE
        X = X.reshape((X_shape[0], input_size, -1))[:, :,:lengh]  # here is hust want to ingnore extra data than for example 200
         # print(X.shape)

        self.X = X.reshape((X_shape[0], seq_length, -1))  ##number of records * seq_lengh*3*15
        # print(self.X.shape)

        self.y = y

    def dataset(self):
        return (self.X, self.y)


params = {
    "epochs": 400,
    "batch_size": 32,
    "seq_length": 20,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 500,
    "validation_split": 0.1,
    "input_size": 12
}
# for single branch like delayo


X_train = train_df.drop(['delayStart', 'delayEnd',	'delayLengh',	'delayCycle','unsafety_state','unsafety_time','sysDown_state','sysDown_time'], axis=1)
y_train = train_df[['delayLengh']]

X_test = test_df.drop(['delayStart', 'delayEnd',	'delayLengh',	'delayCycle','unsafety_state','unsafety_time','sysDown_state','sysDown_time'], axis=1)
X_test_SafeZone = test_SafeZone.drop(['delayStart', 'delayEnd',	'delayLengh',	'delayCycle','unsafety_state','unsafety_time','sysDown_state','sysDown_time'], axis=1)
X_test_UnSafeZone = test_UnSafeZone.drop(['delayStart', 'delayEnd',	'delayLengh',	'delayCycle','unsafety_state','unsafety_time','sysDown_state','sysDown_time'], axis=1)
#X_test_UnStableZone = test_UnStableZone.drop(['delayStart', 'delayEnd',	'delayLengh',	'delayCycle','unsafety_state','unsafety_time','sysDown_state','sysDown_time'], axis=1)
y_test = test_df[['delayLengh']]
y_test_SafeZone = test_SafeZone[['delayLengh']]
y_test_UnSafeZone = test_UnSafeZone[['delayLengh']]
#y_test_UnStableZone = test_UnStableZone[['delayLengh']]



scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(np.concatenate([X_test, X_train], axis=0))
scaler_y.fit(np.concatenate([y_test, y_train], axis=0))
scaler_X.fit(np.concatenate([X_test_SafeZone, X_train], axis=0))
scaler_y.fit(np.concatenate([y_test_SafeZone, y_train], axis=0))
scaler_X.fit(np.concatenate([X_test_UnSafeZone, X_train], axis=0))
scaler_y.fit(np.concatenate([y_test_UnSafeZone, y_train], axis=0))
#scaler_X.fit(np.concatenate([X_test_UnStableZone, X_train], axis=0))
#scaler_y.fit(np.concatenate([y_test_UnStableZone, y_train], axis=0))

data_loader = DataLoader(scaler_X.transform(X_test), scaler_y.transform(y_test), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_test, y_test = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_test_SafeZone), scaler_y.transform(y_test_SafeZone), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_test_SafeZone, y_test_SafeZone = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_test_UnSafeZone), scaler_y.transform(y_test_UnSafeZone), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_test_UnSafeZone, y_test_UnSafeZone = data_loader.dataset()

#data_loader = DataLoader(scaler_X.transform(X_test_UnStableZone), scaler_y.transform(y_test_UnStableZone), params["batch_size"],
                        # params["seq_length"], params["input_size"])
#X_test_UnStableZone, y_test_UnStableZone = data_loader.dataset()


data_loader = DataLoader(scaler_X.transform(X_train), scaler_y.transform(y_train), params["batch_size"],
                         params["seq_length"], params["input_size"])
X_train, y_train = data_loader.dataset()
# print(X_train.shape)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop, Adam
from keras.layers import Bidirectional
from keras import regularizers

###############attation

SINGLE_ATTENTION_VECTOR = False
TIME_STEPS = params["seq_length"]
import time
import keras


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


time_callback = TimeHistory()


def attention_3d_block(inputs, layer_name):
    # inputs.shape = (batch_size, time_steps, input_dim)

    name = layer_name
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    # print(a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name=name)(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""
    maxlen = params["seq_length"]

    inp = Input(shape=(layers[0], layers[1]))

    # attention = attention_3d_block(inp, "input_attention")
    layer_1 = GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu')(inp)
    dropout_1 = Dropout(params['dropout_keep_prob'])(layer_1)
    # attention_input=concatenate([attention, layer_1])
    # print(attention.shape)
    layer_2 = GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu')(dropout_1)
    # print(Bidirectional.shape)
    dropout_2 = Dropout(params['dropout_keep_prob'])(layer_2)

    layer_3 = GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False, activation='relu')(dropout_2)

    dropout_3 = Dropout(params['dropout_keep_prob'])(layer_3)

    '''layer_4 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=True, activation='relu'))(
        dropout_3)

    dropout_4 = Dropout(params['dropout_keep_prob'])(layer_4)
    layer_5 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False, activation='relu'))(
        dropout_4)

    dropout_5 = Dropout(params['dropout_keep_prob'])(layer_5)
    # layer_6 = (GRU(units=layers[2], batch_size=params["batch_size"], return_sequences=False,activation='relu'))(dropout_5)

    # dropout_6 = Dropout(params['dropout_keep_prob'])(layer_6)'''
    #attention_3 = attention_3d_block(dropout_3,'attention')
    # print(attention_3.shape)'''

    dense=Dense(units=layers[2], activation='relu')(dropout_3)
    # print(dense.shape)
    dropout_4 = Dropout(params['dropout_keep_prob'])(dense)
    #attention_mul = Flatten()(dropout_4)
    dense_2 = Dense(units=layers[3], activation='relu')(dropout_4)
    # print(dense_2.shape)
    # optimizer = Adam(clipvalue=0.5)
    adam = Adam(clipvalue=0.5, lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model = Model(inputs=inp, outputs=dense_2)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model


lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
# print()
saved_model = "3gru_F_ROCOF_tieline_150delay"
model = rnn_lstm(lstm_layer, params)

df_his = None

# Train RNN (LSTM) model with train set
history = model.fit(X_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_split=params['validation_split'],
                    callbacks=[ModelCheckpoint(filepath="models/" + saved_model, monitor='loss', verbose=1,
                                               save_best_only=True), \
                               ModelCheckpoint(filepath="models/" + saved_model + "_val", monitor='val_loss', verbose=1,
                                               mode='min', save_best_only=True), time_callback]
                    )


from keras.models import load_model
import time

model = load_model("models/%s" % (saved_model))  # , custom_objects={'Attention': Attention(params["seq_length"])}
#print(model.count_params())
#model.summary()
#start_time = time.time()
predict = scaler_y.inverse_transform(model.predict(X_test))
predict_SafeZone = scaler_y.inverse_transform(model.predict(X_test_SafeZone))
predict_UnSafeZone = scaler_y.inverse_transform(model.predict(X_test_UnSafeZone))
#predict_UnStableZone = scaler_y.inverse_transform(model.predict(X_test_UnStableZone))
#print("--- %s seconds ---" % (time.time() - start_time))
# exit()
y_true = scaler_y.inverse_transform(y_test)
y_true_SafeZone = scaler_y.inverse_transform(y_test_SafeZone)
y_true_UnSafeZone = scaler_y.inverse_transform(y_test_UnSafeZone)
#y_true_UnStableZone = scaler_y.inverse_transform(y_test_UnStableZone)
#y_true=(round(y_true_1/4))*4
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

#true=[y_true,y_true_SafeZone,y_true_UnSafeZone,y_true_UnStableZone]
#pred=[predict,predict_SafeZone,predict_UnSafeZone,predict_UnStableZone]
true=[y_true,y_true_SafeZone,y_true_UnSafeZone]
pred=[predict,predict_SafeZone,predict_UnSafeZone]
def NRMSD(y_true, y_pred):
    rmsd = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    y_min = min(y_true)
    y_max = max(y_true)

    return rmsd / (y_max - y_min)


def MAPE(y_true, y_pred):
    y_true_select = (y_true != 0)

    y_true = y_true[y_true_select]
    y_pred = y_pred[y_true_select]

    errors = y_true - y_pred
    return sum(abs(errors / y_true)) * 100.0 / len(y_true)


# In[13]:
saved_result="overal_result"
testSET_names=["all","SafeZone","UnSafeZone"]
for i,j,k in zip(true,pred,testSET_names):
    nrmsd = NRMSD(i, j)
    mape = MAPE(i, j)
    mae = mean_absolute_error(i, j)
    rmse = np.sqrt(mean_squared_error(i, j))
    print("NRMSD", nrmsd)
    print("MAPE", mape)
    print("neg_mean_absolute_error", mae)
    print("Root mean squared error", rmse)
    openfile = open("results/{0}_{1}.csv".format(saved_model,saved_result), "a")

    # for n in range(int(delayStart / AGC_TIME_STEP), int(delayEnd / AGC_TIME_STEP) - int(m / AGC_TIME_STEP)):
    # print("int(delayStart/AGC_TIME_STEP:",int(delayStart/AGC_TIME_STEP))
    # print ("len(Time)-int(m/AGC_TIME_STEP)",len(Time)-int(m/AGC_TIME_STEP))
    data=[k]
    data += [nrmsd]
    data += [mape]
    data += [mae]
    data += [rmse]

    openfile.write("%s\n" % ",".join(map(str, data)))
    openfile.close()

    pd.DataFrame({"predict": j.flatten(), "y_true": i.flatten()}).to_csv("results/{0}_{1}.csv".format(saved_model,k),
                                                                                    header=True)

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
from keras.utils.vis_utils import plot_model

# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


'''def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
attention_vectors = []
for i in range(2973):
    #testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    x_test =  np.expand_dims(X_test[i,:,:], axis=0)
    #print(X_test[i,:,:].shape)
    attention_vector = np.mean(get_activations(model,x_test,
                                               print_shape_only=True,
                                               layer_name='attention')[0], axis=2).squeeze()
    #print('attention =', attention_vector)
    #assert (np.sum(attention_vector) - 1.0) < 1e-5
    attention_vectors.append(attention_vector)

attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
print(attention_vector_final.shape)
# plot part.
import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()'''