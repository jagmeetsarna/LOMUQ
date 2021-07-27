from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from tensorflow.keras import layers


#/data/LOMUQ/jssarna/data_topios.h5', 'r'
particleDensity = h5py.File('/data/LOMUQ/jssarna/data_topios.h5', 'r')

print(particleDensity.keys())

particleDensity = particleDensity['ParticleDensiy'][()]

#Train-Test split
test_percent = 0.2

test_point = np.round(len(particleDensity)*test_percent)
test_ind = int(len(particleDensity) - test_point)

train = particleDensity[:test_ind]
test = particleDensity[test_ind:]


train_arr =  np.expand_dims(train, axis=-1)
test_arr =  np.expand_dims(test, axis=-1)


x_train=[]
y_train=[]
for i in range(len(train)-10):    
    x_train.append(train_arr[i:i+10])
    y_train.append(train_arr[i+1:i+11])
    
x_train = np.array(x_train)  
y_train = np.array(y_train)


# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# length = 20 # Length of the output sequences (in number of timesteps)
# generator = TimeseriesGenerator(train_arr, train_arr, length=length, batch_size=1)

inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)

# early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


epochs = 15
batch_size =6

# Fit the model to the training data.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs
)

























# seq = Sequential()

# seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
#                    input_shape=(None,10,360,720,1), #important thing to note
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

                   
# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
#                activation='sigmoid',
#                padding='same', data_format='channels_last'))
# seq.compile(loss='binary_crossentropy', optimizer='adadelta')

# seq.summary()

# seq.fit(x_train,y_train,
#     batch_size=6,
#     epochs=25)

seq.save('LOMUQ_model.h5')