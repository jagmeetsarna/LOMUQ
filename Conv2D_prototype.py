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
from tensorflow.keras.layers.convolutional import Conv3D
from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt



particleDensity = h5py.File('/data/LOMUQ/jssarna/data_topios.h5', 'r')

particleDensity = particleDensity['ParticleDensity'][()]

#Train-Test split
test_percent = 0.2

test_point = np.round(len(particleDensity)*test_percent)
test_ind = int(len(particleDensity) - test_point)

train = particleDensity[:test_ind]
test = particleDensity[test_ind:]

train = np.array(train)  
test = np.array(test)

train_arr =  np.expand_dims(train, axis=-1)
test_arr =  np.expand_dims(train, axis=-1)


from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length = 20 # Length of the output sequences (in number of timesteps)
generator = TimeseriesGenerator(train_arr, train_arr, length=length, batch_size=1)



seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
                   input_shape=(None,360,720,1), #important thing to note
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

                   
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

seq.summary()

seq.fit(generator,epochs=5)

seq.save('LOMUQ_model.h5')