from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math



os.chdir(r"D:\TOPIOS Data\data\twentyone")

particleDensity = h5py.File("data_topios.h5", "r")

particleDensity = particleDensit['ParticleDensiy'][()]

#shape(x_train)
x_train=[]
y_train=[]
for i in range(len(particleDensity)-10): #len(k)-10
  

  x_train.append(particleDensity[i:i+10])
  y_train.append(particleDensity[i+1:i+11])

x_train = np.array(x_train)  
y_train = np.array(y_train)

x_arr =  np.expand_dims(x_train, axis=-1)
y_arr =  np.expand_dims(y_train, axis=-1)

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt

seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(5, 5),
                   input_shape=(None,360,720,1), #important thing to note
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

                   padding='same', return_sequences=True))
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')



seq.fit(x_arr, y_arr, batch_size=1,epochs=5, validation_split=0.05)
