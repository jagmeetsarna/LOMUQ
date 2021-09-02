from argparse import ArgumentParser
import h5py
import pandas as pd
import os
import random   
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
from pathlib import Path
from sklearn import preprocessing
import traceback
from sklearn.preprocessing import StandardScaler

try:
    raise ValueError('A very specific bad thing happened.')
    dir_path = "/data/LOMUQ/jssarna"
    #/data/LOMUQ/jssarna/data_topios.h5', 'r'
    #particleDensity = h5py.File(r'D:\TOPIOS Data\data\twentyone\data_topios.h5', 'r')
    particleCountList = h5py.File(str(dir_path)+"/particleCountList.h5", 'r')
    hydrodynamic_U_dataList = h5py.File(str(dir_path)+"/hydrodynamic_U_dataList.h5", 'r')
    hydrodynamic_V_dataList = h5py.File(str(dir_path)+"/hydrodynamic_V_dataList.h5", 'r')
    
    particleCountList = particleCountList['ParticleCount'][()]
    hydrodynamic_V_dataList = hydrodynamic_V_dataList['hydrodynamic_V'][()]
    hydrodynamic_U_dataList = hydrodynamic_U_dataList['hydrodynamic_U'][()]
    
    
    total = len(hydrodynamic_V_dataList)-1
    train_V = hydrodynamic_V_dataList[:total]
    train_U = hydrodynamic_U_dataList[:total]
    
    train_X = np.sqrt((train_V**2 + train_U**2))
    
    train_Y = particleCountList[:total]
    
    test_U = hydrodynamic_U_dataList[-1]  
    test_V = hydrodynamic_V_dataList[-1] 
    
    test_X = np.sqrt((test_V**2 + test_U**2))
    test_Y = particleCountList[-1] 
    
    if len(shape(test_Y))<= 3:
        
        test_X = np.expand_dims(test_X,axis=0)
        test_Y = np.expand_dims(test_Y,axis=0)
        
    
    def normalizeFunc(train_X,train_Y,test_X,test_Y):
    
        scaler = MinMaxScaler()
        normalized_train_X = scaler.fit_transform(train_X.reshape(-1, 1)).reshape(train_X.shape)
        normalized_test_X = scaler.transform(test_X.reshape(-1, 1)).reshape(test_X.shape)
        normalized_train_Y = scaler.fit_transform(train_Y.reshape(-1, 1)).reshape(train_Y.shape)
        normalized_test_Y  = scaler.transform(test_Y.reshape(-1,1)).reshape(test_Y.shape)
    
        
        return normalized_train_X,normalized_train_Y,normalized_test_X,normalized_test_Y
    
    
    def turnIntoSequence(t_x,t_y,repeat,length=8,predictNxt=8,overlap=1):
        x_arr=[]
        y_arr=[]
        
        for j in range(repeat):
            
            for i in range(0,len(t_x[j])-length-predictNxt,overlap):   
                
                x_arr.append(t_x[j][i:i+length])
                y_arr.append(t_y[j][i+length:i+length+predictNxt])
            
        return np.array(x_arr),np.array(y_arr)
    
    normalized_train_X,normalized_train_Y,normalized_test_X,normalized_test_Y = normalizeFunc(train_X,train_Y,test_X,test_Y)
      
    train_seq_X, train_seq_Y  = turnIntoSequence(normalized_train_X,normalized_train_Y,len(normalized_train_X))
    train_seq_X2, train_seq_Y  = turnIntoSequence(normalized_train_Y,normalized_train_Y,len(normalized_train_X))
    
    test_seq_X, test_seq_Y  = turnIntoSequence(normalized_test_X,normalized_test_Y,len(normalized_test_X))
    test_seq_X2, test_seq_Y  = turnIntoSequence(normalized_test_Y,normalized_test_Y,len(normalized_test_X))
     
    
    
    def finalPrep(X,y):
        
        X =  np.expand_dims(X, axis=-1)
        y =  np.expand_dims(y, axis=-1)
        
        return X,y
    
    
    X1_train, y1_train =  finalPrep(train_seq_X, train_seq_Y)
    X2_train, y2_train =  finalPrep(train_seq_X2, train_seq_Y)
    
    X1_test, y1_test   =  finalPrep(test_seq_X, test_seq_Y )
    X2_test, y1_test   =  finalPrep(test_seq_X2, test_seq_Y )
    
    class MonteCarloDropout(keras.layers.Dropout):
      def call(self, inputs):
        return super().call(inputs, training=True)
    
    
    
    samples, timesteps,rows, columns, features = np.shape(X1_train) 
    
    visible1 = Input(shape=(None, rows, columns, features))
    
    model1 = ConvLSTM2D(filters=64, kernel_size=(5,5),activation='relu',padding="Same",return_sequences=True)(visible1)
    model1 = BatchNormalization()(model1)
    model1 = MonteCarloDropout(0.2)(model1)
    model1 = ConvLSTM2D(filters=64, kernel_size=(3,3),activation='relu',padding="Same",return_sequences= True)(model1)
    model1 = BatchNormalization()(model1)
    model1 = MonteCarloDropout(0.2)(model1)
    model1 = ConvLSTM2D(filters=64, kernel_size=(1,1),activation='relu',padding="Same",return_sequences= True)(model1)
    model1 = BatchNormalization()(model1)
    model1 = MonteCarloDropout(0.2)(model1)
    # model1 = Dense(15,activation='relu')(model1)
    
    visible2 = Input(shape=(None, rows, columns, features))
    
    model2 = ConvLSTM2D(filters=64,kernel_size=(5,5),activation='relu',padding="Same",return_sequences=True)(visible2)
    model2 = BatchNormalization()(model2)
    model2 = MonteCarloDropout(0.2)(model2)
    model2 = ConvLSTM2D(filters=32, kernel_size=(3,3),activation='relu',padding="Same",return_sequences=True)(model2)
    model2 = BatchNormalization()(model2)
    model2 = MonteCarloDropout(0.2)(model2)
    
    
    #model2 = Dense(15,activation='relu')(model2)
    
    merge = concatenate([model1,model2])
    
    dense = Dense(100,activation='relu')(merge)
    Output = Dense(1)(dense)
    
    model = Model(inputs=[visible1,visible2], outputs = Output)
    
    model.compile(optimizer='adam', loss='mae')
    
    model.summary()
    
    epochs = 100
    batch_size =2
    
    Early_stopping = EarlyStopping(monitor='val_loss',patience=30)
    
    modelCheckpoint = ModelCheckpoint('/data/LOMUQ/jssarna/best_version.hdf5', save_best_only = True)
    
    # Fit the model to the training data.
    h_callback = model.fit(
        [X1_train,X2_train],
        y1_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X1_test,X2_test], y1_test),
        shuffle = True,
        callbacks=[Early_stopping,modelCheckpoint]
        )
    
    
    model.save("/data/LOMUQ/jssarna/BestModel.hdf5")
except:    
    with open("exceptions.log", "a") as logfile:
        traceback.print_exc(file=logfile)
    raise
        