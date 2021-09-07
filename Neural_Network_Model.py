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
import pylab as plt
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import imageio
import os
from pathlib import Path
from sklearn import preprocessing
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Flatten,Input
from keras.layers.merge import concatenate,add
from keras.models import Model

def normalize(t):
    newmin=0
    newmax=1
    newrange=newmax-newmin

    t_normalize=[]

    oldmin=t.min()
    oldmax=t.max()
    oldrange=oldmax-oldmin
    
    for i in t:
        
        scale=(i-oldmin)/oldrange
        newvalue=(newrange*scale)+newmin
        t_normalize.append(newvalue)
    
    return t_normalize

try:
    
    dir_path = "/data/LOMUQ/jssarna"
    #/data/LOMUQ/jssarna/data_topios.h5', 'r'
    #particleDensity = h5py.File(r'D:\TOPIOS Data\data\twentyone\data_topios.h5', 'r')
    particleCountList = h5py.File(str(dir_path)+"/particleCountList.h5", 'r')
    hydrodynamic_U_dataList = h5py.File(str(dir_path)+"/hydrodynamic_U_dataList.h5", 'r')
    hydrodynamic_V_dataList = h5py.File(str(dir_path)+"/hydrodynamic_V_dataList.h5", 'r')



    particleCountList = particleCountList['ParticleCount'][()]
    hydrodynamic_V_dataList = hydrodynamic_V_dataList['hydrodynamic_V'][()]
    hydrodynamic_U_dataList = hydrodynamic_U_dataList['hydrodynamic_U'][()]
    
    hydrodynamic_U_dataList = normalize(hydrodynamic_U_dataList)
    hydrodynamic_V_dataList = normalize(hydrodynamic_V_dataList)


    total = len(hydrodynamic_V_dataList)-1-3
    train_V = hydrodynamic_V_dataList[:total]
    train_U = hydrodynamic_U_dataList[:total]
    
    train_Y = particleCountList[:total]
    
    test_V = hydrodynamic_V_dataList[-1] 
    test_U = hydrodynamic_U_dataList[-1]  
    
    test_Y = particleCountList[-1] 
    
    if len(shape(test_Y))<= 3:
        
        test_U = np.expand_dims(test_U,axis=0)
        test_V = np.expand_dims(test_V,axis=0)
        test_Y = np.expand_dims(test_Y,axis=0)
        
    
    
    
    def turnIntoSequence(t_x,t_y,repeat,length=10,predictNxt=1,overlap=4):
        x_arr=[]
        y_arr=[]
        
        for j in range(repeat):
            
            for i in range(0,len(t_x[j])-length-predictNxt,overlap):   
                
                x_arr.append(t_x[j][i:i+length])
                y_arr.append(t_y[j][i+length])
            
        return np.array(x_arr),np.array(y_arr)
    
    
      
    train_seq_U, train_seq_Y  = turnIntoSequence(train_U,train_Y,len(train_U))
    train_seq_V, train_seq_Y  = turnIntoSequence(train_V,train_Y,len(train_V))
    train_seq_X2, train_seq_Y  = turnIntoSequence(train_Y,train_Y,len(train_Y))
    
    test_seq_U, test_seq_Y  = turnIntoSequence(test_U,test_Y,len(test_U))
    test_seq_V, test_seq_Y  = turnIntoSequence(test_V,test_Y,len(test_V))
    test_seq_X2, test_seq_Y  = turnIntoSequence(test_Y,test_Y,len(test_Y))
     
    
    
    def finalPrep(X,y):
        
        X =  np.expand_dims(X, axis=-1)
        y =  np.expand_dims(y, axis=-1)
        
        return X,y
    
    
    X1_train, y_train =  finalPrep(train_seq_U, train_seq_Y)
    X2_train, y_train =  finalPrep(train_seq_V, train_seq_Y)
    X3_train, y_train =  finalPrep(train_seq_X2, train_seq_Y)
    
    X1_test, y_test   =  finalPrep(test_seq_U, test_seq_Y )
    X2_test, y_test   =  finalPrep(test_seq_V, test_seq_Y )
    X3_test, y_test   =  finalPrep(test_seq_X2, test_seq_Y )
    
    
    
    class MonteCarloDropout(keras.layers.Dropout):
      def call(self, inputs):
        return super().call(inputs, training=True)
    
    samples, timesteps,rows, columns, features = shape(X1_train) 
    
    visible1 = Input(shape=(None, rows, columns, features))
    
    model1 = ConvLSTM2D(filters=64, kernel_size=(10,10),activation='relu',padding="Same",return_sequences=True)(visible1)
    model1 = BatchNormalization()(model1)
    model1 = MonteCarloDropout(0.2)(model1)
    model1 = ConvLSTM2D(filters=64, kernel_size=(5,5),activation='relu',padding="Same",return_sequences= False)(model1)
    model1 = BatchNormalization()(model1)
    model1 = MonteCarloDropout(0.2)(model1)
    # model1 = Dense(15,activation='relu')(model1)
    
    visible2 = Input(shape=(None, rows, columns, features))
    
    model2 = ConvLSTM2D(filters=64,kernel_size=(10,10),activation='relu',padding="Same",return_sequences=True)(visible2)
    model2 = BatchNormalization()(model2)
    model2 = MonteCarloDropout(0.2)(model2)
    model2 = ConvLSTM2D(filters=64, kernel_size=(5,5),activation='relu',padding="Same",return_sequences=False)(model2)
    model2 = BatchNormalization()(model2)
    model2 = MonteCarloDropout(0.2)(model2)
    
    visible3 = Input(shape=(None, rows, columns, features))
    
    model3 = ConvLSTM2D(filters=64,kernel_size=(10,10),activation='relu',padding="Same",return_sequences=True)(visible3)
    model3 = BatchNormalization()(model3)
    model3 = MonteCarloDropout(0.2)(model3)
    model3 = ConvLSTM2D(filters=64, kernel_size=(5,5),activation='relu',padding="Same",return_sequences=False)(model3)
    model3 = BatchNormalization()(model3)
    model3 = MonteCarloDropout(0.2)(model3)
    
    merge = concatenate([model1,model2,model3])
    
    dense = Dense(200,activation='relu')(merge)
    Output = Dense(1)(dense)
    
    model = Model(inputs=[visible1,visible2,visible3], outputs = Output)
    
    model.compile(optimizer='adam', loss='mae')
    
    model.summary()
    
    epochs = 10
    batch_size =1
    
    Early_stopping = EarlyStopping(monitor='val_loss',patience=5)
    
    modelCheckpoint = ModelCheckpoint('/data/LOMUQ/jssarna/best_version1.hdf5', save_best_only = True)
    
    # Fit the model to the training data.
    h_callback = model.fit(
        [X1_train,X2_train,X3_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X1_test,X2_test,X3_test], y_test),
        shuffle = True,
        callbacks=[Early_stopping,modelCheckpoint]
        )
    
        model.save("/data/LOMUQ/jssarna/BestModel_sept.hdf5")
except:    
    with open("/data/LOMUQ/jssarna/exceptions1.log", "a") as logfile:
        traceback.print_exc(file=logfile)
    raise









