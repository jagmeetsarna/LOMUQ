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
import imageio
import os
from pathlib import Path
from sklearn import preprocessing
import h5py
from sklearn.preprocessing import StandardScaler

dir_path = "/data/LOMUQ/jssarna"
#/data/LOMUQ/jssarna/data_topios.h5', 'r'
#particleDensity = h5py.File(r'D:\TOPIOS Data\data\twentyone\data_topios.h5', 'r')
pathlist = Path(dir_path).rglob('*.h5')

paths=[]
filesDict={}
particleCountList = []
particleCountList_crop=[]
for path in pathlist:
    file = str(path)
    particleCount = h5py.File(file, 'r')
    particleCount = particleCount['ParticleCount'][()] #Every 5 days
    particleCountList.append(particleCount)

r,c = shape(particleCountList[0][0]) 

#Cropping the "action" area
dataDict={}
for i in range(len(particleCountList)):
    
    for j in range(len(particleCountList[i])):
        
        for rows in range(0,r-40,40):
            
            for columns in range(0,c-40,40):
                
                if(particleCountList[i][j][rows:rows+40,columns:columns+40].sum()>0):
                    
                    dataDict[(i,j,rows,columns)] = particleCountList[i][j][rows:rows+40,columns:columns+40].sum()
                    
                    
df = pd.DataFrame(dataDict.keys())
df.columns=['Data','day','rows','columns']

minRow =  df['rows'].min()   
maxRow = df['rows'].max()    
minCol =  df['columns'].min()   
maxCol = df['columns'].max() 


particleCountList = np.asarray(particleCountList)
particleCountList = particleCountList[...,minRow:maxRow+40,minCol:maxCol+40]
train = particleCountList[:6]
test = particleCountList[-1]  

#print(particleDensity.keys())
dataShape_train = shape(train)
dataShape_test = shape(test)


train = np.array(train).reshape(len(train),-1)
test = np.array(test).reshape(1,-1)
scaler = StandardScaler()
       
normalized_train = scaler.fit_transform(train)
normalized_test = scaler.transform(test)

normalized_train = normalized_train.reshape(dataShape_train[0],dataShape_train[1],dataShape_train[2],dataShape_train[3])


normalized_test = normalized_test.reshape(1,dataShape_test[0],dataShape_test[1],dataShape_test[2])


# imageio.mimwrite('particleDensity_5_49.gif', particleCountList[0])


#particleDensity = particleDensity['ParticleDensity'][()]

def turnIntoSequence(t,length=8,overlap=2):
    
    x_arr=[]
    y_arr=[]
    for i in range(0,len(t)-length,overlap):   
    
        x_arr.append(t[i:i+length])
        y_arr.append(t[i+1:i+length+1])
        
    return np.array(x_arr),np.array(y_arr)

train_seq=[]
test_seq=[]

for i in range(len(normalized_train)): #Combining them all in a sequence form
    
    train_seq.append(turnIntoSequence(normalized_train[i]))
    
for i in range(len(normalized_test)): #Combining them all in a sequence form
    
    test_seq.append(turnIntoSequence(normalized_test[i]))
    

def finalPrep(t=train_seq):
    
    t_seq = np.array(t)
    t_seq = t_seq.swapaxes(0, 1)
    
    t_seq =  np.expand_dims(t_seq, axis=-1)
    
    
    X = t_seq[0]
    y = t_seq[1]
    
    X_set=[]
    y_set=[]
    
    for i in range(len(X)):
        
        X_set.extend(X[i])
        y_set.extend(y[i])
    
    X = np.array(X_set)  
    y = np.array(y_set)  
    
    return X,y

X_train, y_train =  finalPrep(train_seq)
X_test, y_test =  finalPrep(test_seq)

inp = layers.Input(shape=(None,*X_train.shape[2:]))


x = layers.ConvLSTM2D(
    filters=20,
    kernel_size=(5, 5),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=15,
    kernel_size=(3, 3),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=15,
    kernel_size=(1, 1),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv3D(
    filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"
)(x)


model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(),
)
model.summary()
# early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)


epochs = 300
batch_size =2

Early_stopping = EarlyStopping(monitor='val_loss',patience=10)

modelCheckpoint = ModelCheckpoint(r'/data/LOMUQ/jssarna/checkpoint.hdf5', save_best_only = True)

# Fit the model to the training data.
h_callback = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[Early_stopping,modelCheckpoint]
    
)

model.save("/data/LOMUQ/jssarna/BestModel.hdf5")