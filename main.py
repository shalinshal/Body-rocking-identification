# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:30:20 2019

@author: Shalin
"""
import os
#import keras
#import tensorflow as tf
#import keras.backend as K
from itertools import chain
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
#from keras.utils import np_utils
import numpy as np
#import argparse
#import matplotlib.pyplot as plt
#import pandas as pd

def create_model():
    model = Sequential()
    # define CNN model
    model.add(Conv2D(256,kernel_size=(3,3),activation='relu',input_shape=(150,12,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    # define LSTM model
    model.add(LSTM(128,batch_input_shape=(32,100,1)))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation='softmax'))
    model.add(Dense(1,activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def load_data(cur_wrist,cur_arm):
    wrist = np.loadtxt(cur_wrist) 
    arm = np.loadtxt(cur_arm)
    w_a = np.append(wrist,arm,axis=1)
    f=[]
    for i in range(len(w_a)-150):
        f.append(w_a[i:150])
# =============================================================================
#     acc_1,gyro_1,acc_2,gyro_2 = np.split(w_a,4,axis=1)
#     f = np.zeros((len(arm),3,4))
#     f[:,:,0] = acc_1
#     f[:,:,1] = acc_2
#     f[:,:,2] = gyro_1
#     f[:,:,3] = gyro_2
# =============================================================================
    return f

def load_det(y):
    return np.loadtxt(y)

if __name__ =='__main__':
    
    label_path = 'Training Data B/'
    predict_file_name = 'prediction.txt'
    
    session_list = [1,2,3,5,6,7,12,13,15,16]
    data ,label={},{}
    for session_id in session_list:
        cur_wrist = os.path.join(label_path,'Session{:02d}'.format(session_id), 'wristIMU.txt')
        cur_arm = os.path.join(label_path,'Session{:02d}'.format(session_id), 'armIMU.txt')
        data[session_id] = load_data(cur_wrist,cur_arm)
    for session_id in session_list:
        y = os.path.join(label_path,'Session{:02d}'.format(session_id), 'detection.txt')
        label[session_id] = load_det(y)
        
    train_sess = [1,5,6,7,12,13,15,16]
    test_sess = [2,3] 
    [train_X, test_X] = map(lambda keys: {x: data[x] for x in keys}, [train_sess, test_sess])
    [train_y, test_y] = map(lambda keys: {x: label[x] for x in keys}, [train_sess, test_sess])

    X_train,y_train,X_test,y_test=[],[],[],[]
    for key,value in train_X.items():
        temp = [value]
        X_train.append(temp)
    for key,value in train_y.items():
        temp = [value[150:]]
        y_train.append(temp)    
    for key,value in test_X.items():
        temp = [value]
        X_test.append(temp)
    for key,value in test_y.items():
        temp = [value[150:]]
        y_test.append(temp) 
    
    X_train=[item for sublist in X_train for item in sublist]
    X_test=[item for sublist in X_test for item in sublist]
    X_train=[item for sublist in X_train for item in sublist]
    X_test=[item for sublist in X_test for item in sublist]
    model = create_model()
    history = model.fit(X_train, y_train, batch_size=32, epochs=10)