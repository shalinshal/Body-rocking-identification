# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:46:07 2019

@author: sjrathi
"""

# lstm model
# cnn lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        print(name)	
        data = load_file(prefix+'/' + name)[:1000,:]
        loaded.append(data)
	# stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group
	# load all 9 files as a single array
    filenames = ['armaccx.txt','armaccy.txt','armaccz.txt','wristaccx.txt','wristaccy.txt','wristaccz.txt','armgyrox.txt','armgyroy.txt','armgyroz.txt','wristgyrox.txt','wristgyroy.txt','wristgyroz.txt']    
    X = load_group(filenames, filepath)
	# load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X,y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    print('load data')
    # load all train
#    trainX, trainy = load_dataset_group('test', prefix)
    trainX,trainy= load_dataset_group('test', prefix)
	# load all test
#    testX, testy = trainX[:-184690,:,:],trainy[:-184690,:]
#    testy = trainy[:-184690,:]
#	print(testX.shape, testy.shape)
	# zero-offset class values
#    trainy = trainy - 1
#    testy = testy - 1
	# one hot encode y
#    trainy = to_categorical(trainy)
#    testy = to_categorical(testy)
#    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX,trainy#,testy#, trainy, testX, testy

def load_test():
    print('load test')
    filepath='test/'
    filenames = ['armaccx.txt','armaccy.txt','armaccz.txt','wristaccx.txt','wristaccy.txt','wristaccz.txt','armgyrox.txt','armgyroy.txt','armgyroz.txt','wristgyrox.txt','wristgyroy.txt','wristgyroz.txt']    
    X = load_group(filenames, filepath)
    return X
     

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 1, 10, 512
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# reshape data into time steps of sub-sequences
    n_steps, n_length = 3, 50
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
	model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return history,accuracy,model

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats=10):
	# repeat experiment
    scores = list()
    history={}
    for r in range(repeats):
        history[r],score,model = evaluate_model(train_X, train_y, valX, valy)
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
	# summarize results
    summarize_results(scores)
    return model,history

#load data
trainX,trainy = load_dataset()

valX,valy,train_X,train_y= trainX[-184690:,:,:],trainy[-184690:,:],trainX[:-184690,:,:],trainy[:-184690,:]
test_X = load_test()
test_X = test_X.reshape((test_X.shape[0], 3, 50, 12))

# run the experiment
model,history = run_experiment(1)
df = pd.DataFrame(history[0].history)
df.to_csv('history.csv')
y = model.predict(test_X)
model.save('my_model.h5')
pred_8 = np.append([0]*150, y[:139599,:])
pred_9 = np.append([0]*150, y[139599:327282,:])
pred_11 = np.append([0]*150, y[327282:1131312,:])
pred_17 = np.append([0]*150, y[-167748:,:])
df = pd.DataFrame(pred_8)
df.to_csv('Test Data 2/Session08/pred.txt')
df = pd.DataFrame(pred_9)
df.to_csv('Test Data 2/Session09/pred.txt')
df = pd.DataFrame(pred_11)
df.to_csv('Test Data 2/Session11/pred.txt')
df = pd.DataFrame(pred_17)
df.to_csv('Test Data 2/Session17/pred.txt')
