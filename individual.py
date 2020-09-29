# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:07:34 2019

@author: Shalin
"""

import sys
import argparse
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', action='store', type=int, dest="type",\
                        help="model to be used = 0:SVM,1:RandomForestClassifier,2:LogisticClassifier, 3:Naive Bayes",choices=range(0,4), required=True)

    return parser.parse_args(sys.argv[1:])

def featureExtraction(x, fs):
    f1 = np.mean(x,0).reshape((1,3))[0] # Means
    C = np.cov(x.T)
    f2=[]
    f2 = np.append(C[0][0:3],C[1][1:3])
    f2 = np.array(f2,dtype=np.float32)
    f2 = np.append(f2,[C[2][2]])
    f3 = (skew(x[:,0]), skew(x[:,1]), skew(x[:,2])) # Skewness
    f4 = (kurtosis(x[:,0], fisher=False), kurtosis(x[:,1], fisher=False), kurtosis(x[:,2],fisher=False)) # Kurtosis
    f5 = np.zeros(3)
    f6 = np.zeros(3)
    F = []
    for i in range(0,3):
        g = abs(np.fft.fft(x[:,i]))
        g = g[0:round(len(g)/2)]
        g[0] = 0
        w = fs * np.arange(0,len(g))/(2*len(g))
        v = max(g)
        idx = np.argmax(g)
        f5[i] = v
        f6[i] = w[idx]

# Putting together feature vector
    F = np.append(f1, f2)
    F = np.append(F,f3)
    F = np.append(F,f4)
    F = np.append(F,f5)
    F = np.append(F,f6)
    return F

def output(y,batch_size):
    size = int(np.ceil(len(y)/batch_size))
    y_new=[]
    for i in range(0,size):
        count_zero = (y[i*batch_size:i*batch_size+batch_size]==0).sum()
        count_one = (y[i*batch_size:i*batch_size+batch_size]==1).sum()
        if count_zero >= count_one:
            y_new.append(0)
        else:
            y_new.append(1)
    return y_new

def model(type=1,n_estimators=10):
    if type==0:
        from sklearn.svm import SVC
        print('SVM model')
        classifier = SVC(kernel='rbf', random_state = 0)
    elif type==1:
        from sklearn.ensemble import RandomForestClassifier
        print('RandomForest model')
        classifier = RandomForestClassifier(n_estimators = n_estimators, random_state = 0)
    elif type==2:
        from sklearn.linear_model import LogisticRegression
        print('Linear model')
        classifier = LogisticRegression(random_state=0)
    elif type==3:
        from sklearn.naive_bayes import GaussianNB
        print('Naive Bayes model')
        classifier = GaussianNB()
        
    return classifier

def fit_model(classifier,X,y):
    return classifier.fit(X,np.ravel(y))

def predict(classifier,X,y):
    #print score
    score=classifier.score(X, y)
    
    #predict
    y_pred= classifier.predict(X)

    cm=confusion_matrix(y,y_pred)
    return y_pred,cm,score
    

if __name__ == '__main__':
    start_time = datetime.now()
    
    FLAGS = get_args()
    train_session = ['Session13','Session01','Session05','Session06','Session07','Session12']
    fs = 50
    batch_size = 150
    cm={}
    y_pred={}
    score={}
    for sess in train_session:
        print('\n',sess)
        arm = pd.read_csv('Training Data/'+sess+'/armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        wrist = pd.read_csv('Training Data/'+sess+'/wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        y = pd.read_csv('Training Data/'+sess+'/detection.txt',skipinitialspace=True,header=None)
        y = y.as_matrix()

        arm = arm.as_matrix()
        wrist = wrist.as_matrix()
        size = int(np.ceil(len(arm)/batch_size))
        F = []
        
        for i in range(0,len(wrist)-batch_size):
    #        print(i)
            F.append(featureExtraction(arm[i:i+batch_size,:3],fs))
            F[i] = np.append(F[i],featureExtraction(arm[i:i+batch_size,3:],fs))
            F[i] = np.append(F[i],featureExtraction(wrist[i:i+batch_size,:3],fs))
            F[i] = np.append(F[i],featureExtraction(wrist[i:i+batch_size,3:],fs))
            
        #y = output(y,batch_size)
        y = y[batch_size:]
    
        #split train test data
        X_train, X_test, y_train, y_test = train_test_split(F, y, test_size=0.25, random_state=0)
    
        #feature scaling
        print('feature scaling')
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        #create classifier
        print('define classifier')
        if FLAGS.type == 1:
            n_estimators = 50
            classifier = model(n_estimators=n_estimators,type=FLAGS.type)
        else:
            classifier = model(type=FLAGS.type)
        
        #Fit Model
        print('fit model')
        classifier = fit_model(classifier,X_train,y_train)
        
        #predict accuracy
        print('accuracy')
        y_pred[sess],cm[sess],score[sess] = predict(classifier,X_test,y_test)
        print('save file')
        df = pd.DataFrame(y_pred[sess])
        df.to_csv('Individual Predict- batch-150/'+sess+'/detection.txt', index=False, header=False)
    
    print(score)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))