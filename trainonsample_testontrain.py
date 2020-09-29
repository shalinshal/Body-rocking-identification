# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 10:47:21 2019

@author: Shalin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:40:51 2019

@author: Shalin
"""
import sys
import argparse
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
#import pickle
from datetime import datetime

#from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

def predict(classifier,X):
    #print score
#    score=classifier.score(X, y)
    
    #predict
    y_pred= classifier.predict(X)
    y_pred = np.append([0]*150,y_pred)

#    cm=confusion_matrix(y,y_pred)
    
    return y_pred#, cm#, score

def predict_train(classifier,X,y):
    #print score
#    score=classifier.score(X, y)
    
    #predict
    y_pred= classifier.predict(X)
    y_pred = np.append([0]*150,y_pred)

    cm=confusion_matrix(y,y_pred)
    
    return y_pred, cm#, score

def features(arm,wrist,fs=50,batch_size=50):
    X=[]
    for i in range(0,len(wrist)-batch_size):
#        print(i)
        X.append(featureExtraction(arm[i:i+batch_size,:3],fs).tolist())
# =============================================================================
#         X[i] = np.append(X[i],featureExtraction(arm[i:i+batch_size,3:],fs))
#         X[i] = np.append(X[i],featureExtraction(wrist[i:i+batch_size,:3],fs))
#         X[i] = np.append(X[i],featureExtraction(wrist[i:i+batch_size,3:],fs))    
# =============================================================================
        X[i].extend(featureExtraction(arm[i:i+batch_size,3:],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,:3],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,3:],fs).tolist())  
    return X

if __name__ == '__main__':
    start_time = datetime.now()
    FLAGS = get_args()
    fs = 50
    batch_size = 150
    y_train=[]
    train_session = ['Session13','Session01','Session05','Session06','Session07','Session12']
    X_test={}
    y_pred={}
    cm={}
    score={}
    y_test={}
    
    print('Training')
    arm_train = pd.read_csv('armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
    wrist_train = pd.read_csv('wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
    y_train = pd.read_csv('detection.txt',delim_whitespace= True,skipinitialspace=True,header=None)
    y_train = y_train.as_matrix()
    y_train = y_train[batch_size:]
    arm_train = arm_train.as_matrix()
    wrist_train = wrist_train.as_matrix()
    X_train = features(fs=fs,batch_size=batch_size,arm=arm_train, wrist=wrist_train)
    
    #feature scaling
    print('feature scaling')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    
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
  
    #test
    print('Predicting detection')
    test_session = ['Session02','Session03','Session15','Session16']
    for i in test_session:   
        print(i)
        arm_test = pd.read_csv('Test Data 1/'+i+'/armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        wrist_test = pd.read_csv('Test Data 1/'+i+'/wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
#        y_test[i] = pd.read_csv('Test Data 1/'+i+'/detection.txt',delim_whitespace= True,skipinitialspace=True,header=None)
#        y_test[i] = y_test[i].as_matrix()
        arm_test = arm_test.as_matrix()
        wrist_test = wrist_test.as_matrix()
        
        X_test[i] = features(fs=fs,batch_size=batch_size,arm=arm_test, wrist=wrist_test)
        X_test[i] = sc.transform(X_test[i])
    
        #predict
#        y_pred[i],cm[i],score[i] = predict(classifier, X_test[i],y_test[i])
#        y_pred[i],cm[i] = predict(classifier, X_test[i],y_test[i])
        y_pred[i] = predict(classifier, X_test[i])
#        score[i] = (cm[i][0][0]+cm[i][1][1])/np.sum(cm[i]) 
        df = pd.DataFrame(y_pred[i])
        df.to_csv('Test Predict-trained on sample, batch-150/'+i+'/detection.txt', index=False, header=False)
    
    for i in train_session:   
        print(i)
        arm_test = pd.read_csv('Training Data/'+i+'/armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        wrist_test = pd.read_csv('Training Data/'+i+'/wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        y_test[i] = pd.read_csv('Training Data/'+i+'/detection.txt',delim_whitespace= True,skipinitialspace=True,header=None)
        y_test[i] = y_test[i].as_matrix()
        arm_test = arm_test.as_matrix()
        wrist_test = wrist_test.as_matrix()
        
        X_test[i] = features(fs=fs,batch_size=batch_size,arm=arm_test, wrist=wrist_test)
        X_test[i] = sc.transform(X_test[i])
    
        #predict
#        y_pred[i],cm[i],score[i] = predict(classifier, X_test[i],y_test[i])
        y_pred[i],cm[i] = predict_train(classifier, X_test[i],y_test[i])
#        y_pred[i] = predict(classifier, X_test[i])
        score[i] = (cm[i][0][0]+cm[i][1][1])/np.sum(cm[i]) 
        
        df = pd.DataFrame(y_pred[i])
        df.to_csv('Train Predict-trained on sample, batch-150/'+i+'/detection.txt', index=False, header=False)
    print(score)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    
                