# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:45:24 2019

@author: sjrathi
"""
import numpy as np
import pandas as pd

armaccx,armaccy,armaccz,wristaccx,wristaccy,wristaccz,armgyrox,armgyroy,armgyroz,wristgyrox,wristgyroy,wristgyroz = [],[],[],[],[],[],[],[],[],[],[],[]
train=[]
d =[]
sess = ['Session01','Session02','Session03','Session05','Session06','Session07','Session12','Session13','Session15','Session16']
for sess_id in sess:    
    a = pd.read_csv('Training Data B/'+sess_id+'/armIMU.txt',skipinitialspace=True,delim_whitespace=True,header=None).values
    b = pd.read_csv('Training Data B/'+sess_id+'/wristIMU.txt',skipinitialspace=True,delim_whitespace=True,header=None).values
    c = pd.read_csv('Training Data B/'+sess_id+'/detection.txt',skipinitialspace=True,delim_whitespace=True,header=None).values
    d.append(len(a))
    arm_acc_x = a[:,0]
    arm_acc_y = a[:,1]
    arm_acc_z = a[:,2]
    arm_gyro_x = a[:,3]
    arm_gyro_y = a[:,4]
    arm_gyro_z = a[:,5]
    wrist_acc_x = b[:,0]
    wrist_acc_y = b[:,1]
    wrist_acc_z = b[:,2]
    wrist_gyro_x = b[:,3]
    wrist_gyro_y = b[:,4]
    wrist_gyro_z = b[:,5]
    for i in range(len(a)-150):
        armaccx.append(arm_acc_x[i:i+150])
        armaccy.append(arm_acc_y[i:i+150])
        armaccz.append(arm_acc_z[i:i+150])
        wristaccx.append(wrist_acc_x[i:i+150])
        wristaccy.append(wrist_acc_x[i:i+150])
        wristaccz.append(wrist_acc_x[i:i+150])
        armgyrox.append(arm_gyro_x[i:i+150])
        armgyroy.append(arm_gyro_x[i:i+150])
        armgyroz.append(arm_gyro_x[i:i+150])
        wristgyrox.append(wrist_gyro_x[i:i+150])
        wristgyroy.append(wrist_gyro_x[i:i+150])
        wristgyroz.append(wrist_gyro_x[i:i+150])
    train= np.append(train,c[150:])
#abcd = ['armaccx','armaccy','armaccz','wristaccx','wristaccy','wristaccz','armgyrox','armgyroy','armgyroz','wristgyrox','wristgyroy','wristgyroz']    

df = pd.DataFrame(armaccx)
df.to_csv('train/armaccx.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(armaccy)
df.to_csv('train/armaccy.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(armaccz)
df.to_csv('train/armaccz.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristaccx)
df.to_csv('train/wristaccx.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristaccy)
df.to_csv('train/wristaccy.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristaccz)
df.to_csv('train/wristaccz.txt',sep=' ',index=False,header=False)

df = pd.DataFrame(armgyrox)
df.to_csv('train/armgyrox.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(armgyroy)
df.to_csv('train/armgyroy.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(armgyroz)
df.to_csv('train/armgyroz.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristgyrox)
df.to_csv('train/wristgyrox.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristgyroy)
df.to_csv('train/wristgyroy.txt',sep=' ',index=False,header=False)
df = pd.DataFrame(wristgyroz)
df.to_csv('train/wristgyroz.txt',sep=' ',index=False,header=False)


df = pd.DataFrame(train)
df.to_csv('train/y_train.txt',index=False,header=False)
