# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:24:37 2019

@author: Shalin
"""
import pandas as pd

prediction={}
test_session = ['Session02','Session03','Session15','Session16']
for sess in test_session:
    pd.set_option('display.float_format', lambda x: '%.7e' % x)
    prediction[sess] = pd.read_csv('Test_Features/'+sess+'/prediction.txt',header=None)
    prediction[sess] = prediction[sess].values.ravel()
    df = pd.DataFrame(prediction[sess],dtype="{:0.7e}")
    df.to_csv('trial.txt')
#    for i in range(len(prediction[sess])):
#        prediction[sess][i] = "{:0.7E}".format(Decimal(prediction[sess][i]))
#        prediction[sess][i] = format(prediction[sess][i],'0.7E')
    

        