# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:25:21 2019

@author: Shalin
"""

import os
try:
    print('features_trial.py')
    os.system('python features_trial.py --model 1')
except Exception as e:
    print('Error in features_trail\n',e)    
try:
    print('trainonsample_testontrain.py')
    os.system('python trainonsample_testontrain.py --model 1')
except Exception as e:
    print('Error in trainonsample_testontrail\n',e)
try:
    print('individual')
    os.system('python individual.py --model 1')
except Exception as e:
    print('Error in individual\n',e)
        