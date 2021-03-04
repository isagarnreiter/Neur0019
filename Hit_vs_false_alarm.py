# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:26:57 2021

@author: Isabelle
"""
import scipy.io as sc
import pandas as pd
import os.path
import numpy as np
import bz2
import pickle
import _pickle as cPickle
import matplotlib.pyplot as plt

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
 
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

%matplotlib inline
#%%

#load the data from individual matlab files and save them in a dataframe
#each matlab file is named Data_Mouse_PL{i}_sessionNum_{j} with i being the 
#name of the mouse and j being the trial number
description = sc.loadmat("dataDescription")

data = {} #create empty dictionary to host data 

for i in range(210):
    for j in range(1,11):
        if os.path.isfile(f"./Data_Mouse_PL{i}_sessionNum_{j}.mat") == True:
            mouse_trial = sc.loadmat(f"Data_Mouse_PL{i}_sessionNum_{j}")
            mouse_trial = pd.DataFrame(mouse_trial['tmp'][0])
            mouse_trial = pd.DataFrame(mouse_trial.loc[0][['Session_Type','Trial_Counter','Trial_StartTime','Trial_ID', 'Trial_LFP_wS1', 'Trial_LFP_wS2', 'Trial_FirstLickTime']])
            data[f'{i}_{j}'] = mouse_trial
        else:
            break

DT_sessions = []
X_sessions = []
wS1_sessions = []
for i in range(219,226):
    mouse_trial_2 = sc.loadmat(f"Data_Mouse_PL{i}_sessionNum_1")
    mouse_trial_2 = pd.DataFrame(mouse_trial_2['tmp'][0])
    mouse_trial_2 = pd.DataFrame(mouse_trial_2.loc[0]['Trial_LFP_wS1'][0])
    
    if max(mouse_trial_2[0]) != np.nan:
        wS1_sessions.append(i)        

dataf = pd.concat(data)

compressed_pickle('test_data', dataf) 

#dataf.to_pickle('data_concat')
#diusc = pd.read_pickle('data_concat')

#%%
test_data = decompress_pickle('test_data.pbz2') 

plt.plot(test_data.loc['209_1'][0]['Trial_LFP_wS1'][0])