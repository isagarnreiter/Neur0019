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
description = sc.loadmat("D:/UserFolder/neur0019/dataDescription.mat")

data = {} #create empty dictionary to host data 

for i in [200, 201, 202, 203, 204, 205, 206, 207, 209]:
    for j in range(1,14):
        if os.path.isfile(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}.mat") == True:
            mouse_trial = sc.loadmat(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}")
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

compressed_pickle('D:/UserFolder/neur0019/almost_all_data', almost_all_data) 

#dataf.to_pickle('data_concat')
#diusc = pd.read_pickle('data_concat')

#%%
test_data = decompress_pickle('D:/UserFolder/neur0019/test_data.pbz2') 
almost_all_data = decompress_pickle('D:/UserFolder/neur0019/almost_all_data.pbz2') 


mice_nbs = [200, 201, 202, 204, 205, 207, 209]    

wS1 = {}
for i in mice_nbs:
    wS1[f'{i}'] = [[], [], [], []]
    for j in range(1,11):
        try:
            for k in range(len(almost_all_data.loc[f'{i}_{j}'][0]['Trial_Counter'])):
                if almost_all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 1:
                    wS1[f'{i}'][1].append(np.array(almost_all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k]))
                elif almost_all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 3:
                    wS1[f'{i}'][3].append(np.array(almost_all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k]))
                elif almost_all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 0:
                    wS1[f'{i}'][0].append(np.array(almost_all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k]))
                elif almost_all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 2:
                    wS1[f'{i}'][2].append(np.array(almost_all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k]))
        except KeyError:
            break


#dictionary with the average response of a neuron for each mouse depending on the trial condition
wS1_stats = {'0':[], '1':[], '2':[], '3':[]}
for i in mice_nbs:
    for j in range(4):
        wS1_stats[f'{j}'].append([np.mean(np.array(wS1[f'{i}'][j]).T, axis=1)*10**6, np.std(np.array(wS1[f'{i}'][j]).T, axis=1)*10**6])

    

#LFP Data recorded at 20 kHz - so one value recorded every 0.05 ms.
#total recording length in 0.5 s


x = np.linspace(-300,200, 10000)


fig1 = plt.figure(figsize=(8,4))
ax1 = plt.subplot(111)
ax1.plot(x[5800:], wS1_stats[5800:], label = 'FA', alpha=0.8)
ax1.plot(x[5800:], mean_Hit_SEP[5800:], label='Hit', alpha=0.8)
ax1.plot(x[5800:], mean_Miss_SEP[5800:], label='Miss', alpha=0.8)
ax1.plot(x[5800:], mean_Catch_SEP[5800:], label='Catch', alpha=0.8)
ax1.set_ylim(-300, 100)
ax1.set_ylabel('uV')
ax1.set_xlabel('Time (ms)')
ax1.legend()

plt.plot()

