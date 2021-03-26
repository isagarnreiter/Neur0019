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
from scipy import stats

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

#description = sc.loadmat("D:/UserFolder/neur0019/dataDescription.mat")

data = {} #create empty dictionary to host data 

for i in [200, 201, 202, 203, 204, 205, 206, 207, 209, 222, 223, 224, 225]:
    print(f'Mouse {i}')
    for j in range(1,14):
        if os.path.isfile(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}.mat") == True:
            mouse_trial = sc.loadmat(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}")
            mouse_trial = pd.DataFrame(mouse_trial['tmp'][0])
            mouse_trial = pd.DataFrame(mouse_trial.loc[0][['Trial_Counter', 'Trial_ID', 'Trial_LFP_wS1', 'Trial_LFP_wS2', 'Trial_LFP_wM1', 'Trial_LFP_dCA1', 'Trial_LFP_mPFC', 'Trial_FirstLickTime']])
            data[f'{i}_{j}'] = mouse_trial
        else:
            break

#DT_sessions = []
#X_sessions = []
#wS1_sessions = []
#for i in range(219,226):
#    mouse_trial_2 = sc.loadmat(f"Data_Mouse_PL{i}_sessionNum_1")
#    mouse_trial_2 = pd.DataFrame(mouse_trial_2['tmp'][0])
#    mouse_trial_2 = pd.DataFrame(mouse_trial_2.loc[0]['Trial_LFP_wS1'][0])
#    
#    if max(mouse_trial_2[0]) != np.nan:
#        wS1_sessions.append(i)        

for i in mice_nbs:
    for j in range(1,14):
        if f'{i}_{j}' in data:
            for k in [2,3,4,5,6]:
                data[f'{i}_{j}'][0][k] = data[f'{i}_{j}'][0][k][:, 5800:10000]
        else:
            break

dataf = pd.concat(data)

index = ['Trial_LFP_wS1', 'Trial_LFP_wS2', 'Trial_LFP_wM1', 'Trial_LFP_dCA1', 'Trial_LFP_mPFC']

compressed_pickle('D:/UserFolder/neur0019/all_data_short', dataf) 

#dataf.to_pickle('data_concat')
#diusc = pd.read_pickle('data_concat')

#%%
all_data = decompress_pickle('D:/UserFolder/neur0019/all_data_short.pbz2') 

mice_nbs = [200, 201, 202, 203, 204, 205, 207, 209, 222, 223, 224, 225]    


#%% =============================================================================
# wS1 = dictionary with the sorted hit/Miss/CR/FA trials, for each Mouse
# wS1_stats = dictionary with the average response of a neuron for each mouse depending on the trial condition
# =============================================================================

regions = {'wS1': {}, 'wS2': {}, 'wM1': {}, 'dCA1': {}, 'mPFC': {}}
region_list = list(regions.keys())

LickTime_data = []
for i in mice_nbs:
    print(f'Mouse {i}')
    regions['wS1'][f'{i}'] = [[], [], [], []]
    regions['wS2'][f'{i}'] = [[], [], [], []]
    regions['wM1'][f'{i}'] = [[], [], [], []]
    regions['dCA1'][f'{i}'] = [[], [], [], []]
    regions['mPFC'][f'{i}'] = [[], [], [], []]
    for l in region_list:
        if np.isnan(all_data.loc[f'{i}_1'][0][f'Trial_LFP_{l}'][0][0]) == False:
            for j in range(1,11):
                try:
                    for k in range(len(all_data.loc[f'{i}_{j}'][0]['Trial_Counter'])):
                        if all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 1:
                            regions[l][f'{i}'][1].append(np.array(all_data.loc[f'{i}_{j}'][0][f'Trial_LFP_{l}'][k]))
                        elif all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 3:
                            regions[l][f'{i}'][2].append(np.array(all_data.loc[f'{i}_{j}'][0][f'Trial_LFP_{l}'][k]))
                        elif all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 0:
                            regions[l][f'{i}'][0].append(np.array(all_data.loc[f'{i}_{j}'][0][f'Trial_LFP_{l}'][k]))
                        elif all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0] == 2:
                                regions[l][f'{i}'][3].append(np.array(all_data.loc[f'{i}_{j}'][0][f'Trial_LFP_{l}'][k]))
                        
    #                        if np.isnan(all_data.loc[f'{i}_{j}'][0]['Trial_FirstLickTime'][k][0]) == False:
    #                            a = np.min(all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k][7000:10000])
    #                            b = np.where(all_data.loc[f'{i}_{j}'][0]['Trial_LFP_wS1'][k][7000:10000] == a)
    #                            b = b[0][0]
    #                            LickTime_data.append([a, b, all_data.loc[f'{i}_{j}'][0]['Trial_FirstLickTime'][k][0], all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0]])

                except KeyError:
                    break
        else:
            print(f'no {l} data for mouse {i}')


for j in region_list:
    for i in mice_nbs:        
        for k in [0,1,2,3]:
            regions[j][str(i)][k] = np.array(regions[j][str(i)][k])
    
        if len(regions[j][str(i)][0]) == 0:
            regions[j].pop(str(i))
        else:
            regions[j][str(i)] = np.array(regions[j][str(i)])



            
LickTime_data = np.array(LickTime_data, dtype=object)

for i in range(len(LickTime_data)):
    if LickTime_data[i, 3]==1:
        LickTime_data[i,3] = 'green'
    elif LickTime_data[i, 3]==3:
        LickTime_data[i,3] = 'blue'


stats = {wS1_stats:
wS1_stats = [[], [], [], []]
for i in mice_nbs:
    for j in range(4):
        a = (np.array(regions['wS1'][f'{i}'][j]).T)*10**6
        mean = np.mean(a, axis=1)
        sem = np.std(a, axis=1, ddof=1)/np.sqrt(np.size(a[0]))
        wS1_stats[j].append([mean, sem])

wS1_stats = np.array(wS1_stats)

for i in region_list:
    for j in mice_nb:
        if len(regions[region_list][str(i)][0]) == 0:
            

#%%
#plot all the trials for a single mice, decide which mouse by determining mice_nb

#LFP Data recorded at 20 kHz - so one value recorded every 0.05 ms.
#total recording length in 0.5 s


x = np.linspace(-10,200, 4200)
mice_nb = 1
labels = ['Miss', 'Hit', 'FA', 'CR']

fig1 = plt.figure(figsize=(8,4))
ax1 = plt.subplot(111)
for i in range(4):
    ax1.errorbar(x, wS1_stats[i][mice_nb], wS1_stats[i][mice_nb], label = labels[i], alpha=0.5)

ax1.set_ylim(-300, 100)
ax1.set_ylabel('uV')
ax1.set_xlabel('Time (ms)')
ax1.legend()

#%%
#set list with the average response, across mice for the same trial type
average_across_mice = []
for i in range(4): 
    average_across_mice.append(np.mean(np.array(wS1_stats[i,:,0].T), axis=1))

average_across_mice = np.array(average_across_mice)

fig2 = plt.figure(figsize=(8,4))
ax1 = plt.subplot(111)
for i in range(4):
    ax1.plot(x, average_across_mice[i], label = labels[i])
ax1.set_ylim(-300, 100)
ax1.set_ylabel('uV')
ax1.set_xlabel('Time (ms)')
ax1.legend()

#%%
#plot/compare the average response of neurons between 100-200 ms

data_rs = wS1_stats[0:3, :, 0, 2200:4200].reshape(3, 7*2000)

mean_late_resp = np.mean(data_rs, axis=1)
late_resp_sd = np.std(data_rs, axis=1, ddof=1)

data_all = np.mean(wS1_stats[0:3, :, 0, 8000:10000], axis=2)

fig3 = plt.figure(figsize=(5, 7))
ax1 = plt.subplot(111)
for j in range(len(wS1_stats[0])):
    ax1.plot([1,2,3], np.mean(wS1_stats[0:3, j, 0, 8000:10000], axis=1), c='grey')

ax1.errorbar([1,2,3], mean_late_resp, late_resp_sem, marker= 'o', ms=10, ls='')
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['Miss', 'Hit', 'FA'])
ax1.set_title('Average Response 100 to 200ms after Stimulus')
ax1.set_ylabel('uV')

tStat_MissFA, pValue_MissFA = stats.ttest_ind(data_all[0], data_all[2], equal_var = False)
tStat_HitFA, pValue_HitFA = stats.ttest_ind(data_all[1], data_all[2], equal_var = False)
tStat_MissHit, pValue_MissHit = stats.ttest_ind(data_all[0], data_all[1], equal_var = False)

#%%
#plot relating the amplitude and the time of the max response

from sklearn.linear_model import LinearRegression

amplitude = LickTime_data[:, 0]*10**6*(-1)
amplitude = amplitude.reshape(-1,1)
resp_time = LickTime_data[:, 1]/20+50
resp_time =resp_time.reshape(-1,1)
Licktime = LickTime_data[:,2]

model1 = LinearRegression().fit(amplitude, Licktime)
r_sq = model1.score(amplitude, Licktime)

model2 = LinearRegression().fit(resp_time, Licktime)
r_sq2 = model2.score(resp_time, Licktime)

x_reg = np.linspace(200, 2000, 11)
amplitude_pred = model1.coef_*x_reg+model1.intercept_
RespTime_pred = model2.coef_*x_reg+model2.intercept_

plt.figure(figsize = (10, 5))
ax1 = plt.subplot(121)
ax1.scatter(Licktime, amplitude, c=LickTime_data[:, 3], alpha = 0.2) 
ax1.plot(x_reg, amplitude_pred)
#ax1.set_ylim(0, 1450)

ax2 = plt.subplot(122)
ax2.scatter(Licktime, resp_time, c=LickTime_data[:, 3], alpha = 0.2)


