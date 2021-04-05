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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec, colors

def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
 
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pd.read_pickle(data)
    return data

def ms_to_freq(a):
    return (a+100)*2

def freq_to_ms(a):
    return 0.5*a-100

def take_second(elem):
    return elem[1]

def binArray(data, axis, binstep, binsize):
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

%matplotlib inline
#%%

#load the data from individual matlab files and save them in a dataframe
#each matlab file is named Data_Mouse_PL{i}_sessionNum_{j} with i being the 
#name of the mouse and j being the trial number

#description = sc.loadmat("D:/UserFolder/neur0019/dataDescription.mat")

data = {} #create empty dictionary to host data 

for i in [200, 201, 202, 203, 204, 205, 207, 209, 222, 223, 224, 225]:
    print(f'Mouse {i}')
    for j in range(1,11):
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

mice_nbs = [200, 201, 202, 204, 205, 207, 209, 222, 223, 224, 225]    


#%% =============================================================================
# wS1 = dictionary with the sorted hit/Miss/CR/FA trials, for each Mouse
# wS1_stats = dictionary with the average response of a neuron for each mouse depending on the trial condition
# =============================================================================

regions = {'wS1': {}, 'wS2': {}, 'wM1': {}, 'dCA1': {}, 'mPFC': {}}
region_list = list(regions.keys())

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


#%%
stat = {'wS1': [[], [], [], []], 'wS2': [[], [], [], []], 'wM1': [[], [], [], []], 'dCA1': [[], [], [], []], 'mPFC': [[], [], [], []]}
stat_list = list(stat.keys())

for k in region_list:
    for i in mice_nbs:
        try:
            for j in range(4):
                a = (np.array(regions[k][f'{i}'][j]).T)*10**6
                mean = np.mean(a, axis=1)
                sem = np.std(a, axis=1, ddof=1)/np.sqrt(np.size(a[0]))
                stat[k][j].append([mean, sem])
        except:
            KeyError

for j in stat_list:
    stat[j] = np.array(stat[j])
    for k in range(4):
            try:
                stat[j][k] = np.array(stat[j][k])
                for l in range(11):    
                    stat[j][k][l] = np.array(stat[j][k][l])
            except:
                KeyError


#%%
#plot all the trials for a single mice, decide which mouse by determining mice_nb

#LFP Data recorded at 20 kHz - so one value recorded every 0.05 ms.
#total recording length in 0.5 s


x = np.linspace(-10,200, 4200)
mice_nb = 3
labels = [['Miss', 'red'], ['Hit', 'green'], ['FA', 'blue'], ['CR', 'orange']]
region_index = 4
act_mouse_nb = list(regions[region_list[region_index]].keys())[mice_nb]

fig1 = plt.figure(figsize=(8,4))
ax1 = plt.subplot(111)
for i in range(4):
    ax1.plot(x, stat[stat_list[region_index]][i][mice_nb][0], label = labels[i][0], alpha=0.8)

ax1.set_title(f'average SEP for mouse {act_mouse_nb} in {stat_list[region_index]}')
    

ax1.set_ylim(-300, 100)
ax1.set_ylabel('uV')
ax1.set_xlabel('Time (ms)')
ax1.legend()

#%%
early_days = stat
#%%
late_days = stat

#%%
#set list with the average response, across mice for the same trial type
#plot total average response for a single trial 
#plot/compare the average response of neurons between 100-200 ms

region = 'mPFC'
lower= 50 #in ms
upper = 200

dim =late_days[region][0:3, :, 0, lower*20+200:upper*20+200].shape[1]

data_rs = late_days[region][0:3, :, 0, lower*20+200:upper*20+200].reshape(3, dim*((lower*20+200)-(upper*20+200)))
mean_late_resp = np.mean(data_rs, axis=1)
late_resp_sd = np.std(data_rs, axis=1, ddof=1)

data_all = np.mean(late_days[region][0:3, :, 0, lower*20+200:upper*20+200], axis=2)

Miss_FA = stats.wilcoxon(data_all[0], data_all[2])
Hit_FA =  stats.wilcoxon(data_all[1], data_all[2])
Miss_Hit = stats.wilcoxon(data_all[0], data_all[1])

print('Miss-FA', Miss_FA)
print('Hit-FA', Hit_FA)
print('Miss-Hit', Miss_Hit)


fig1 = plt.figure(figsize = (10, 4))
fig1.suptitle(region)
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[5, 2])

ax0 = fig1.add_subplot(spec[0])

for i in range(4):
    ax0.errorbar(x, np.mean(late_days[region][i,:,0].T, axis=1), np.mean(late_days[region][i,:,1].T, axis=1), alpha=0.03, c = labels[i][1])
    ax0.plot(x, np.mean(late_days[region][i,:,0].T, axis=1), c = labels[i][1], label = labels[i][0])
    
ax0.vlines(0, -300, 100, color='grey')
ax0.plot(np.linspace(lower, upper, 5), np.ones(5)*(-200), c='black')

ax0.set_ylim(-300, 100)
ax0.set_ylabel('Amplitude (uV)')
ax0.set_xlabel('Time from stimulus (ms)')
ax0.set_yticks([-300, -200, -100, 0, 100])
ax0.set_yticklabels([-300, -200, -100, 0, 100], fontsize=8)
ax0.set_xticks([ 0, 100, 200])
ax0.set_xticklabels([0, 100, 200], fontsize=8)


ax1 = fig1.add_subplot(spec[1])
for j in range(len(data_all[1])):
    ax1.plot([1,2,3], data_all[:, j], c='grey', alpha=0.5)
    
for i in range(3):
    ax1.errorbar([i+1], mean_late_resp[i], late_resp_sd[i], marker= 'o', ms=10, ls='', c=labels[i][1])

#ax1.set_ylim(-45, 30)
ax1.set_xticks([1,2,3])
ax1.set_xticklabels(['Miss', 'Hit', 'FA'])

ax1.plot([1,2], [20, 20], c= 'black', alpha=0.8)
ax1.plot([2,3], [17, 17], c= 'black', alpha=0.8)
ax1.plot([1,3], [25, 25], c= 'black', alpha=0.8)
ax1.text(1.5, 20, s='*')
ax1.text(2.5, 17, s='-')
ax1.text(2, 25, s='*')

#%%
#plot early training days vs late training days

region = 'mPFC'
lower= 50 #in ms
upper = 200

dim =late_days[region][0:3, :, 0, lower*20+200:upper*20+200].shape[1]

colors = ['dodgerblue', 'blueviolet']

data_rs_early = early_days[region][2, :, 0, lower*20+200:upper*20+200].reshape(dim*((lower*20+200)-(upper*20+200)))
data_rs_late = late_days[region][2, :, 0, lower*20+200:upper*20+200].reshape(dim*((lower*20+200)-(upper*20+200)))

mean_early_late = [np.mean(data_rs_early), np.mean(data_rs_late)]
sd_early_late = [np.std(data_rs_early, ddof=1), np.std(data_rs_late, ddof=1)]


data_early = np.mean(early_days[region][2, :, 0, lower*20+200:upper*20+200], axis=1)
data_late = np.mean(late_days[region][2, :, 0, lower*20+200:upper*20+200], axis=1)

data_all = np.array([data_early, data_late])

early_late = stats.wilcoxon(data_all[0], data_all[1])

print(early_late)


fig1 = plt.figure(figsize = (10, 3))
fig1.suptitle(region)
spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[5, 2])


ax0 = fig1.add_subplot(spec[0])


ax0.errorbar(x, np.mean(late_days[region][2,:,0].T, axis=1), np.mean(late_days[region][2,:,1].T, axis=1), alpha=0.03, c=colors[0] )
ax0.plot(x, np.mean(late_days[region][2,:,0].T, axis=1), c=colors[0])
ax0.errorbar(x, np.mean(early_days[region][2,:,0].T, axis=1), np.mean(late_days[region][2,:,1].T, axis=1), alpha=0.03, c=colors[1])
ax0.plot(x, np.mean(early_days[region][2,:,0].T, axis=1), c=colors[1])
 

ax0.vlines(0, -200, 100, color='grey')
ax0.plot(np.linspace(lower, upper, 5), np.ones(5)*(-100), c='black')

ax0.set_ylim(-200, 100)
ax0.set_ylabel('Amplitude (uV)')
ax0.set_xlabel('Time from stimulus (ms)')
ax0.set_yticks([-200, -100, 0, 100])
ax0.set_yticklabels([-200, -100, 0, 100], fontsize=8)
ax0.set_xticks([ 0, 100, 200])
ax0.set_xticklabels([0, 100, 200], fontsize=8)


ax1 = fig1.add_subplot(spec[1])
for j in range(len(data_all[1])):
    ax1.plot([1,2], data_all[:, j], c='grey', alpha=0.5)
    
for i in range(2):
    ax1.errorbar([i+1], mean_early_late[i], sd_early_late[i], marker= 'o', ms=10, ls='', c=colors[i])

#ax1.set_ylim(-45, 30)
ax1.set_xticks([1,2])
ax1.set_xlim(0.7, 2.3)
ax1.set_xticklabels(['Early', 'Late'])

ax1.plot([1,2], [10, 10], c= 'black', alpha=0.8)

ax1.text(1.5, 10, s='-')


#%%
# Linear regression


amplitude = LickTime_data[:, 0]*10**6*(-1)
amplitude = amplitude.reshape(-1,1)
#amplitude = StandardScaler().fit_transform(amplitude)
resp_time = LickTime_data[:, 1]/20+50
resp_time =resp_time.reshape(-1,1)
Licktime = LickTime_data[:,2].reshape(-1,1)
#Licktime = StandardScaler().fit_transform(Licktime)


model1 = LinearRegression().fit(amplitude, Licktime)
r_sq = model1.score(amplitude, Licktime)

model2 = LinearRegression().fit(resp_time, Licktime)
r_sq2 = model2.score(resp_time, Licktime)

x_reg = np.linspace(-1.2, 3.2, 11)
x_reg = x_reg.reshape(-1,1)
amplitude_pred = model1.coef_*x_reg+model1.intercept_
RespTime_pred = model2.coef_*x_reg+model2.intercept_



plt.figure(figsize = (10, 5))
ax1 = plt.subplot(121)
ax1.scatter(Licktime, amplitude, c = LickTime_data[:,3].reshape(-1,1), alpha = 0.2) 
#ax1.plot(x_reg, amplitude_pred, c='black')
ax1.set_ylim(0,1400)
ax1.legend()
#ax1.set_title()
ax1.set_ylabel('Amplitude (uV)')
ax1.set_xlabel('First response time (ms)')


ax2 = plt.subplot(122)
ax2.scatter(Licktime, resp_time, c = LickTime_data[:,3].reshape(-1,1), alpha = 0.2)
#ax1.set_title()
ax2.set_ylabel('Time (ms)')
ax2.set_xlabel('First response time (ms)')



 #%%
Hit_trials = []
FA_trials = []

for i in range(len(LickTime_data)):
    if LickTime_data[i, 3] == 1:
        Hit_trials.append(LickTime_data[i])
    else:
        FA_trials.append(LickTime_data[i])
        
Hit_trials = np.array(Hit_trials)
FA_trials = np.array(FA_trials)



from scipy.stats import gaussian_kde

Hit_dens = gaussian_kde(list(Hit_trials[:,2]))
FA_dens = gaussian_kde(list(FA_trials[:,2]))
xs = np.linspace(200,2000,100)
#Hit_dens.covariance_factor = lambda : .25
#FA_dens.covariance_factor = lambda : .25
#
#Hit_dens._compute_covariance()
#FA_dens._compute_covariance()

fig3= plt.figure()
ax1 = plt.subplot(111)

ax1.hist([Hit_trials[:,2], FA_trials[:,2]], bins=20 , color=['red', 'blue'], label = ['Hit Trials', 'FA trials'], density=True)

ax1.set_xlabel('Response time (ms)')
ax1.set_title('Distribution of response time for Hit and FA trials')
ax1.legend()

#ax1.hist(FA_trials[:,2], bins30, color='blue')

#ax1.plot(xs,FA_dens(xs))
#ax1.plot(xs,Hit_dens(xs))

#%%

#plot relating the amplitude and the time of the max response
Trial_LFP_region = all_data.loc['207_1'].index[2:-1]
region_list = list(regions.keys())


LickTime_data = {}
for l in range(len(Trial_LFP_region)):
    LickTime_data[region_list[l]] = np.array([], dtype=object)
    for i in mice_nbs:
        if np.isnan(all_data.loc[f'{i}_1'][0][Trial_LFP_region[l]][0][0]) == False:
            for j in range(1,11):
                try:
                    for k in range(len(all_data.loc[f'{i}_{j}'][0]['Trial_Counter'])):
                        if np.isnan(all_data.loc[f'{i}_{j}'][0]['Trial_FirstLickTime'][k][0]) == False:
                            a = all_data.loc[f'{i}_{j}'][0][Trial_LFP_region[l]][k]
                            LickTime_data[region_list[l]] = np.append(LickTime_data[region_list[l]], np.array([a, all_data.loc[f'{i}_{j}'][0]['Trial_FirstLickTime'][k][0], all_data.loc[f'{i}_{j}'][0]['Trial_ID'][k][0]]), axis=0)
        
                except:
                    KeyError
    LickTime_data[region_list[l]] = LickTime_data[region_list[l]].reshape(int(LickTime_data[region_list[l]].shape[0]/3),3)

#%%

bins = 20
data = {}
for j in ['Hit', 'FA']:
    data[j] = {}
    for i in region_list:
        if j == 'Hit':
            data[j][i] = np.array([x[0:2] for x in LickTime_data[i] if x[2]==1], dtype=object)
        if j == 'FA':
            data[j][i] = np.array([x[0:2] for x in LickTime_data[i] if x[2]==3], dtype=object)
    
        data[j][i] = np.array(sorted(data[j][i], key=take_second))
        data[j][i] = binArray(data[j][i], axis=0, binstep=bins, binsize=bins)
        data[j][i] = np.array([np.mean(x, axis=0) for x in data[j][i][:,0]])
    
        data[j][i] = [np.array([x[0] for x in data[j][i]])*10**6, data[j][i][:,1]]
        
        max_depol_time = np.array([])
        for k in range(len(data[j][i][0])):
            b = min(data[j][i][0][k, 320:])
            max_depol_time = np.append(max_depol_time, np.where(data[j][i][0][k]==b))
        
        data[j][i].append(max_depol_time)


vmin = -375.5739138921249
vmax = 134.49016984515336

#%%
#raster plot

region = 'mPFC'

x_ticks = np.array([200, 1200, 2200, 3200, 4200])

fig7 = plt.figure(figsize=(12,6), constrained_layout=True)
spec7 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig7, width_ratios=(5,5,1))
fig7.suptitle(region)
ax1 = plt.subplot(121)
ax1.set_title('Hit trials')
ax1.set_xlabel('Time from stimulus (ms)')
ax1.set_ylabel('Trial group')
ax1.imshow(data['Hit'][region][0], aspect='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax))
ax1.set_xticks(x_ticks)
ax1.set_xlim(0,4200)
ax1.set_ylim(len(data['Hit'][region][0]),0)
ax1.set_xticklabels(freq_to_ms(x_ticks).astype(int))
ax1.eventplot(ms_to_freq(data['Hit'][region][1]).reshape(-1,1), color='black')

ax2 = plt.subplot(122)
ax2.set_title('FA trials')
ax2.set_xlabel('Time from stimulus (ms)')
ax2.imshow(data['FA'][region][0], aspect='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax))
ax2.set_xticks(x_ticks)
ax2.set_xlim(0,4200)
ax2.set_ylim(len(data['FA'][region][0]),0)
ax2.set_xticklabels(freq_to_ms(x_ticks).astype(int))
ax2.eventplot(ms_to_freq(data['FA'][region][1]).reshape(-1,1), color='black')


plt.colorbar(ax1.imshow(data['Hit'][region][0], aspect='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax)),fraction=0.08, orientation='horizontal', label='mV')
plt.tight_layout()


#%%

#relate max amplitude to the first lick time


fig8 = plt.figure(figsize=(10, 4))

ax1 = plt.subplot(122)
ax1.scatter(data_Hit[1], freq_to_ms(data['Hit'][region][2]), alpha=0.5, color='green')
ax1.scatter(data_FA[1], freq_to_ms(data['FA'][region][2]), alpha=0.5, color='blue')
ax1.set_xticks(freq_to_ms(x_ticks))
ax1.set_yticks(freq_to_ms(x_ticks))

ax1.set_ylim(0,2000)
ax1.set_xlim(0,2000)