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
from scipy.signal import butter, lfilter, freqz
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib import gridspec, colors
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import gaussian_kde, probplot
import pylab
import yellowbrick

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

%matplotlib inline
#%%
#load the data from individual matlab files and save them in a dataframe
#each matlab file is named Data_Mouse_PL{i}_sessionNum_{j} with i being the 
#name of the mouse and j being the trial number

data = {} #create empty dictionary to host data 

for i in mice_nbs:
    print(f'Mouse {i}')
    for j in range(1,11):
        if os.path.isfile(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}.mat") == True:
            mouse_trial = sc.loadmat(f"D:/UserFolder/neur0019/Data_Mouse_PL{i}_sessionNum_{j}")
            mouse_trial = pd.DataFrame(mouse_trial['tmp'][0])
            mouse_trial = pd.DataFrame(mouse_trial.loc[0][['Trial_Counter', 'Trial_ID', 'Trial_LFP_wS1', 'Trial_LFP_wS2', 'Trial_LFP_wM1', 'Trial_LFP_dCA1', 'Trial_LFP_mPFC', 'Trial_FirstLickTime']])
            data[f'{i}_{j}'] = mouse_trial
        else:
            break   

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

#%%
#load the data
all_data = decompress_pickle('D:/UserFolder/neur0019/all_data_short.pbz2') 

#%% 

#sort the data to seperate into miss, hit, FA and CR trials, per region and per mouse

mice_nbs = [200, 201, 202, 204, 205, 207, 209, 222, 223, 224, 225]    
regions = {'wS1': {}, 'wS2': {}, 'wM1': {}, 'dCA1': {}, 'mPFC': {}}
region_list = list(regions.keys())

for i in mice_nbs:
    print(f'Mouse {i}')
    for l in region_list:
        regions[l][f'{i}'] = [[], [], [], []]
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

#baseline correction and band pass filter between 0.1 and 100 Hz
fs = 2000
lowcut = 0.1
highcut = 100

for region in region_list:
    for i in mice_nbs: 
        if len(regions[region][str(i)][0]) == 0:
            regions[region].pop(str(i))
        else:
            regions[region][str(i)] = np.array(regions[region][str(i)])
            for k in range(4):
                regions[region][str(i)][k] = np.array(regions[region][str(i)][k])
                for l in range(len(regions[region][str(i)][k])):
                    regions[region][str(i)][k][l] = butter_bandpass_filter(regions[region][str(i)][k][l], lowcut, highcut, fs)
                    regions[region][str(i)][k][l] =  regions[region][str(i)][k][l] - regions[region][str(i)][k][l][100]

#create dict with the mean (+/- sem) LFP, per mouse, per region, per trial type
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
early_days = stat
#%%
late_days = stat
#%%
all_days = stat
#%%
#code to produce figure 2 

region = 'wS2'
lower= -50 #in ms
upper = 600
trial_period = all_days
labels = [['Miss', 'red'], ['Hit', 'green'], ['FA', 'blue'], ['CR', 'orange']]
bin_size= 10*2

p_value_data = trial_period[region][:,:,0,ms_to_freq(lower):ms_to_freq(upper)]
p_value_data = p_value_data.reshape(4, p_value_data.shape[1], int((ms_to_freq(upper)-ms_to_freq(lower))/bin_size), bin_size)
p_value_data = np.mean(p_value_data, axis = 3)
p_values = []
for i in range(p_value_data.shape[2]):
    FA_Hit = stats.wilcoxon(p_value_data[1,:,i], p_value_data[2,:,i])
    FA_CR = stats.wilcoxon(p_value_data[2,:,i], p_value_data[3,:,i])
    p_values.append(np.array([FA_Hit, FA_CR]))

p_values=np.array(p_values)

fig1 = plt.figure(figsize = (10, 4))
fig1.suptitle(region)
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         height_ratios=[5, 1])

ax0 = fig1.add_subplot(spec[0])

x = np.linspace(lower,upper, ms_to_freq(upper)-ms_to_freq(lower))
for i in range(4):
    ax0.errorbar(x, np.mean(trial_period[region][i,:,0,ms_to_freq(lower):ms_to_freq(upper)].T, axis=1), np.mean(trial_period[region][i,:,1,ms_to_freq(lower):ms_to_freq(upper)].T, axis=1), alpha=0.03, c = labels[i][1])
    ax0.plot(x, np.mean(trial_period[region][i,:,0,ms_to_freq(lower):ms_to_freq(upper)].T, axis=1), c = labels[i][1], label = labels[i][0])

ax0.vlines(0, -300,100, color='grey')
ax0.set_ylim(-300, 100)
ax0.set_ylabel('Amplitude (uV)')
ax0.set_xlabel('Time from stimulus (ms)')
ax0.set_yticks([-300, -200, -100, 0, 100])
ax0.set_yticklabels([-300, -200, -100, 0, 100], fontsize=8)
ax0.set_xlim(lower, upper)

ax1 = fig1.add_subplot(spec[1])
ax1.imshow(p_values[:,:,1].T, aspect='auto', norm=colors.LogNorm(vmin=0.009, vmax=1))
ax1.tick_params(bottom=False, labelbottom=False)
ax1.set_yticks([0,1])
ax1.set_yticklabels(['FA-Hit', 'FA-CR'])

plt.tight_layout()

#plt.colorbar(ax1.imshow(p_values[:,:,1].T, aspect='auto', norm=colors.LogNorm(vmin=0.009, vmax=1)), orientation='horizontal', ticks = [0.01, 0.05, 0.1, 1])
#%%
#code to produce Figure 3

region = 'mPFC'
i = 2
c = ['blueviolet', 'dodgerblue']
bin_size= 10*2
lower= -50 #in ms
upper = 600

#extract the p_values across the trial length 
p_value_data = np.array([early_days[region][i,:,0,100:], late_days[region][i,:,0,100:]])
p_value_data = p_value_data.reshape(2, p_value_data.shape[1], int((ms_to_freq(upper)-ms_to_freq(lower))/bin_size), bin_size)
p_value_data = np.mean(p_value_data, axis = 3)
p_values = []
for j in range(p_value_data.shape[2]):
    early_late = stats.wilcoxon(p_value_data[0,:,j], p_value_data[1,:,j])
    p_values.append(np.array(early_late))
p_values = np.array(p_values)


fig1 = plt.figure(figsize = (10, 4))
fig1.suptitle(region)
spec = gridspec.GridSpec(ncols=1, nrows=2,
                         height_ratios=[6, 1])

ax0 = fig1.add_subplot(spec[0])

x = np.linspace(-50, 2000, 4100)

ax0.errorbar(x, np.mean(late_days[region][i,:,0,100:].T, axis=1), np.mean(late_days[region][2,:,1,100:].T, axis=1), alpha=0.03, c=c[1] )
ax0.plot(x, np.mean(late_days[region][i,:,0, 100:].T, axis=1), c=c[1])
ax0.errorbar(x, np.mean(early_days[region][i,:,0, 100:].T, axis=1), np.mean(late_days[region][2,:,1,100:].T, axis=1), alpha=0.03, c=c[0])
ax0.plot(x, np.mean(early_days[region][i,:,0, 100:].T, axis=1), c=c[0])
 
ax0.vlines(0, -200, 130, color='grey')
ax0.set_ylim(-200, 130)
ax0.set_ylabel('Amplitude (uV)')
ax0.set_xlabel('Time from stimulus (ms)')
ax0.set_yticks([-200, -100, 0, 100])
ax0.set_yticklabels([-200, -100, 0, 100], fontsize=8)
ax0.set_xlim(-50,2000)

ax1 = fig1.add_subplot(spec[1])
ax1.imshow(p_values[:,1].reshape(1,-1), aspect='auto', norm=colors.LogNorm(vmin=0.009, vmax=1))
ax1.tick_params(bottom=False, labelbottom=False)
ax1.set_yticks([0])
ax1.tick_params(left=False, labelleft=False)
#ax1.set_yticklabels(['early-late'])

plt.tight_layout()

plt.colorbar(ax1.imshow(p_values[:,1].reshape(1,-1), aspect='auto', norm=colors.LogNorm(vmin=0.009, vmax=1)), orientation='horizontal')

#%%

#extract data to produce figure 4 and 5

Trial_LFP_region = all_data.loc['207_1'].index[2:-1]
region_list = ['wS1', 'wS2', 'wM1', 'dCA1', 'mPFC']
mice_nbs = [200, 201, 202, 204, 205, 207, 209, 222, 223, 224, 225]    

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

#get average LFP for trials with close Licktime and preprocess data for inear regression and raster plot

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
            b = min(data[j][i][0][k, 400:])
            max_depol_time = np.append(max_depol_time, freq_to_ms(np.where(data[j][i][0][k]==b)[0][0]))
        
        data[j][i].append(max_depol_time)



#%%
#raster plots for figure 4 

vmin = -375.5739138921249
vmax = 134.49016984515336

region = 'wS2'
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

plt.colorbar(ax1.imshow(data['Hit'][region][0], aspect='auto', norm=colors.Normalize(vmin=vmin, vmax=vmax)),fraction=0.08, orientation='horizontal', label='uV')
plt.tight_layout()

#%%
# produce plots for figure 5
# and optional linear regression

region = 'wS1'

fig, ax = plt.subplots(1,2, figsize=(10, 5))
fig.suptitle(region)
x=0
conditions = ['Hit','FA']
colors = ['green', 'blue']

for i in range(2):

    ax[i].set_title(conditions[i])
    ax[i].scatter(X, y, color=colors[i], alpha=0.5)
    ax[i].set_ylim(200, 2000)

    if i == 1:
        
        df = pd.DataFrame()
        
        X = data[conditions[i]][region][2]
        y = data[conditions[i]][region][1]
        
        X = np.array(X).reshape(-1,1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9)
        
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        
        pred = regr.predict(X_test)
        
        test_set_rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        test_set_r2 = r2_score(y_test, pred)
        
        print('region: \n', region)
        print(f'{conditions[i]} data nb: \n', len(X))
        print(f'{conditions[i]} Intercept: \n', regr.intercept_)
        print(f'{conditions[i]} Coefficients: \n', regr.coef_)
        print(f'{conditions[i]} rmse: \n', test_set_rmse)
        print(f'{conditions[i]} r2: \n', test_set_r2)
        
        y_plot = regr.predict(np.array([min(X), max(X)]).reshape(-1,1))
        ax[i].plot(np.array([min(X), max(X)]), y_plot, color=colors[i], alpha=0.8)
   
ax[i].set_ylabel('ms')
plt.tight_layout()
    
#stats.probplot(data[conditions[i]][region][2], dist="norm", plot=pylab)

#%% Sanity check 

#Plot distribution of Licktime for Hit and FA trials

Hit_trials = []
FA_trials = []
for i in range(len(LickTime_data[region])):
    if LickTime_data[region][i, 2] == 1:
        Hit_trials.append(LickTime_data[region][i])
    else:
        FA_trials.append(LickTime_data[region][i])
        
Hit_trials = np.array(Hit_trials)
FA_trials = np.array(FA_trials)

fig3= plt.figure()
ax1 = plt.subplot(111)

ax1.hist([Hit_trials[:,1], FA_trials[:,1]], bins=20 , color=['red', 'blue'], label = ['Hit Trials', 'FA trials'], density=True)
ax1.set_xlabel('Response time (ms)')
ax1.set_title('Distribution of response time for Hit and FA trials')
ax1.legend()

#check if there is a trend in the distribution of response time versus training progression
x = np.linspace(0, len(FA_trials[:,1]), len(FA_trials[:,1]))

fig4, ax = plt.subplots(1,1)
ax.scatter(x, FA_trials[:,1], marker = 'o', color = 'black', alpha=0.5)
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Trial Number')

