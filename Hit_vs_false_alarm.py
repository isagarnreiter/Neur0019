# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:26:57 2021

@author: Isabelle
"""
import scipy.io as sc
import pandas as pd
import os.path

# data = sc.loadmat('dataDescription')

data = {}

for i in range(200, 215):
    for j in range(1,13):
        if os.path.isfile(f"./Data_Mouse_PL{i}_sessionNum_{j}.mat") == True:
            mouse_trial = sc.loadmat(f"Data_Mouse_PL{i}_sessionNum_{j}")
            mouse_trial = pd.DataFrame(mouse_trial['tmp'][0])
            data[f'{i}_{j}'] = mouse_trial
        else:
            break


dataf = pd.concat(data)

