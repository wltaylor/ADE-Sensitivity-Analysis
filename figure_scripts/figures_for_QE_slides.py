#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:08:51 2024

@author: williamtaylor
"""
import sys
import os

# Get the project root (one level up from the figure_scripts folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to Python’s search path
sys.path.append(project_root)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from model import model
from SALib.sample import saltelli
import seaborn as sns
import matplotlib.patches as mpatches
from pandas.plotting import table
plt.style.use('ggplot')
# create a group of parameters to sample from
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0.001, 1], # v
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**2)

params_df = pd.DataFrame(data=param_values,
                         columns=['Theta', 'Rho_b','D','v','Lamb','Alpha','Kd','Co'])

for param in params_df.columns:
    sns.kdeplot(params_df[param])
    plt.title(param, fontweight='bold', fontsize=18)
    plt.show()
    
#%% fake three metric plot

early = np.random.normal(10,2,1000)
peak = np.random.normal(15,3,1000)
late = np.random.normal(50,10,1000)
sns.kdeplot(early, fill=True, label='Early arrival')
sns.kdeplot(peak, fill=True, label='Peak concentration')
sns.kdeplot(late, fill=True, label='Late time tailing')
plt.legend()
plt.xlabel('Time (minutes)')
plt.ylabel('Density')
plt.show()

#%% variance decomp pie plot

theta = 0.2
rho_b = 0.05
D = 0.3
v = 0.4
lamb = 0.01
alpha = 0.02
kd = 0.10
data = np.array([0.2,0.05,0.3,0.35,0.01,0.02,0.07])
print(np.sum(data))
names = ['theta', 'rho_b','D','v','lamb','alpha','kd']

pie_df = pd.DataFrame(data = data, index=names, columns=['Sensitivity'])
sns.set_theme()
plt.pie(x=data, labels=names, startangle=0)
plt.show()

#%% btc curve comparison

times = np.linspace(1,25000000,100000)

theta = 0.5
rho_b = 1.5
D = 2
v = 0.001
lamb = 0
alpha = 1
kd = 0
Co = 1
ts = 5
L = 2
x = 2

early_102, peak_102, late_102, concs_102 = model.concentration_102_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
early_106, peak_106, late_106, concs_106 = model.concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)


#plt.plot(times, concs_102, label='Type 1 BC')
#plt.scatter(times[early_102], concs_102[early_102], marker='^', c='red', zorder=5, s=50)
#plt.scatter(times[peak_102], concs_102[peak_102], marker='*', c='black', zorder=5, s=50)
#plt.scatter(times[late_102], concs_102[late_102], marker='v', c='green', zorder=5, s=50)

plt.plot(times, concs_106, label='Type 3 BC')
plt.scatter(times[early_106], concs_106[early_106], marker='^', c='red', zorder=5, s=50)
plt.scatter(times[peak_106], concs_106[peak_106], marker='*', c='black', zorder=5, s=50)
plt.scatter(times[late_106], concs_106[late_106], marker='v', c='green', zorder=5, s=50)
plt.legend()
plt.xlabel('Time (dimensionless)')
plt.ylabel('Normalized Concentration (mg/mg)')
plt.show()


#%% sample curves
plt.style.use('ggplot')
times = np.linspace(1,200,100)
for i in range(len(params_df)):
    params = params_df.iloc[i,:]
    _,_,_,concs = model.concentration_106_all_metrics(times, *params, L)
    plt.plot(concs, alpha = 0.25, c = 'red')
    print(i)
plt.xlabel('Time (dimensionless)')
plt.ylabel('Concentration (dimensionless)')
plt.title('BTCs from sampled parameters')


#%%

times = np.linspace(1,20000,1000)

theta = 0.5
rho_b = 1.5
D = 0.83*np.log10(2)**2.414 * v + 1.0*10**-9 * 86400
v = 0.05
lamb = 0
alpha = 1
kd = 0
Co = 1
ts = 5
L = 2
x = 2

early_102, peak_102, late_102, concs_102, times_102 = model.concentration_102_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
early_106, peak_106, late_106, concs_106, times_106 = model.concentration_106_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)


plt.plot(times_102, concs_102, label='Type 1 BC')
plt.scatter(times_102[early_102], concs_102[early_102], marker='^', c='red', zorder=5, s=50)
plt.scatter(times_102[peak_102], concs_102[peak_102], marker='*', c='black', zorder=5, s=50)
plt.scatter(times_102[late_102], concs_102[late_102], marker='v', c='green', zorder=5, s=50)

plt.plot(times_106, concs_106, label='Type 3 BC')
plt.scatter(times_106[early_106], concs_106[early_106], marker='^', c='red', zorder=5, s=50)
plt.scatter(times_106[peak_106], concs_106[peak_106], marker='*', c='black', zorder=5, s=50)
plt.scatter(times_106[late_106], concs_106[late_106], marker='v', c='green', zorder=5, s=50)
plt.legend()
plt.xlabel('Time (dimensionless)')
plt.ylabel('Normalized Concentration (mg/mg)')
plt.show()




