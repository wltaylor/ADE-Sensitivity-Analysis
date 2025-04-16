#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:45:17 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
import model
from SALib.sample import saltelli
import seaborn as sns
import matplotlib.patches as mpatches
from pandas.plotting import table
plt.style.use('ggplot')
import time
from scipy import special
from scipy import interpolate
from mpmath import invertlaplace
from mpmath import mp, exp
mp.dps = 12

# test a sample of each system class's BTCs to see if they are converging
L = 2
times = np.linspace(0,12000,24000)
#%% dispersion transport
times = np.linspace(0,3000,3000)
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0, 0.0005], # lamb
               [0, 0.0005], # alpha
               [0, 0.0005], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**2)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

diff_trans = params_df[(params_df['Pe'] < 1) & (params_df['Da'] < 1)]
Y_early_dt = np.zeros(diff_trans.shape[0])
Y_peak_dt = np.zeros(diff_trans.shape[0])
Y_late_dt = np.zeros(diff_trans.shape[0])

start = time.perf_counter()

results = {}
for i, X in enumerate(param_values):
    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    results[i] = concentrations_, times_
    print(f'Dispersion transport iteration: {i}')

end = time.perf_counter()
elapsed = end-start
print(f'Time taken to simulate {len(params_df)} samples of dispersion-dominated transport: {elapsed:.2f} seconds')
#%%
plt.figure(figsize=(6,6))
for i,concs in enumerate(results):
    plt.plot(results[i][1], results[i][0], c='gold', alpha=0.5)
plt.ylabel('Concentration (Co/C)')
plt.xlabel('Time (dimensionless)')
plt.title('Dispersion transport dominated BTCs')
plt.xlim(0,40)
plt.show()


#%% dispersion reaction
times = np.linspace(0,6000,3000)
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 0.05], # v
               [0.05, 1], # lamb
               [0.05, 1], # alpha
               [0.05, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**1)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

Y_early_dt = np.zeros(len(params_df))
Y_peak_dt = np.zeros(len(params_df))
Y_late_dt = np.zeros(len(params_df))

start = time.perf_counter()

results = {}
for i, X in enumerate(param_values):
    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    results[i] = concentrations_, times_
    print(f'Dispersion reaction iteration: {i}')

end = time.perf_counter()
elapsed = end-start
print(f'Time taken to simulate {len(params_df)} samples of dispersion-dominated reaction: {elapsed:.2f} seconds')
#%%
plt.figure(figsize=(6,6))
for i,concs in enumerate(results):
    plt.plot(results[i][1], results[i][0], c='blue', alpha=0.5)
plt.ylabel('Concentration (Co/C)')
plt.xlabel('Time (dimensionless)')
plt.title('Dispersion reaction dominated BTCs')
plt.xlim(0,30)
plt.show()

#%% advective transport
times = np.linspace(0,300,1000)
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.1, 1], # v
               [0, 0.05], # lamb
               [0, 0.05], # alpha
               [0, 0.05], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**2)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])

# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

Y_early_dt = np.zeros(len(params_df))
Y_peak_dt = np.zeros(len(params_df))
Y_late_dt = np.zeros(len(params_df))

start = time.perf_counter()

results = {}
for i, X in enumerate(param_values):
    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    results[i] = concentrations_, times_
    print(f'Advective transport iteration: {i}')

end = time.perf_counter()
elapsed = end-start
print(f'Time taken to simulate {len(params_df)} samples of advective-dominated transport: {elapsed:.2f} seconds')

plt.figure(figsize=(6,6))
for i,concs in enumerate(results):
    plt.plot(results[i][1], results[i][0], c='red', alpha=0.5)
plt.ylabel('Concentration (Co/C)')
plt.xlabel('Time (dimensionless)')
plt.title('Advective dominated transport BTCs')
plt.xlim(0,50)
plt.show()

#%% advective reaction
times = np.linspace(0, 250, 1000)
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.01, 0.1], # D
               [0.05, 1], # v
               [0.6, 1], # lamb
               [0.5, 1], # alpha
               [0.5, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**2)

params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd','Co'])


# calculate Peclet and Dahmkoler
params_df['Pe'] = (params_df['v'] * 2) / params_df['D']
params_df['Da'] = (params_df['lamb'] * 2) / params_df['v']

Y_early_dt = np.zeros(len(params_df))
Y_peak_dt = np.zeros(len(params_df))
Y_late_dt = np.zeros(len(params_df))

start = time.perf_counter()

results = {}
for i, X in enumerate(param_values):
    Y_early_dt[i], Y_peak_dt[i], Y_late_dt[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    results[i] = concentrations_, times_
    print(f'Advective reaction iteration: {i}')

end = time.perf_counter()
elapsed = end-start
print(f'Time taken to simulate {len(params_df)} samples of advective-dominated reaction: {elapsed:.2f} seconds')

plt.figure(figsize=(6,6))
for i,concs in enumerate(results):
    plt.plot(results[i][1], results[i][0], c='purple', alpha=0.5)
plt.ylabel('Concentration (Co/C)')
plt.xlabel('Time (dimensionless)')
plt.title('Advective dominated reaction BTCs')
plt.show()