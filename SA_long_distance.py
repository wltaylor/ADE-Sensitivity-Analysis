#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:45:34 2024

@author: williamtaylor
"""
import os
os.chdir('/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis')
import numpy as np
import pandas as pd
from scipy import special
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%% helper functions
def time_discretization(v, num_steps):
    # this is approximately correct
    upper_bound = 50000/v**2
    times = np.linspace(1,upper_bound,num_steps)

    return times

#%% set parameters
problem = {
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [1, 2], # rho_b
               [0.1, 2], # D
               [0.001, 1], # v
               [0, 1], # lamb
               [0, 1], # alpha
               [0, 1]] # kd 
}
ts = 750
x = 300
L = 300
times = np.linspace(0,750000,10000)
param_values = saltelli.sample(problem, 2**8)
params_df = pd.DataFrame(data=param_values,
                         columns=['theta', 'rho_b','D','v','lamb','alpha','kd'])
#%%

Y_early = np.zeros(param_values.shape[0])
Y_peak = np.zeros(param_values.shape[0])
Y_late = np.zeros(param_values.shape[0])

for i, X in enumerate(param_values):
    Y_early[i], Y_peak[i], Y_late[i], concentrations_, times_ = model.concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,ts,x,L)
    print(f'Sobol iteration: {i}')

Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

total_Si_early.to_csv('results/spatial/total_Si_early.csv', index=True)
first_Si_early.to_csv('results/spatial/first_Si_early.csv', index=True)
second_Si_early.to_csv('results/spatial/second_Si_early.csv', index=True)
total_Si_peak.to_csv('results/spatial/total_Si_peak.csv', index=True)
first_Si_peak.to_csv('results/spatial/first_Si_peak.csv', index=True)
second_Si_peak.to_csv('results/spatial/second_Si_peak.csv', index=True)
total_Si_late.to_csv('results/spatial/total_Si_late.csv', index=True)
first_Si_late.to_csv('results/spatial/first_Si_late.csv', index=True)
second_Si_late.to_csv('results/spatial/second_Si_late.csv', index=True)

#%%

total_Si_early = pd.read_csv('results/spatial/total_Si_early.csv', index_col=0)
first_Si_early = pd.read_csv('results/spatial/first_Si_early.csv', index_col=0)
second_Si_early = pd.read_csv('results/spatial/second_Si_early.csv', index_col=0)
total_Si_peak = pd.read_csv('results/spatial/total_Si_peak.csv', index_col=0)
first_Si_peak = pd.read_csv('results/spatial/first_Si_peak.csv', index_col=0)
second_Si_peak = pd.read_csv('results/spatial/second_Si_peak.csv', index_col=0)
total_Si_late = pd.read_csv('results/spatial/total_Si_late.csv', index_col=0)
first_Si_late = pd.read_csv('results/spatial/first_Si_late.csv', index_col=0)
second_Si_late = pd.read_csv('results/spatial/second_Si_late.csv', index_col=0)

SIs_dict = {
    'total_Si_early': total_Si_early,
    'first_Si_early': first_Si_early,
    'second_Si_early': second_Si_early,
    'total_Si_peak': total_Si_peak,
    'first_Si_peak': first_Si_peak,
    'second_Si_peak': second_Si_peak,
    'total_Si_late': total_Si_late,
    'first_Si_late': first_Si_late,
    'second_Si_late': second_Si_late
    }
indices = ['early','peak','late']
types = ['total','first','second']
names= ['theta', 'rho_b','D','v','lamb','alpha','kd']


fig, ax = plt.subplots(3,3,figsize=(16,12))
for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict[second].index
    second_index_str = [str(idx) for idx in second_index]
    
    ax[i, 0].bar(names, SIs_dict[total]['ST'])
    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
    #ax[i, 0].set_ylim(0,1)
    
    ax[i, 1].bar(names, SIs_dict[first]['S1'])
    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
    #ax[i, 1].set_ylim(0,1)

    ax[i, 2].bar(second_index_str, SIs_dict[second]['S2'])
    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
    #ax[i, 2].set_ylim(0,1)
plt.suptitle('Type I BC SA Results')
fig.tight_layout()  # Adjust layout to prevent overlaping
plt.show()