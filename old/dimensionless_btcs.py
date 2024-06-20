#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:08:13 2024

@author: williamtaylor
"""

import os
os.chdir('/Users/williamtaylor/Documents/GitHub/ADE-Sensitivity-Analysis')

import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpmath import invertlaplace, mp
import model
from SALib.sample import saltelli
from SALib.analyze import sobol
from numba import njit

mp.dps = 12


# Initial parameters
times = np.linspace(0, 3000, 100)  # Time array
theta = 1
rho_b = 1.5
D = 0.05
lamb = 0.5
v = 0.05
alpha = 1
kd = 1
Co = 10
L = 2

dimensionless_times = times/(L/v)
test_peak = model.concentration_102_peak(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
test_early = model.concentration_102_early_arrival(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
test_concentrations = model.concentration_102(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
test_late = model.concentration_102_late_arrival(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)

early_arrival = dimensionless_times[test_early]
peak_time = dimensionless_times[test_peak]
late_time = dimensionless_times[test_late]
plt.plot(dimensionless_times,test_concentrations)
plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='black', zorder=5)
plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='black',zorder=5)
plt.scatter(late_time, test_concentrations[test_late], marker='*', c='black',zorder=5)
plt.show()

#%%
# first perform SA for model 102

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

Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in enumerate(param_values):
    Y_early[i], Y_peak[i], Y_late[i] = model.concentration_102_all_metrics(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7],L)
    print(i)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

# convert all to dfs
total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

#%% plotting

# early
Si_early.plot()
plt.tight_layout()
plt.title('Early')
plt.show()

# peak
Si_peak.plot()
plt.tight_layout()
plt.title('Peak')
plt.show()

# late
Si_late.plot()
plt.tight_layout()
plt.title('Late')
plt.show()