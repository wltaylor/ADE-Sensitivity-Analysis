#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:14:02 2024

@author: williamtaylor
"""

import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpmath import invertlaplace, mp
import model

mp.dps = 12
#%%
times = np.linspace(1,80,80)

theta = 0.25
rho_b = 1.5
D = 0.05
v = 0.1
lamb = 0
alpha = 1
kd = 0
Co = 1
ts = 5
L = 2
x = 2

concs_102 = model.concentration_102(times,theta,rho_b,D,v,lamb,alpha,kd,Co) # generates a list of mpf objects
concs_102_float = [float(conc) for conc in concs_102] # convert to floats
concs_102_array = np.array(concs_102_float) # convert to array

concs_106 = model.concentration_106(times,theta,rho_b,D,v,lamb,alpha,kd,Co)
concs_106_float = [float(conc) for conc in concs_106]
concs_106_array = np.array(concs_106_float)
#%%
interpolated_concs_102 = interpolate.interp1d(times, concs_102_array, kind='linear')
interpolated_concs_106 = interpolate.interp1d(times, concs_106_array, kind='linear')
times_dense = np.linspace(times.min(), times.max(), 1000)  # More dense times for interpolation
concs_dense_102 = interpolated_concs_102(times_dense)
concs_dense_106 = interpolated_concs_106(times_dense)

# peak values
peak_102 = np.max(concs_102_array) # 10th percentile
tenth_percentile_value_102 = 0.1*peak_102

peak_106 = np.max(concs_106_array)
tenth_percentile_value_106 = 0.1*peak_106

peak_index_102 = np.argmax(concs_dense_102)
peak_index_106 = np.argmax(concs_dense_106)

# early arrival
mask_increasing_102 = np.r_[True, concs_dense_102[1:] > concs_dense_102[:-1]] & (np.arange(len(concs_dense_102)) <= peak_index_102)
mask_increasing_106 = np.r_[True, concs_dense_106[1:] > concs_dense_106[:-1]] & (np.arange(len(concs_dense_106)) <= peak_index_106)

early_idx_102 = np.argmax(concs_dense_102[mask_increasing_102] >= tenth_percentile_value_102)
early_idx_106 = np.argmax(concs_dense_106[mask_increasing_106] >= tenth_percentile_value_106)

time_10th_percentile_102 = times_dense[mask_increasing_102][early_idx_102]
time_10th_percentile_106 = times_dense[mask_increasing_106][early_idx_106]

# late time tailing
mask_decreasing_102 = np.r_[True, concs_dense_102[1:] < concs_dense_102[:-1]] & (np.arange(len(concs_dense_102)) >= peak_index_102)
mask_decreasing_106 = np.r_[True, concs_dense_106[1:] < concs_dense_106[:-1]] & (np.arange(len(concs_dense_106)) >= peak_index_106)

second_idx_102 = np.argmax(concs_dense_102[mask_decreasing_102] <= tenth_percentile_value_102)
second_idx_106 = np.argmax(concs_dense_106[mask_decreasing_106] <= tenth_percentile_value_106)

late_10th_percentile_102 = times_dense[mask_decreasing_102][second_idx_102]
late_10th_percentile_106 = times_dense[mask_decreasing_106][second_idx_106]

plt.plot(times_dense, concs_dense_102, label = 'Type 1 BC')
plt.plot(times_dense, concs_dense_106, label = 'Type 3 BC')
plt.scatter(time_10th_percentile_102, tenth_percentile_value_102, color='red', zorder=5, marker = "^")
plt.scatter(time_10th_percentile_106, tenth_percentile_value_106, color='red', zorder=5, marker = "^")
plt.scatter(times_dense[peak_index_102], peak_102, color='black', zorder = 10, marker = "*", s = 50)
plt.scatter(times_dense[peak_index_106], peak_106, color='black', zorder = 10, marker = "*", s = 50)
plt.scatter(late_10th_percentile_102, tenth_percentile_value_102, color='green', zorder=5, marker = "v")
plt.scatter(late_10th_percentile_106, tenth_percentile_value_106, color='green', zorder=5, marker = "v")

early_marker = mlines.Line2D([], [], color='red', marker='^', linestyle='None',
                             markersize=8, label='Early 10th Percentile')
peak_marker = mlines.Line2D([], [], color='black', marker='*', linestyle='None',
                            markersize=8, label='Peak Concentration')
late_marker = mlines.Line2D([], [], color='green', marker='v', linestyle='None',
                            markersize=8, label='Late 10th Percentile')
line1, = plt.plot([], [], color='blue', label='Type 1 BC')
line2, = plt.plot([], [], color='orange', label='Type 3 BC')

handles = [line1, line2, early_marker, peak_marker, late_marker]

# Create the legend with the custom handles
plt.legend(handles=handles, loc='best')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (mg/m)')
plt.title('Boundary Condition Affect on BTC (Fig 3.20 Goltz)')
plt.show()

#%% SA comparison
from SALib.sample import saltelli
from SALib.analyze import sobol

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

param_values = saltelli.sample(problem, 2**7)


times = np.linspace(1,50,50)
y_102 = np.array([model.concentration_102(times, *params) for params in param_values])
Si_102 = [sobol.analyze(problem, Y) for Y in y_102.T]

y_106 = np.array([model.concentration_106(times, *params) for params in param_values])
Si_106 = [sobol.analyze(problem, Y) for Y in y_106.T]

#%% plotting

S1s_102 = np.array([s['S1'] for s in Si_102])
S2s_102 = np.array([s['S2'] for s in Si_102])
STs_102 = np.array([s['ST'] for s in Si_102])

S1s_106 = np.array([s['S1'] for s in Si_106])
S2s_106 = np.array([s['S2'] for s in Si_106])
STs_106 = np.array([s['ST'] for s in Si_106])

plt.figure(figsize=(9,6))

# S1
plt.subplot(1,2,1)
for i in range(0,8):
    plt.plot(S1s_102[:,i], label = problem['names'][i])
plt.legend()
plt.ylim(0,1)
plt.xlabel('Time (min)')
plt.ylabel('First Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1,2,2)
for i in range(0,8):
    plt.plot(S1s_106[:,i], label = problem['names'][i])
plt.legend()
plt.ylim(0,1)
plt.xlabel('Time (min)')
plt.ylabel('First Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('First Order SI', fontsize=18)
plt.tight_layout()
plt.show()

# S2
plt.figure(figsize=(9,6))
plt.subplot(1,2,1)
for i in range(0,8):
    plt.plot(S2s_102[:,i], label = problem['names'][i])
#plt.legend()
plt.ylim(0,1)
plt.xlabel('Time (min)')
plt.ylabel('Second Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1,2,2)
for i in range(0,8):
    plt.plot(S2s_106[:,i], label = problem['names'][i])
#plt.legend()
plt.ylim(0,1)
plt.xlabel('Time (min)')
plt.ylabel('Second Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('Second Order SI', fontsize=18)
plt.tight_layout()
plt.show()

# ST
plt.figure(figsize=(9,6))

plt.subplot(1,2,1)
for i in range(0,8):
    plt.plot(STs_102[:,i], label = problem['names'][i])
plt.legend()
plt.ylim(0,1.5)
plt.xlabel('Time (min)')
plt.ylabel('Total Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1,2,2)
for i in range(0,8):
    plt.plot(STs_106[:,i], label = problem['names'][i])
plt.legend()
plt.ylim(0,1.5)
plt.xlabel('Time (min)')
plt.ylabel('Total Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('Total Order SI', fontsize=18)
plt.tight_layout()
plt.show()
#%%

# Define a color palette with distinct colors
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# Plotting S1
plt.figure(figsize=(9, 6))
plt.subplot(1, 2, 1)
for i in range(8):
    plt.plot(S1s_102[:, i], label=problem['names'][i], color=color_palette[i])
plt.legend()
plt.ylim(0, 1)
plt.xlabel('Time (min)')
plt.ylabel('First Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1, 2, 2)
for i in range(8):
    plt.plot(S1s_106[:, i], label=problem['names'][i], color=color_palette[i])
plt.legend()
plt.ylim(0, 1)
plt.xlabel('Time (min)')
plt.ylabel('First Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('First Order SI', fontsize=18)
plt.tight_layout()
plt.show()

# Plotting S2
plt.figure(figsize=(9, 6))
plt.subplot(1, 2, 1)
for i in range(8):
    plt.plot(S2s_102[:, i], label=problem['names'][i], color=color_palette[i])
plt.ylim(0, 1)
plt.xlabel('Time (min)')
plt.ylabel('Second Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1, 2, 2)
for i in range(8):
    plt.plot(S2s_106[:, i], label=problem['names'][i], color=color_palette[i])
plt.ylim(0, 1)
plt.xlabel('Time (min)')
plt.ylabel('Second Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('Second Order SI', fontsize=18)
plt.tight_layout()
plt.show()

# Plotting ST
plt.figure(figsize=(9, 6))
plt.subplot(1, 2, 1)
for i in range(8):
    plt.plot(STs_102[:, i], label=problem['names'][i], color=color_palette[i])
plt.legend()
plt.ylim(0, 1.5)
plt.xlabel('Time (min)')
plt.ylabel('Total Order Sobol Sensitivity')
plt.title('Type 1 BC')

plt.subplot(1, 2, 2)
for i in range(8):
    plt.plot(STs_106[:, i], label=problem['names'][i], color=color_palette[i])
plt.legend()
plt.ylim(0, 1.5)
plt.xlabel('Time (min)')
plt.ylabel('Total Order Sobol Sensitivity')
plt.title('Type 3 BC')
plt.suptitle('Total Order SI', fontsize=18)
plt.tight_layout()
plt.show()