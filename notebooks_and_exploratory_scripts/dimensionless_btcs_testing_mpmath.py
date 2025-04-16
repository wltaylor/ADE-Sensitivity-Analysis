#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:14:56 2024

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
mp.dps = 12
import model
from numba import njit
import cProfile
import pstats

# def laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
#     big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
#     r1 = 1 / (2 * D) * (v + mp.sqrt(v ** 2 + 4 * D * big_theta))
#     r2 = 1 / (2 * D) * (v - mp.sqrt(v ** 2 + 4 * D * big_theta))
    
#     term1_numerator = r2 * mp.exp(r2 * L + r1 * x) - r1 * mp.exp(r1 * L + r2 * x)
#     term1_denominator = r2 * mp.exp(r2 * L) - r1 * mp.exp(r1 * L)
    
#     term1 = mp.fdiv(term1_numerator, term1_denominator)
    
#     C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
#     return C

# def concentration_102_all_metrics(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
#     # Compute concentration for each time t
#     concentration = []
    
#     # convert to dimensionless time
#     t = t/(L/v)

#     for time in t:
#         if time == 0:
#             conc = 0  # deal with time 0 case, if there is already concentration in the system change to that value
#         else:
#             conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog')
#         concentration.append(conc)
#     # Convert to array and normalize
#     C_array = np.array(concentration, dtype=float) / Co
    
#     # Find peak concentration
#     peak_C = np.max(C_array)
#     peak_index = np.argmax(C_array)

#     # Compute 10% of peak concentration
#     tenth_percentile_value = 0.1 * peak_C
    
#     # Find the index where the concentration first reaches 10% of peak value
#     early_arrival_idx = 0
#     for i in range(len(C_array)):
#         if C_array[i] >= tenth_percentile_value:
#             early_arrival_idx = i
#             break

#     # Find the index where the concentration first reaches 10% of peak value
#     late_arrival_idx = 0
#     for i in range(peak_index, len(C_array)):
#         if C_array[i] <= tenth_percentile_value:
#             late_arrival_idx = i
#             break

#     return early_arrival_idx, peak_index, late_arrival_idx, C_array

# def concentration_102_all_metrics_adaptive(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
#     # Compute concentration for each time t with an adaptive time step
#     # t is an input array of time values, the others are scalar parameters
#     # initialize concentration and adaptive time lists
#     concentration = []
#     adaptive_times = []
#     # convert to dimensionless time
#     t = t/(L/v)
#     default_step = t.max()/len(t)
#     current_time = 0
    
#     # tolerance limit of step size
#     tolerance = 0.01
    
#     while current_time < t.max():
#         if current_time == 0:
#             conc = 0  # deal with time 0 case, if there is already concentration in the system change to that value
#         else:
#             conc = invertlaplace(lambda s: laplace_102(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), current_time, method='cohen')
#         concentration.append(conc)
#         adaptive_times.append(current_time)
#         # check if concentration at current and previous time step changed substantially (> 1%)
#         if len(concentration) < 2:
#             current_time += default_step
#         if len(concentration) > 1 and abs(concentration[-1] - concentration[-2]) > tolerance:
#             current_time += default_step
        
#         # speed up a lot if it's past the peak
#         if len(concentration) > 1 and concentration[-1] / np.max(concentration) < 0.1:
#             current_time += default_step * 100
#         else:
#             current_time += default_step * 1.5
        
    
#     # Convert to array and normalize
#     C_array = np.array(concentration, dtype=float) / Co
    
#     # Find peak concentration
#     peak_C = np.max(C_array)
#     peak_index = np.argmax(C_array)

#     # Compute 10% of peak concentration
#     tenth_percentile_value = 0.1 * peak_C
    
#     # Find the index where the concentration first reaches 10% of peak value
#     early_arrival_idx = 0
#     for i in range(len(C_array)):
#         if C_array[i] >= tenth_percentile_value:
#             early_arrival_idx = i
#             break

#     # Find the index where the concentration first reaches 10% of peak value
#     late_arrival_idx = 0
#     for i in range(peak_index, len(C_array)):
#         if C_array[i] <= tenth_percentile_value:
#             late_arrival_idx = i
#             break

#     return early_arrival_idx, peak_index, late_arrival_idx, C_array, adaptive_times
#%%
times = np.linspace(0, 3000, 100)  # Time array
theta = 1
rho_b = 1.5
D = 0.05
lamb = 0.5
v = 0.05
alpha = 1
kd = 1
Co = 1
L = 2
 
dimensionless_times = times/(L/v)
test_early, test_peak, test_late, test_concentrations = model.concentration_102_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
  # original method
early_arrival = dimensionless_times[test_early]
peak_time = dimensionless_times[test_peak]
late_time = dimensionless_times[test_late]

plt.scatter(dimensionless_times,test_concentrations, marker=".",c='blue', alpha = 0.5, label= 'Original evaluation points')
plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='black', zorder=5)
plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='black',zorder=5)
plt.scatter(late_time, test_concentrations[test_late], marker='*', c='black',zorder=5)

# adaptive method
test_early, test_peak, test_late, test_concentrations, adaptive_times = model.concentration_102_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
early_arrival = adaptive_times[test_early]
peak_time = adaptive_times[test_peak]
late_time = adaptive_times[test_late]
plt.scatter(adaptive_times,test_concentrations, marker=".",c='red', alpha = 0.5, label = 'Adaptive evaluation points')
plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='green', zorder=5)
plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='green',zorder=5)
plt.scatter(late_time, test_concentrations[test_late], marker='*', c='green',zorder=5)
plt.legend()
plt.xlabel('Time (dimensionless)')
plt.ylabel('Concentration (dimensionless)')
plt.show()

#%%
times = np.linspace(0, 5000, 500)  # Time array
theta = 1
rho_b = 1.5
D = 0.05
lamb = 0.5
v = 0.05
alpha = 1
kd = 1
Co = 1
L = 2

dimensionless_times = times/(L/v)
test_early, test_peak, test_late, test_concentrations, adaptive_times = model.concentration_102_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
early_arrival = adaptive_times[test_early]
peak_time = adaptive_times[test_peak]
late_time = adaptive_times[test_late]

plt.scatter(adaptive_times,test_concentrations, marker=".",c='blue', alpha = 0.5, label= 'Type I BC')
plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='black', zorder=5)
plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='black',zorder=5)
plt.scatter(late_time, test_concentrations[test_late], marker='*', c='black',zorder=5)

# adaptive method
test_early, test_peak, test_late, test_concentrations, adaptive_times = model.concentration_106_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
early_arrival = adaptive_times[test_early]
peak_time = adaptive_times[test_peak]
late_time = adaptive_times[test_late]
plt.scatter(adaptive_times,test_concentrations, marker=".",c='red', alpha = 0.5, label = 'Type III BC')
plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='green', zorder=5)
plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='green',zorder=5)
plt.scatter(late_time, test_concentrations[test_late], marker='*', c='green',zorder=5)
plt.legend()
plt.xlabel('Time (dimensionless)')
plt.ylabel('Concentration (dimensionless)')
plt.show()


#%%
# def main():
#     # Initial parameters
#     times = np.linspace(0, 100000, 10000)  # Time array
#     theta = 1
#     rho_b = 1.5
#     D = 0.05
#     lamb = 0.5
#     v = 0.05
#     alpha = 1
#     kd = 1
#     Co = 10
#     L = 2
    
#     dimensionless_times = times/(L/v)
#     test_early, test_peak, test_late, test_concentrations = concentration_102_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
    
#     pass

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler)
#     stats.sort_stats('cumulative').print_stats(10)

#%%
# def main():
#     # Initial parameters
#     times = np.linspace(0, 10000, 1000)  # Time array
#     theta = 1
#     rho_b = 1.5
#     D = 0.05
#     lamb = 0.5
#     v = 0.05
#     alpha = 1
#     kd = 1
#     Co = 10
#     L = 2
    
#     dimensionless_times = times/(L/v)
#     test_early, test_peak, test_late, test_concentrations, adaptive_times = concentration_102_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
    
#     pass

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()
#     stats = pstats.Stats(profiler)
#     stats.sort_stats('cumulative').print_stats(10)
#%% slowest scenario?

# times = np.linspace(0, 60000, 12000)  # Time array
# theta = 1
# rho_b = 2
# D = 0.001
# lamb = 0.5
# v = 0.01
# alpha = 1
# kd = 1
# Co = 1
# L = 2
 
# test_early, test_peak, test_late, test_concentrations, adaptive_times = concentration_102_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co, L)
# early_arrival = adaptive_times[test_early]
# peak_time = adaptive_times[test_peak]
# late_time = adaptive_times[test_late]
# plt.scatter(adaptive_times,test_concentrations, marker=".",c='red', alpha = 0.5, label = 'Adaptive evaluation points')
# plt.scatter(early_arrival, test_concentrations[test_early], marker="*",c='green', zorder=5)
# plt.scatter(peak_time, test_concentrations[test_peak], marker='*', c='green',zorder=5)
# plt.scatter(late_time, test_concentrations[test_late], marker='*', c='green',zorder=5)
# plt.legend()
# plt.xlabel('Time (dimensionless)')
# plt.ylabel('Concentration (dimensionless)')
# plt.show()

#%% now implement the above in SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
# first perform SA for model 102

problem = {
    'num_vars': 7,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.001, 2], # D
               [0.01, 1], # v
               [0, 0.5], # lamb
               [0, 1], # alpha
               [0, 1]] # kd
}

param_values = saltelli.sample(problem, 2**8)
times = np.linspace(0,12000,24000)
L = 2
Y_early = np.zeros([param_values.shape[0]])
Y_peak = np.zeros([param_values.shape[0]])
Y_late = np.zeros([param_values.shape[0]])

# evaluate the model at the sampled parameter values, across the same dimensionless time and fixed L
for i,X in enumerate(param_values):
    Y_early[i], Y_peak[i], Y_late[i], concentrations_, times_ = concentration_102_all_metrics_adaptive(times,X[0],X[1],X[2],X[3],X[4],X[5],X[6],1,L)
    print(i)

# apply Sobol method to each set of results    
Si_early = sobol.analyze(problem, Y_early, print_to_console=False)
Si_peak = sobol.analyze(problem, Y_peak, print_to_console=False)
Si_late = sobol.analyze(problem, Y_late, print_to_console=False)

# convert all to dfs
#total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
#total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
#total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()



# plotting

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

#%% better plots
# convert all to dfs
total_Si_early, first_Si_early, second_Si_early = Si_early.to_df()
total_Si_peak, first_Si_peak, second_Si_peak = Si_peak.to_df()
total_Si_late, first_Si_late, second_Si_late = Si_late.to_df()

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

#%%
fig, ax = plt.subplots(3,3,figsize=(16,12))
for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict[second].index
    second_index_str = ['+'.join(map(str, idx)) for idx in second_index]
    
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

fig.tight_layout()  # Adjust layout to prevent overlaping
plt.show()

