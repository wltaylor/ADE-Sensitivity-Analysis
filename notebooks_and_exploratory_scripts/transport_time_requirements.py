#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:16:04 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy import interpolate
from numba import njit
from mpmath import invertlaplace
from mpmath import mp, exp
import model
mp.dps = 12


def laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
    '''Laplace time solution for a Type III boundary condition pulse injection in one dimension
    Returns a concentration value "C" in the laplace domain
    s: laplace frequency variable
    rho_b: bulk density
    D: dispersion
    v: pore velocity
    lamb: first order decay rate constant
    alpha: first order desorption rate constant
    kd: sorption distribution coefficient
    Co: initial concentration (injected, not already present in system)
    ts: pulse duration
    x: measured concentration location
    L: column length
    '''

    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    delta = 1/(2*D) * mp.sqrt((v**2 + 4*D*big_theta))
    d = 2 * delta * L
    h = D/v
    sigma = v/(2*D)
    
    r1 = sigma + delta
    r2 = sigma - delta
    
    term1_numerator = r2 * mp.exp(r1 * x - d) - r1 * mp.exp(r2 * x)
    term1_denominator = r2 * (1 - h * r1) * mp.exp(-d) - (1 - h * r2)*r1
    
    term1 = mp.fdiv(term1_numerator, term1_denominator)
    
    C = mp.fdiv(Co, s) * (1 - mp.exp(-ts * s)) * term1
    
    return C

def concentration_106_all_metrics(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L, ts):
    '''Converts the laplace values from function laplace_106 to the real time domain
    Returns indexes for early arrival, peak concentration, and late time tailing, and an array of the concentration values
    Indexes are returned in dimensionless time
    '''
    concentration = []
    
    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co 
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=ts, x=L, L=L), time, method='dehoog')
        concentration.append(conc)
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration first reaches 10% of peak value
    late_arrival_idx = 0
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array

def concentration_106_all_metrics_adaptive(t, theta, rho_b, D, v, lamb, alpha, kd, Co, L):
    

    '''Converts the laplace solution from the function laplace_106 to the real time domain, with an adaptive time step to reduce computation time
    Returns indexes for early arrival, peak concentration, and late time tailing, and arrays of the concentration values and corresponding adaptive times
    Indexes are returned in dimensionless time
    '''
    # t is an input array of time values, the others are scalar parameters
    # initialize concentration and adaptive time lists
    concentration = []
    adaptive_times = []
    default_step = t.max()/len(t)
    current_time = 0
    
    # tolerance limit of step size
    tolerance = 0.01
    
    while current_time < t.max():
        if current_time == 0:
            conc = 0  # deal with time 0 case, if there is already concentration in the system change to that value
        else:
            conc = invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), current_time, method='dehoog')
        concentration.append(conc)
        adaptive_times.append(current_time)
        # check if concentration at current and previous time step changed substantially (> 1%)
        if len(concentration) < 2:
            current_time += default_step
        if len(concentration) > 1 and abs(concentration[-1] - concentration[-2]) > tolerance:
            current_time += default_step
        
        # speed up a lot if it's past the peak
        if len(concentration) > 1 and concentration[-1] / np.max(concentration) < 0.1:
            current_time += default_step * 100
        else:
            current_time += default_step * 1
            
    # Convert to array and normalize
    C_array = np.array(concentration, dtype=float) / Co
    
    # Find peak concentration
    peak_C = np.max(C_array)
    peak_index = np.argmax(C_array)

    # Compute 10% of peak concentration
    tenth_percentile_value = 0.1 * peak_C
    
    # Find the index where the concentration first reaches 10% of peak value
    early_arrival_idx = 0
    for i in range(len(C_array)):
        if C_array[i] >= tenth_percentile_value:
            early_arrival_idx = i
            break

    # Find the index where the concentration first reaches 10% of peak value
    late_arrival_idx = 0
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array, adaptive_times

#%% diffusive controlled transport
theta = 0.5
rho_b = 1.5
D = 0.01
v = 0.001
lamb = 0.005
alpha = 0.5
kd = 0.5
times = np.linspace(0,5000,1000)
L = 2

# run sim
early, peak, late, concs = concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=2, ts=5)
print('done')

early6, peak6, late6, concs6, times6 = concentration_106_all_metrics_adaptive(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=2)

print('done')

#%%
plt.plot(times, concs, c='blue')
plt.scatter(times[early], concs[early], marker='*', c='black', s=100)
plt.scatter(times[peak], concs[peak], marker='*', c='black', s=100)
plt.scatter(times[late], concs[late], marker='*', c='black', s=100)

plt.plot(times6, concs6, c='red')
plt.scatter(times6[early6], concs6[early6], marker='*', c='black', s=100)
plt.scatter(times6[peak6], concs6[peak6], marker='*', c='black', s=100)
plt.scatter(times6[late6], concs6[late6], marker='*', c='black', s=100)

plt.xlabel('Time')
plt.ylabel('Normalized C')
plt.title('Diffusive Controlled Transport')
#plt.ylim(0,.01)
plt.show()

#%% diffusive controlled reaction
theta = 0.5
rho_b = 1.5
D = 0.1
v = 0.001
lamb = 0
alpha = 1
kd = 1
times = np.linspace(0,20000,1000)
L = 2

# run sim
early, peak, late, concs = concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=2, ts=5)
print('done')

#%%
plt.plot(times, concs, c='orange')
plt.scatter(times[early], concs[early], marker='*', c='black', s=100)
plt.scatter(times[peak], concs[peak], marker='*', c='black', s=100)
plt.scatter(times[late], concs[late], marker='*', c='black', s=100)

plt.xlabel('Time')
plt.ylabel('Normalized C')
plt.title('Diffusive Controlled Reaction')
#plt.ylim(0,.0001)
plt.show()

#%% advective controlled transport
theta = 0.5
rho_b = 1.5
D = 0.01
v = 0.1
lamb = 0.05
alpha = 0.05
kd = 0.05
times = np.linspace(0,200,1000)
L = 2

# run sim
early, peak, late, concs = concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=2)

#%%
plt.plot(times, concs, c='red')
plt.scatter(times[early], concs[early], marker='*', c='black', s=100)
plt.scatter(times[peak], concs[peak], marker='*', c='black', s=100)
plt.scatter(times[late], concs[late], marker='*', c='black', s=100)

plt.xlabel('Time')
plt.ylabel('Normalized C')
plt.title('Advective Controlled Transport')
plt.show()


#%% advective controlled reaction
theta = 0.5
rho_b = 1.5
D = 0.1
v = 0.05
lamb = 0.6
alpha = 0.5
kd = 0.5
times = np.linspace(0,200,1000)
L = 2

# run sim
early, peak, late, concs = concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=2)
print('done')

#%%
plt.plot(times, concs, c='purple')
plt.scatter(times[early], concs[early], marker='*', c='black', s=100)
plt.scatter(times[peak], concs[peak], marker='*', c='black', s=100)
plt.scatter(times[late], concs[late], marker='*', c='black', s=100)

plt.xlabel('Time')
plt.ylabel('Normalized C')
plt.title('Advective Controlled Reaction')
plt.ylim(0,.01)
plt.show()


#%% what about when L is much bigger?

theta = 0.5
rho_b = 1.5
D = 0.0001
v = 0.05
lamb = 0
alpha = 0
kd = 0
times = np.linspace(0,200000,1000)
L = 300
ts = 50000
# run sim
early, peak, late, concs = concentration_106_all_metrics(times, theta, rho_b, D, v, lamb, alpha, kd, Co=1, L=L, ts=ts)
print('done')

#%%
plt.plot(times, concs, c='purple')
plt.scatter(times[early], concs[early], marker='*', c='black', s=100)
plt.scatter(times[peak], concs[peak], marker='*', c='black', s=100)
plt.scatter(times[late], concs[late], marker='*', c='black', s=100)

plt.xlabel('Time')
plt.ylabel('Normalized C')
plt.title('Advective Controlled Reaction')
#plt.ylim(0,.01)
plt.show()

