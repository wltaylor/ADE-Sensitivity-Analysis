#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:42:37 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model
from numba import jit, complex128, float64, int64

# it would be nice if I could use a numba-compatible laplace transform method
# this would speed up my model evaluation x100
# chatGPT says I can write my own version of the Talbot method

@jit(float64[:](complex128[:], float64), nopython=True)
def talbot_contour_points(N, t):
    '''
    Compute Talbot contour points for the inverse Laplace transform
    '''
    laplace_theta = np.linspace(0, np.pi, N)
    s = (N / t) * (0.5 * laplace_theta * (1 / np.tan(laplace_theta)) + 1j)
    return s

def laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    '''Laplace time solution for a Type I boundary condition pulse injection in one dimension
    Returns a concentration value "C" in the laplace domain
    s: laplace frequency variable
    rho_b: bulk density
    D: dispersion
    lamb: first order decay rate constant
    alpha: first order desorption rate constant
    kd: sorption distribution coefficient
    Co: initial concentration (injected, not already present in system)
    v: pore velocity (now a static value between simulations)
    ts: pulse duration
    x: measured concentration location
    L: column length
    '''

    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    
    r1 = 1 / (2 * D) * (v + np.sqrt(v ** 2 + 4 * D * big_theta))
    r2 = 1 / (2 * D) * (v - np.sqrt(v ** 2 + 4 * D * big_theta))
    
    term1_numerator = r2 * np.exp(r2 * L + r1 * x) - r1 * np.exp(r1 * L + r2 * x)
    term1_denominator = r2 * np.exp(r2 * L) - r1 * np.exp(r1 * L)
    
    term1 = term1_numerator / term1_denominator
    
    C = (Co / s) * (1 - np.exp(-ts * s)) * term1
    
    return C

def concentration_102_all_metrics(t, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    '''Converts the laplace values from function laplace_102 to the real time domain
    Returns indexes for early arrival, peak concentration, and late time tailing, and an array of the concentration values
    Indexes are returned in dimensionless time
    '''
    concentration = []
    
    # convert to dimensionless time
    t = t/(L/v)

    for time in t:
        if time == 0:
            conc = 0  # Assuming concentration at t=0 is Co 
        else:
            conc = talbot_contour_points(lambda s: laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x), time)
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


#%%

import numpy as np
from numba import jit, complex128, float64, int64
from numba import types
import matplotlib.pyplot as plt


#@jit(complex128[:](int64, float64), nopython=True)
def talbot_contour_points(N, t):
    '''
    Compute Talbot contour points for the inverse Laplace transform
    N: Number of contour points
    t: Time at which to evaluate the inverse Laplace transform
    Returns an array of complex points on the Talbot contour
    '''
    laplace_theta = np.linspace(1e-2, np.pi, N)
    tan_values = np.tan(laplace_theta)
    print("Tan values near 0:", tan_values[:5])  # First few values near 0

    s = (N / t) * (0.5 * laplace_theta * (1 / np.tan(laplace_theta)) + 1j)
    #print(s)
    return s

#@jit(complex128(complex128, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
def laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x):
    '''Laplace time solution for a Type I boundary condition pulse injection in one dimension'''
    big_theta = s + lamb + (rho_b * alpha * kd * s) / (theta * (s + alpha))
    #print(f"s (real): {np.real(s)}, s (imag): {np.imag(s)}")
    #print(f"big_theta (real): {np.real(big_theta)}, big_theta (imag): {np.imag(big_theta)}")
   
    
    r1 = 1 / (2 * D) * (v + np.sqrt(v ** 2 + 4 * D * big_theta))
    r2 = 1 / (2 * D) * (v - np.sqrt(v ** 2 + 4 * D * big_theta))
    #r1 = np.clip(r1, -1e10, 1e10)
    #r2 = np.clip(r2, -1e10, 1e10)

    term1_numerator = r2 * np.exp(r2 * L + r1 * x) - r1 * np.exp(r1 * L + r2 * x)
    term1_denominator = r2 * np.exp(r2 * L) - r1 * np.exp(r1 * L)
    
    term1 = term1_numerator / term1_denominator

    C = (Co / s) * (1 - np.exp(-ts * s)) * term1
    
    return C

#@jit(types.Tuple((int64, int64, int64, float64[:]))(float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, int64), nopython=False)
def concentration_102_all_metrics(t, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x, N=64):
    '''
    Converts the Laplace values from function laplace_102 to the real time domain using Talbot's method.
    Returns indexes for early arrival, peak concentration, and late time tailing, and an array of concentration values.
    N: Number of contour points for the Talbot method
    '''
    concentration = []
    
    # Convert to dimensionless time
    #t = t / (L / v)

    for time in t:
        if time == 0:
            conc = 0.0  # Assuming concentration at t=0 is Co
        else:
            s_points = talbot_contour_points(N, time)
            real_sum = 0.0
            for s in s_points:
                laplace_result = laplace_102(s, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x)
                if np.isnan(laplace_result):
                    #print(f"Warning: NaN encountered in laplace 102 for s = {s}")
                    continue
                real_sum += np.real(laplace_result * np.exp(s * time))
            conc = (2.0 / N) * real_sum
        concentration.append(conc)

    # Convert to array and normalize
    C_array = np.array(concentration, dtype=np.float64) / Co
    
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

    # Find the index where the concentration first drops to 10% of peak value after the peak
    late_arrival_idx = 0
    for i in range(peak_index, len(C_array)):
        if C_array[i] <= tenth_percentile_value:
            late_arrival_idx = i
            break

    return early_arrival_idx, peak_index, late_arrival_idx, C_array

theta = 1
rho_b = 1.5
D = 0.1
v = 0.01
lamb = 0
alpha = 0
kd = 0
Co = 1
ts = 5
L = 2
x = 2
times = np.linspace(1,500,500)

early, peak, late, concentrations = concentration_102_all_metrics(times, theta, rho_b, D, lamb, alpha, kd, Co, v, ts, L, x, N=64)

# plotting
plt.plot(times, concentrations)
plt.show()





