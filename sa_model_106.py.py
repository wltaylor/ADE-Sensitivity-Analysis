#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:45:55 2024

@author: williamtaylor
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpmath import invertlaplace, mp

mp.dps = 12

def laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2):
    
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

def concentration_106(t, theta, rho_b, D, v, lamb, alpha, kd, Co):
    concentration = [invertlaplace(lambda s: laplace_106(s, theta, rho_b, D, v, lamb, alpha, kd, Co, ts=5, x=2, L=2), time, method='dehoog') for time in t]
    print('transformed')
    return concentration

times = np.linspace(1,30,30)

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

concs = concentration_106(times,theta,rho_b,D,v,lamb,alpha,kd,Co)
plt.plot(times, concs, label = 'v = 0.02 m/min')


plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration vs. Time')
plt.legend()
plt.show()

#%%
problem = {
    'num_vars': 8,
    'names': ['theta', 'rho_b','D','v','lamb','alpha','kd','Co'],
    'bounds': [[0, 1], # theta
               [0.5, 2], # rho_b
               [0.000001, 1], # D
               [0.0001, 1], # v
               [0, 1], # lamb
               [0, 1], # alpha
               [0, 1], # kd
               [0,10]] # Co
}

param_values = saltelli.sample(problem, 2**6)


times = np.linspace(1,30,30)
y = np.array([concentration_106(times, *params) for params in param_values])
Si = [sobol.analyze(problem, Y) for Y in y.T]
#%%
S1s = np.array([s['S1'] for s in Si])
S2s = np.array([s['S2'] for s in Si])
STs = np.array([s['ST'] for s in Si])


for i in range(0,8):
    
    plt.plot(S1s[:,i], label = problem['names'][i])
plt.legend()
plt.show()

for i in range(0,8):
    
    plt.plot(S2s[:,i], label = problem['names'][i])
plt.legend()
plt.show()

for i in range(0,8):
    
    plt.plot(STs[:,i], label = problem['names'][i])
plt.legend()
plt.show()