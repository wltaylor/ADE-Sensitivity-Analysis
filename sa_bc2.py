#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:40:53 2024

@author: williamtaylor
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import model
from SALib.sample import saltelli
from SALib.analyze import sobol

# define a dictionary of the model inputs (ranges to be changed based on literature findings)
problem = {
    'num_vars':4,
    'names': ['Initial Concentration','Unit Discharge','Porosity','Retardation'],
    'bounds': [[1000,1500],
               [0.01, 1],
               [0.125,0.5],
               [0.5,4]]
}


param_values = saltelli.sample(problem, 2**8)

# evaluate
t = np.linspace(0,1000,1000)
y = np.array([model.bc2(t, *params) for params in param_values])
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]
#%% just S1s

S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(12,6), constrained_layout = True)
gs = fig.add_gridspec(2,3)

ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,2])

for i, ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.plot(t, S1s[:,i],
            label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]),
            color = 'black')
    ax.set_xlabel('time')
    ax.set_ylabel('First-order Sobol index')
    ax.set_xlim(1,1000)
    ax.set_ylim(0,1)
    ax.set_title(problem["names"][i])
    ax.legend()
    
ax0.plot(t, np.mean(y, axis=0), label = 'Mean', color = 'black')

# in percent
prediction_interval = 95

ax0.fill_between(t,
                 np.percentile(y, 50 - prediction_interval/2., axis = 0),
                 np.percentile(y, 50 + prediction_interval/2., axis = 0),
                 alpha = 0.5, color = 'black',
                 label = f"{prediction_interval} % prediction interval")

#%% try with S2s and ST

S1s = np.array([s['S1'] for s in sobol_indices])
S2s = np.array([s['S2'] for s in sobol_indices])
STs = np.array([s['ST'] for s in sobol_indices])

fig = plt.figure(figsize=(12,6), constrained_layout = True)
gs = fig.add_gridspec(2,3)

ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[0,2])
ax4 = fig.add_subplot(gs[1,2])

for i, ax in enumerate([ax1,ax2,ax3,ax4]):
    ax.plot(t, S1s[:,i],
            label='S1',
            color = 'blue')
    ax.plot(t, S2s[:,i],
            label = 'S2',
            color = 'red')
    ax.plot(t, STs[:,i],
            label = 'ST',
            color = 'black')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Sobol index')
    ax.set_xlim(1,1000)
    ax.set_ylim(0,1)
    ax.set_title(problem["names"][i])
    ax.legend()
    
ax0.plot(t, np.mean(y, axis=0), label = 'Mean', color = 'black')
ax0.set_title('Modeled Concentration')
ax0.set_xlabel('Time (days)')
ax0.set_ylabel('Concentration (mg/L)')

# in percent
prediction_interval = 95

ax0.fill_between(t,
                 np.percentile(y, 50 - prediction_interval/2., axis = 0),
                 np.percentile(y, 50 + prediction_interval/2., axis = 0),
                 alpha = 0.5, color = 'black',
                 label = f"{prediction_interval} % prediction interval")

fig.suptitle('Sobol Sensitivity Analysis - Type 2 Boundary Condition', fontsize = 16, fontweight='bold')



