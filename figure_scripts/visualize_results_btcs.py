#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:01:50 2024

@author: williamtaylor
"""

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#%%
keys = ['dr','ar','dt','at']
colors = sns.color_palette('colorblind', 6)
titles = ['Dispersive Reaction','Advective Reaction','Dispersive Transport','Advective Transport']

fig, axes = plt.subplots(2,2,figsize=(8,6))
axes = axes.flatten()

for i,ax in enumerate(axes):
    # load BTCs
    with open(f'results/btc_data_{keys[i]}.json', 'r') as f:
        btc_data = json.load(f)
        print(f'{keys[i]} btc length {len(btc_data)}')
    # plot curves
    for btc in btc_data:
        times = np.array(btc['times'])
        concentrations = np.array(btc['concentrations'])
        ax.plot(times, concentrations, alpha=0.15, c=colors[i])
        #ax.scatter(times, concentrations, c='black')
        
        # peak = np.max(concentrations)
        # time_at_peak = times[np.argmax(concentrations)]
        # tailing_value = 0.1*peak
        
        # after_peak_mask = times > time_at_peak
        # if np.any(concentrations[after_peak_mask] < tailing_value):
        #     continue
        # else:
        #     print(f'BTC for {keys[i]} did not converge')

    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized C')
    ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(0,3000)
    #ax.set_xlim(0,200)
    ax.set_ylim(10e-7,1)

fig.suptitle('BTCs under different transport scenarios', fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

#%%
fig, ax = plt.subplots(1,1,figsize=(8,8))
with open(f'results/btc_data_dr.json', 'r') as f:
    btc_data = json.load(f)
    # plot curves
for btc in btc_data:
    times = np.array(btc['times'])
    concentrations = np.array(btc['concentrations'])
    ax.plot(times, concentrations, alpha=0.5, c='orange')
    #ax.scatter(times, concentrations, c='black')
    
    peak = np.max(concentrations)
    time_at_peak = times[np.argmax(concentrations)]
    tailing_value = 0.1*peak
    
    after_peak_mask = times > time_at_peak
    if np.any(concentrations[after_peak_mask] < tailing_value):
        continue
    else:
        print(f'BTC did not converge')

ax.set_xlabel('Time')
ax.set_ylabel('Normalized C')
#ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=12)
ax.set_yscale('log')
#ax.set_xlim(0,50)
#ax.set_xlim(0,200)
ax.set_ylim(10e-8,10e-2)

fig.suptitle('BTCs under different transport scenarios', fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

#%%
keys = ['model102','model106']
titles = ['Type I BC BTCs','Type III BC BTCs']

fig, axes = plt.subplots(1,2,figsize=(8,4))
axes = axes.flatten()

for i,ax in enumerate(axes):
    # load BTCs
    with open(f'results/btc_data_{keys[i]}.json', 'r') as f:
        btc_data = json.load(f)

    # plot curves
    for btc in btc_data:
        times = np.array(btc['times'])
        concentrations = np.array(btc['concentrations'])
        ax.plot(times, concentrations, alpha=0.15, c=colors[i+4])
        peak = np.max(concentrations)
        time_at_peak = times[np.argmax(concentrations)]
        tailing_value = peak * 0.1
        
        after_peak_mask = times > time_at_peak
        if np.any(concentrations[after_peak_mask] < tailing_value):
            print(f'BTC for {keys[i]} converged')
        else:
            print(f'BTC for {keys[i]} did NOT converge')

    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized C')
    ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=12)
    #ax.set_xlim(0,200)
    ax.set_ylim(10e-8, 10e-1)
    ax.set_yscale('log')


fig.suptitle('BTCs under different boundary conditions', fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()




#%% long distance

fig, ax = plt.subplots(1,1,figsize=(8,4))
with open(f'results/btc_data_long_model102.json', 'r') as f:
    btc_data = json.load(f)

for btc in btc_data:
    times = np.array(btc['times'])
    concentrations = np.array(btc['concentrations'])
    ax.plot(times, concentrations, alpha=0.15, c='red')

ax.set_yscale('log')
ax.set_xlim(0,10000)
plt.show()

