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

keys = ['dr','ar','dt','at']
colors = ['blue','purple','orange','red']
titles = ['Diffusive Reaction','Advective Reaction','Diffusive Transport','Advective Transport']

fig, axes = plt.subplots(2,2,figsize=(8,6))
axes = axes.flatten()

for i,ax in enumerate(axes):
    # load BTCs
    with open(f'results/btc_data_{keys[i]}.json', 'r') as f:
        btc_data = json.load(f)

    # plot curves
    for btc in btc_data:
        times = btc['times']
        concentrations = btc['concentrations']
        ax.plot(times, concentrations, alpha=0.15, c=colors[i])

    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized C')
    ax.set_title(f'{titles[i]}', fontweight='bold', fontsize=12)
    ax.set_xlim(0,200)
    #ax.set_ylim(0,.01)

fig.suptitle('BTCs under different transport scenarios', fontweight='bold', fontsize=18)
plt.tight_layout()
plt.show()

