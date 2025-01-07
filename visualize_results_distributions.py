#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:06:05 2024

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

# import the results metrics
dt = pd.read_csv('results/metrics_ar.csv', index_col=0)

sns.kdeplot(dt['Early'], label='Early')
sns.kdeplot(dt['Peak'], label='Peak')
sns.kdeplot(dt['Late'], label='Late')
plt.legend()
plt.xlim(0,)
plt.show()


#%% quad plot
keys = ['dr','ar','dt','at']
titles = ['Diffusive Reaction','Advective Reaction','Diffusive Transport','Advective Transport']
xlims = [100,100,100,25]
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()

for i,ax in enumerate(axes):
    
    data = pd.read_csv(f'results/metrics_{keys[i]}.csv', index_col=0)
    
    sns.kdeplot(data=data['Early'], ax=ax, c='red')
    sns.rugplot(data=data['Early'], ax=ax, c='red')
    
    sns.kdeplot(data=data['Peak'], ax=ax, c='blue')
    sns.rugplot(data=data['Peak'], ax=ax, c='blue')
    
    sns.kdeplot(data=data['Late'], ax=ax, c='purple')
    sns.rugplot(data=data['Late'], ax=ax, c='purple')
    
    
    #ax.set_xlim(0,xlims[i])
    ax.set_title(titles[i], fontweight='bold')   
    ax.set_xlabel('Time')

legend_early = plt.Line2D([0], [0], color='red', lw=2, label='Early')
legend_peak = plt.Line2D([0], [0], color='blue', lw=2, label='Peak')
legend_late = plt.Line2D([0], [0], color='purple', lw=2, label='Late')
fig.tight_layout()
fig.subplots_adjust(top = 0.88, bottom = 0.1)  # Manually adjust the bottom margin
fig.legend(handles=[legend_early, legend_peak, legend_late], loc='lower center', bbox_to_anchor=(0.52, -0.01), ncol=3)

fig.suptitle('Distribution of Time Metrics by Transport Scenario', fontweight='bold', fontsize=18)

plt.show()

#%% how many of the late time tailing values occur at very late intervals?
position = [1,2,3,4]
colors = ['blue','purple','orange','red']
titles = ['Diffusive \nReaction','Advective \nReaction','Diffusive \nTransport','Advective \nTransport']
fig, ax = plt.subplots(figsize=(8,8))

for i,key in enumerate(keys):
    
    data = pd.read_csv(f'results/metrics_{key}.csv', index_col=0)
    
    ax.boxplot(data['Late'], positions=[position[i]],
               boxprops=dict(color=colors[i]), capprops=dict(color=colors[i]), 
               whiskerprops=dict(color=colors[i]), flierprops=dict(markeredgecolor=colors[i]),
               medianprops=dict(color='black'))
    
ax.set_xticklabels(titles)
ax.set_ylabel('Late time tailing')
fig.suptitle('Late time tailing results for all scenarios', fontweight='bold', fontsize=18)
plt.show()