#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:48:35 2024

@author: williamtaylor
"""

import numpy as np
from scipy import interpolate
import pandas as pd
from mpmath import invertlaplace, mp
mp.dps = 12
import model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

transports = ['dt','dr','at','ar']
SIs_dict = {}
for transport in transports:
    total_Si_early = pd.read_csv(f'results/total_Si_early_{transport}.csv', index_col=0)
    first_Si_early = pd.read_csv(f'results/first_Si_early_{transport}.csv', index_col=0)
    second_Si_early = pd.read_csv(f'results/second_Si_early_{transport}.csv', index_col=0)
    total_Si_peak = pd.read_csv(f'results/total_Si_peak_{transport}.csv', index_col=0)
    first_Si_peak = pd.read_csv(f'results/first_Si_peak_{transport}.csv', index_col=0)
    second_Si_peak = pd.read_csv(f'results/second_Si_peak_{transport}.csv', index_col=0)
    total_Si_late = pd.read_csv(f'results/total_Si_late_{transport}.csv', index_col=0)
    first_Si_late = pd.read_csv(f'results/first_Si_late_{transport}.csv', index_col=0)
    second_Si_late = pd.read_csv(f'results/second_Si_late_{transport}.csv', index_col=0)

    SIs_dict[transport] = {
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
offset = 0.25
width = 0.25
indices = ['early','peak','late']
types = ['total','first','second']
names= ['theta', 'rho_b','D','v','lamb','alpha','kd']
colors = ['red','blue','green','purple']
handles = ['Diffusive Transport','Diffusive Reaction','Advective Transport','Advective Reaction']
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
for j,transport in enumerate(transports):
    bar_data = SIs_dict[transport]
    for i, x in enumerate(indices):
        total = f'total_Si_{x}'
        first = f'first_Si_{x}'
        second = f'second_Si_{x}'
        
        # Convert tuple indices to strings for the second order indices
        second_index = bar_data[second].index
        second_index_str = [str(idx) for idx in second_index]
        
        # Create an array of positions for the bar plots
        positions = np.arange(len(names))
        second_positions = np.arange(len(second_index_str))
        
        ax[i, 0].bar(positions + offset*j, bar_data[total]['ST'], color=colors[j], width=width)
    
        ax[i, 0].set_title(f'Total {x.capitalize()}')
        ax[i, 0].set_ylabel('Sensitivity Index')
        ax[i, 0].set_xticks(positions + offset / 2)
        ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
        ax[i, 0].set_ylim(0, 1)
        
        ax[i, 1].bar(positions + offset*j, bar_data[first]['S1'], color=colors[j], width=width)
    
        ax[i, 1].set_title(f'First Order {x.capitalize()}')
        ax[i, 1].set_xticks(positions + offset / 2)
        ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
        ax[i, 1].set_ylim(0, 1)
    
        ax[i, 2].bar(second_positions + offset*j, bar_data[second]['S2'], color=colors[j], width=width)
    
        ax[i, 2].set_title(f'Second Order {x.capitalize()}')
        ax[i, 2].set_xticks(second_positions + offset / 2)
        ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
        ax[i, 2].set_ylim(0, 1)

plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
fig.legend(handles, bbox_to_anchor=(1.15,0.93), loc='upper right')
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.show()    


#%%
total_Si_early_102 = pd.read_csv('results/total_Si_early_102.csv', index_col=0)
first_Si_early_102 = pd.read_csv('results/first_Si_early_102.csv', index_col=0)
second_Si_early_102 = pd.read_csv('results/second_Si_early_102.csv', index_col=0)
total_Si_peak_102 = pd.read_csv('results/total_Si_peak_102.csv', index_col=0)
first_Si_peak_102 = pd.read_csv('results/first_Si_peak_102.csv', index_col=0)
second_Si_peak_102 = pd.read_csv('results/second_Si_peak_102.csv', index_col=0)
total_Si_late_102 = pd.read_csv('results/total_Si_late_102.csv', index_col=0)
first_Si_late_102 = pd.read_csv('results/first_Si_late_102.csv', index_col=0)
second_Si_late_102 = pd.read_csv('results/second_Si_late_102.csv', index_col=0)

SIs_dict_102 = {
    'total_Si_early': total_Si_early_102,
    'first_Si_early': first_Si_early_102,
    'second_Si_early': second_Si_early_102,
    'total_Si_peak': total_Si_peak_102,
    'first_Si_peak': first_Si_peak_102,
    'second_Si_peak': second_Si_peak_102,
    'total_Si_late': total_Si_late_102,
    'first_Si_late': first_Si_late_102,
    'second_Si_late': second_Si_late_102
    }
indices = ['early','peak','late']
types = ['total','first','second']
names= ['theta', 'rho_b','D','v','lamb','alpha','kd']


fig, ax = plt.subplots(3,3,figsize=(16,12))
for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict_102[second].index
    second_index_str = [str(idx) for idx in second_index]
    
    ax[i, 0].bar(names, SIs_dict_102[total]['ST'])
    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 0].set_ylim(0,1)
    
    ax[i, 1].bar(names, SIs_dict_102[first]['S1'])
    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 1].set_ylim(0,1)

    ax[i, 2].bar(second_index_str, SIs_dict_102[second]['S2'])
    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
    ax[i, 2].set_ylim(0,1)
plt.suptitle('Type I BC SA Results')
fig.tight_layout()  # Adjust layout to prevent overlaping
plt.show()

total_Si_early_106 = pd.read_csv('results/total_Si_early_106.csv', index_col=0)
first_Si_early_106 = pd.read_csv('results/first_Si_early_106.csv', index_col=0)
second_Si_early_106 = pd.read_csv('results/second_Si_early_106.csv', index_col=0)
total_Si_peak_106 = pd.read_csv('results/total_Si_peak_106.csv', index_col=0)
first_Si_peak_106 = pd.read_csv('results/first_Si_peak_106.csv', index_col=0)
second_Si_peak_106 = pd.read_csv('results/second_Si_peak_106.csv', index_col=0)
total_Si_late_106 = pd.read_csv('results/total_Si_late_106.csv', index_col=0)
first_Si_late_106 = pd.read_csv('results/first_Si_late_106.csv', index_col=0)
second_Si_late_106 = pd.read_csv('results/second_Si_late_106.csv', index_col=0)

SIs_dict_106 = {
    'total_Si_early': total_Si_early_106,
    'first_Si_early': first_Si_early_106,
    'second_Si_early': second_Si_early_106,
    'total_Si_peak': total_Si_peak_106,
    'first_Si_peak': first_Si_peak_106,
    'second_Si_peak': second_Si_peak_106,
    'total_Si_late': total_Si_late_106,
    'first_Si_late': first_Si_late_106,
    'second_Si_late': second_Si_late_106
    }
indices = ['early','peak','late']
types = ['total','first','second']
names= ['theta', 'rho_b','D','v','lamb','alpha','kd']


fig, ax = plt.subplots(3,3,figsize=(16,12))
for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict_106[second].index
    second_index_str = [str(idx) for idx in second_index]
    
    ax[i, 0].bar(names, SIs_dict_106[total]['ST'])
    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 0].set_ylim(0,1)
    
    ax[i, 1].bar(names, SIs_dict_106[first]['S1'])
    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 1].set_ylim(0,1)

    ax[i, 2].bar(second_index_str, SIs_dict_106[second]['S2'])
    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
    ax[i, 2].set_ylim(0,1)

plt.suptitle('Type III BC SA Results')
fig.tight_layout()  # Adjust layout to prevent overlaping
plt.show()

#%%

offset = 0.25
width = 0.25
fig, ax = plt.subplots(3, 3, figsize=(16, 12))

for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict_106[second].index
    second_index_str = [str(idx) for idx in second_index]
    
    # Create an array of positions for the bar plots
    positions = np.arange(len(names))
    second_positions = np.arange(len(second_index_str))
    
    ax[i, 0].bar(positions, SIs_dict_106[total]['ST'], color='blue', width=width)
    ax[i, 0].bar(positions + offset, SIs_dict_102[total]['ST'], color='red', width=width)

    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticks(positions + offset / 2)
    ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 0].set_ylim(0, 1)
    
    ax[i, 1].bar(positions, SIs_dict_106[first]['S1'], color='blue', width=width)
    ax[i, 1].bar(positions + offset, SIs_dict_102[first]['S1'], color='red', width=width)

    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticks(positions + offset / 2)
    ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
    ax[i, 1].set_ylim(0, 1)

    ax[i, 2].bar(second_positions, SIs_dict_106[second]['S2'], color='blue', width=width)
    ax[i, 2].bar(second_positions + offset, SIs_dict_102[second]['S2'], color='red', width=width)

    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticks(second_positions + offset / 2)
    ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
    ax[i, 2].set_ylim(0, 1)

plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
plt.legend()
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

#%% try as a circular barplot

# create a helper function to determine rotation and alignment of labels
def get_label_rotation(angle, offset):
    # rotation specified in degrees
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = 'right'
        rotation = rotation+180
    else:
        alignment = 'left'
    return rotation, alignment

# function to add labels to the plot
def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
    padding = 0.2
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 

ANGLES = np.linspace(0,2*np.pi, len(first_Si_early_102), endpoint=False)
VALUES_S1 = first_Si_early_102['S1'].values
VALUES_ST = total_Si_early_102['ST'].values
#LABELS = first_Si_early_102.index
LABELS = ['Water Content','Bulk Density','Dispersion','Pore Velocity','Lambda','Alpha','Sorption Cofficient']

WIDTH = 0.1

# start first bar at 90 deg
OFFSET = np.pi / 2

fig, ax = plt.subplots(figsize=(10,10), subplot_kw={"projection":"polar"})

# specify offset
ax.set_theta_offset(OFFSET)

# set limits for y axis. negative lower bound creates the whole in the middle
ax.set_ylim(-0.2,1)

# remove spines
ax.set_frame_on(False)

ax.xaxis.grid(False)
ax.yaxis.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Add bars
ax.bar(
    ANGLES, VALUES_S1, width=WIDTH, linewidth=2,
    color="red", edgecolor="black"
)
ax.bar(
    ANGLES+1, VALUES_ST, width=WIDTH, linewidth=2,
    color="blue", edgecolor="black"
)

# Add labels
add_labels(ANGLES, VALUES_ST, LABELS, OFFSET, ax)

plt.legend(['S1','ST'],loc='upper right')


























