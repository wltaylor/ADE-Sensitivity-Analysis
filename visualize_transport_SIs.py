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

