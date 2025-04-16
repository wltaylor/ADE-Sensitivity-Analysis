#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:50:31 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.patches as mpatches
plt.style.use('default')

# total_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_early_106.csv', index_col=0)
# first_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_early_106.csv', index_col=0)
# second_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_early_106.csv', index_col=0)
# total_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_peak_106.csv', index_col=0)
# first_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_peak_106.csv', index_col=0)
# second_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_peak_106.csv', index_col=0)
# total_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_late_106.csv', index_col=0)
# first_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_late_106.csv', index_col=0)
# second_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_late_106.csv', index_col=0)

# SIs_dict = {
#     'total_Si_early': total_Si_early_106,
#     'first_Si_early': first_Si_early_106,
#     'second_Si_early': second_Si_early_106,
#     'total_Si_peak': total_Si_peak_106,
#     'first_Si_peak': first_Si_peak_106,
#     'second_Si_peak': second_Si_peak_106,
#     'total_Si_late': total_Si_late_106,
#     'first_Si_late': first_Si_late_106,
#     'second_Si_late': second_Si_late_106
#     }
# indices = ['early','peak','late']
# types = ['total','first','second']
# names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']

# time = 'early'
# threshold = 0.25 * SIs_dict['total_Si_early']['ST']['kd']
# st_values = {name: SIs_dict['total_Si_'+str(time)]['ST'][name] for name in names}
# sorted_names = sorted(st_values, key=st_values.get)
# greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
# sorted_greek_labels = [greek_labels[names.index(name)] for name in sorted_names]
# positions = np.arange(1,len(sorted_names) + 1, 1)

# fig, axes = plt.subplots(1,1,figsize=(6,4))

# for i, name in enumerate(sorted_names):
#     # Total order
#     if st_values[name] < threshold:    
#         axes.barh(positions[i], SIs_dict['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
#         axes.errorbar(SIs_dict['total_Si_'+str(time)]['ST'][name], positions[i], 
#                   xerr=SIs_dict['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
#     else:
#         axes.barh(positions[i], SIs_dict['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
#         axes.errorbar(SIs_dict['total_Si_'+str(time)]['ST'][name], positions[i], 
#                   xerr=SIs_dict['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
#     # First order
#     if st_values[name] < threshold:
#         axes.barh(positions[i] - 0.3, SIs_dict['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
#     else:
#         axes.barh(positions[i] - 0.3, SIs_dict['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

#     # Second order
#     second_order_contributions = SIs_dict['second_Si_'+str(time)]['S2']
#     second_order_conf_ints = SIs_dict['second_Si_'+str(time)]['S2_conf']
#     second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
#     #second_order_CI_sum = calculate_combo_CIs([name], 
#     #                                          SIs_dict['first_Si_'+str(time)]['S1'],
#     #                                          SIs_dict['first_Si_'+str(time)]['S1_conf'],
#     #                                          SIs_dict['second_Si_'+str(time)]['S2'],
#     #                                          SIs_dict['second_Si_'+str(time)]['S2_conf'])

#     bottom_position = SIs_dict['first_Si_'+str(time)]['S1'][name]

#     if st_values[name] < threshold:  
#         axes.barh(positions[i] - 0.3, 
#                   second_order_sum,
#                   left=bottom_position,
#                   color='skyblue',
#                   height=0.3,
#                   edgecolor='black',
#                   alpha=0.33)
#     else:
#         axes.barh(positions[i] - 0.3, 
#                   second_order_sum,
#                   left=bottom_position,
#                   color='skyblue',
#                   height=0.3,
#                   edgecolor='black',
#                   alpha=1)
    
    
# axes.axvline(threshold, c='black', linestyle='--')

# handles = [
#     mpatches.Patch(color='red', label='Total Order'),
#     mpatches.Patch(color='blue', label='First Order'),
#     mpatches.Patch(color='skyblue', label='Second Order')
# ]
# axes.set_xlim(0,1.3)
# axes.set_yticks(positions)
# axes.set_yticklabels(sorted_greek_labels)
# axes.set_xlabel('Sensitivity Indices', fontweight='bold', fontsize=12)
# axes.set_ylabel('Parameter', fontweight='bold', fontsize=12)
# axes.legend(handles=handles, loc='lower right', edgecolor='black')
# #axes.set_title(f'Sensitivity Indices for time metric: {time}', fontweight='bold', fontsize=12)

# plt.tight_layout()
# #plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_full_range_SIs.pdf', format='pdf', bbox_inches='tight')
# plt.show()


#%% print percentage of first order to second order sensitivity indices
# time = 'early'
# for name in names:
#     second_order_contributions = SIs_dict['second_Si_'+str(time)]['S2']
#     second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
#     #print(f'{name} first order: {SIs_dict['fi]}')
#     ratio = SIs_dict['first_Si_early']['S1'][name]/(second_order_sum+SIs_dict['first_Si_early']['S1'][name])
#     print(f'{name} ratio of first order to second order sensitivity: {ratio}')


#%%

total_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_early_106.csv', index_col=0)
first_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_early_106.csv', index_col=0)
second_Si_early_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_early_106.csv', index_col=0)
total_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_peak_106.csv', index_col=0)
first_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_peak_106.csv', index_col=0)
second_Si_peak_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_peak_106.csv', index_col=0)
total_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_late_106.csv', index_col=0)
first_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_late_106.csv', index_col=0)
second_Si_late_106 = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_late_106.csv', index_col=0)

SIs_dict = {
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
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']

time = 'late'
threshold = 0.25 * SIs_dict['total_Si_early']['ST']['kd']
st_values = {name: SIs_dict['total_Si_'+str(time)]['ST'][name] for name in names}
sorted_names = sorted(st_values, key=st_values.get)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
sorted_greek_labels = [greek_labels[names.index(name)] for name in sorted_names]
positions = np.arange(1,len(sorted_names) + 1, 1)

fig, axes = plt.subplots(1,1,figsize=(6,4))

for i, name in enumerate(sorted_names):
    # Total order
    if st_values[name] < threshold:    
        axes.barh(positions[i], SIs_dict['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
        axes.errorbar(SIs_dict['total_Si_'+str(time)]['ST'][name], positions[i], 
                  xerr=SIs_dict['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
    else:
        axes.barh(positions[i], SIs_dict['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
        axes.errorbar(SIs_dict['total_Si_'+str(time)]['ST'][name], positions[i], 
                  xerr=SIs_dict['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
    # First order
    if st_values[name] < threshold:
        axes.barh(positions[i] - 0.3, SIs_dict['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
    else:
        axes.barh(positions[i] - 0.3, SIs_dict['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

    # Second order
    second_order_contributions = SIs_dict['second_Si_'+str(time)]['S2']
    second_order_conf_ints = SIs_dict['second_Si_'+str(time)]['S2_conf']
    second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
    #second_order_CI_sum = calculate_combo_CIs([name], 
    #                                          SIs_dict['first_Si_'+str(time)]['S1'],
    #                                          SIs_dict['first_Si_'+str(time)]['S1_conf'],
    #                                          SIs_dict['second_Si_'+str(time)]['S2'],
    #                                          SIs_dict['second_Si_'+str(time)]['S2_conf'])

    bottom_position = SIs_dict['first_Si_'+str(time)]['S1'][name]

    if st_values[name] < threshold:  
        axes.barh(positions[i] - 0.3, 
                  second_order_sum,
                  left=bottom_position,
                  color='skyblue',
                  height=0.3,
                  edgecolor='black',
                  alpha=0.33)
    else:
        axes.barh(positions[i] - 0.3, 
                  second_order_sum,
                  left=bottom_position,
                  color='skyblue',
                  height=0.3,
                  edgecolor='black',
                  alpha=1)
    
    
axes.axvline(threshold, c='black', linestyle='--')

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
axes.set_xlim(0,1.3)
axes.set_yticks(positions)
axes.set_yticklabels(sorted_greek_labels)
axes.set_xlabel('Sensitivity Indices', fontweight='bold', fontsize=12)
axes.set_ylabel('Parameter', fontweight='bold', fontsize=12)
axes.legend(handles=handles, loc='lower right', edgecolor='black')
#axes.set_title(f'Sensitivity Indices for time metric: {time}', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_full_range_SIs.pdf', format='pdf', bbox_inches='tight')
plt.show()