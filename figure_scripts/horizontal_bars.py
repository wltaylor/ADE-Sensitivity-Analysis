#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:56:37 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.patches as mpatches
plt.style.use('ggplot')

transports = ['dt','dr','at','ar']
SIs_dict = {}
for transport in transports:
    total_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_early_{transport}.csv', index_col=0)
    first_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_early_{transport}.csv', index_col=0)
    second_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_early_{transport}.csv', index_col=0)
    total_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_peak_{transport}.csv', index_col=0)
    first_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_peak_{transport}.csv', index_col=0)
    second_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_peak_{transport}.csv', index_col=0)
    total_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_late_{transport}.csv', index_col=0)
    first_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_late_{transport}.csv', index_col=0)
    second_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_late_{transport}.csv', index_col=0)

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
    names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
    
#%% total and stacked first/second order
positions = np.arange(1,9,1)

fig, ax = plt.subplots(1,1,figsize=(6,6))
variable = 'kd'
# total order
ax.bar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'])
ax.errorbar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'], 
            yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['kd'], fmt='none', ecolor='black', capsize=2)

# first order
ax.bar(positions[1], SIs_dict['dt']['first_Si_peak']['S1']['kd'], color='skyblue')
ax.errorbar(positions[1], SIs_dict['dt']['first_Si_peak']['S1']['kd'],
            yerr=SIs_dict['dt']['first_Si_peak']['S1_conf']['kd'], fmt='none', ecolor='black', capsize=2)
# second order
second_order_contributions = SIs_dict['dt']['second_Si_peak']['S2']
second_order_conf_ints = SIs_dict['dt']['second_Si_peak']['S2_conf']
second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()

bottom_position = SIs_dict['dt']['first_Si_peak']['S1']['kd']

ax.bar(positions[1], 
       second_order_sum,
       bottom = bottom_position,
       label='Second Order',
       color='lightcoral')
ax.errorbar(positions[1], bottom_position + second_order_sum, yerr=second_order_CI_sum, fmt='none', ecolor='black', capsize=2)
plt.show()

#%% dr
scenario = 'dr'
times = ['early','peak','late']
fig, axes = plt.subplots(1,3, figsize=(12,5))
positions = np.arange(1,7,1)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
axes = axes.flatten()
for j,time in enumerate(times):
    for i,name in enumerate(names):
        
        axes[j].bar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], width=0.3, color='red', edgecolor='black')
        axes[j].errorbar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], 
                    yerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    
        # first order
        axes[j].bar(positions[i]+0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', width=0.3, edgecolor='black')
        #ax.errorbar(positions[i]+0.3, SIs_dict[scenario]['first_Si_peak']['S1'][name],
        #            yerr=SIs_dict[scenario]['first_Si_peak']['S1_conf'][name], fmt='none', ecolor='black', capsize=2)
        # second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
        #second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()
    
        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]
    
        axes[j].bar(positions[i]+0.3, 
               second_order_sum,
               bottom = bottom_position,
               label='Second Order',
               color='skyblue',
               width=0.3,
               edgecolor='black')
        #ax.errorbar(positions[1], bottom_position + second_order_sum, yerr=second_order_CI_sum, fmt='none', ecolor='black', capsize=2)
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')]
    
    axes[j].set_xticks(positions)
    axes[j].set_xticklabels(greek_labels)
    axes[j].set_ylabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_xlabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles = handles, loc='upper left', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')
fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()
#%% dt
scenario = 'dt'
times = ['early','peak','late']
fig, axes = plt.subplots(1,3, figsize=(12,5))
positions = np.arange(1,7,1)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
axes = axes.flatten()
for j,time in enumerate(times):
    for i,name in enumerate(names):
        
        axes[j].bar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], width=0.3, color='red', edgecolor='black')
        axes[j].errorbar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], 
                    yerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    
        # first order
        axes[j].bar(positions[i]+0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', width=0.3, edgecolor='black')
        #ax.errorbar(positions[i]+0.3, SIs_dict[scenario]['first_Si_peak']['S1'][name],
        #            yerr=SIs_dict[scenario]['first_Si_peak']['S1_conf'][name], fmt='none', ecolor='black', capsize=2)
        # second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
        #second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()
    
        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]
    
        axes[j].bar(positions[i]+0.3, 
               second_order_sum,
               bottom = bottom_position,
               label='Second Order',
               color='skyblue',
               width=0.3,
               edgecolor='black')
        #ax.errorbar(positions[1], bottom_position + second_order_sum, yerr=second_order_CI_sum, fmt='none', ecolor='black', capsize=2)
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')]
    
    axes[j].set_xticks(positions)
    axes[j].set_xticklabels(greek_labels)
    axes[j].set_ylabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_xlabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles = handles, loc='upper left', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')
fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% at
scenario = 'at'
times = ['early','peak','late']
fig, axes = plt.subplots(1,3, figsize=(12,5))
positions = np.arange(1,7,1)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
axes = axes.flatten()
for j,time in enumerate(times):
    for i,name in enumerate(names):
        
        axes[j].bar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], width=0.3, color='red', edgecolor='black')
        axes[j].errorbar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], 
                    yerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    
        # first order
        axes[j].bar(positions[i]+0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', width=0.3, edgecolor='black')
        #ax.errorbar(positions[i]+0.3, SIs_dict[scenario]['first_Si_peak']['S1'][name],
        #            yerr=SIs_dict[scenario]['first_Si_peak']['S1_conf'][name], fmt='none', ecolor='black', capsize=2)
        # second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
        #second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()
    
        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]
    
        axes[j].bar(positions[i]+0.3, 
               second_order_sum,
               bottom = bottom_position,
               label='Second Order',
               color='skyblue',
               width=0.3,
               edgecolor='black')
        #ax.errorbar(positions[1], bottom_position + second_order_sum, yerr=second_order_CI_sum, fmt='none', ecolor='black', capsize=2)
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')]
    
    axes[j].set_xticks(positions)
    axes[j].set_xticklabels(greek_labels)
    axes[j].set_ylabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_xlabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles = handles, loc='upper left', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')
fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% ar
scenario = 'ar'
times = ['early','peak','late']
fig, axes = plt.subplots(1,3, figsize=(12,5))
positions = np.arange(1,7,1)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
axes = axes.flatten()
for j,time in enumerate(times):
    for i,name in enumerate(names):
        
        axes[j].bar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], width=0.3, color='red', edgecolor='black')
        axes[j].errorbar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], 
                    yerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    
        # first order
        axes[j].bar(positions[i]+0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', width=0.3, edgecolor='black')
        #ax.errorbar(positions[i]+0.3, SIs_dict[scenario]['first_Si_peak']['S1'][name],
        #            yerr=SIs_dict[scenario]['first_Si_peak']['S1_conf'][name], fmt='none', ecolor='black', capsize=2)
        # second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
        #second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()
    
        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]
    
        axes[j].bar(positions[i]+0.3, 
               second_order_sum,
               bottom = bottom_position,
               label='Second Order',
               color='skyblue',
               width=0.3,
               edgecolor='black')
        #ax.errorbar(positions[1], bottom_position + second_order_sum, yerr=second_order_CI_sum, fmt='none', ecolor='black', capsize=2)
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')]
    
    axes[j].set_xticks(positions)
    axes[j].set_xticklabels(greek_labels)
    axes[j].set_ylabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_xlabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles = handles, loc='upper left', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')
fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% with combined CIs

def calculate_combo_CIs(names, S1_central_values, S1_CIs, S2_central_values, S2_CIs):
    summed_SIs = {}
    yerr_values = {}
    
    # compute the standard errors
    S1_SEs = S1_CIs / 1.96 # 1.96 for a 95% confidence interval
    S2_SEs = S2_CIs / 1.96
    
    # propagate uncertainty
    for name in names:
        relevant_S2 = S2_central_values.loc[S2_central_values.index.map(lambda x: name in x)]
        summed_SIs[name] = S1_central_values[name] + relevant_S2.sum()
        
        SE_sum = np.sqrt(S1_SEs[name]**2 + np.sum(S2_SEs[relevant_S2.index]**2))
        
        # compute new CIs
        yerr_values[name] = 1.96 * SE_sum
    
    return yerr_values

S1_central_values = SIs_dict['at']['first_Si_peak']['S1']
S1_CIs = SIs_dict['at']['first_Si_peak']['S1_conf']
S2_central_values = SIs_dict['at']['second_Si_peak']['S2']
S2_CIs = SIs_dict['at']['second_Si_peak']['S2_conf']
new_CIs = calculate_combo_CIs(names, S1_central_values, S1_CIs, S2_central_values, S2_CIs)

#%% test above function

scenario = 'ar'
times = ['early','peak','late']
fig, axes = plt.subplots(1,3, figsize=(12,5))
positions = np.arange(1,7,1)
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
axes = axes.flatten()
for j,time in enumerate(times):
    for i,name in enumerate(names):
        
        axes[j].bar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], width=0.3, color='red', edgecolor='black')
        axes[j].errorbar(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], 
                    yerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    
        # first order
        axes[j].bar(positions[i]+0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', width=0.3, edgecolor='black')
        #ax.errorbar(positions[i]+0.3, SIs_dict[scenario]['first_Si_peak']['S1'][name],
        #            yerr=SIs_dict[scenario]['first_Si_peak']['S1_conf'][name], fmt='none', ecolor='black', capsize=2)
        # second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
        second_order_CI_sum = calculate_combo_CIs([name], 
                                                  SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                                  SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                                  SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                                  SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'],)
    
        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]
    
        axes[j].bar(positions[i]+0.3, 
               second_order_sum,
               bottom = bottom_position,
               label='Second Order',
               color='skyblue',
               width=0.3,
               edgecolor='black')
        
        
        axes[j].errorbar(positions[i]+0.3, bottom_position + second_order_sum, yerr=second_order_CI_sum[name], fmt='none', ecolor='black', capsize=2)
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')]
    
    axes[j].set_xticks(positions)
    axes[j].set_xticklabels(greek_labels)
    axes[j].set_ylabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_xlabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles = handles, loc='upper left', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')
fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% horizontal bar plot

scenario = 'ar'
times = ['early', 'peak', 'late']
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)  # Share y-axis for alignment
positions = np.arange(1, 7, 1)
names = ['kd','dispersivity','lamb','alpha','rho_b','theta']
names = ['theta','rho_b','alpha','lamb','dispersivity','kd']
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']
axes = axes.flatten()

for j, time in enumerate(times):
    for i, name in enumerate(names):
        # Total order
        axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black')
        axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                         xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)

        # First order
        axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black')

        # Second order
        second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
        second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
        second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
        second_order_CI_sum = calculate_combo_CIs([name], 
                                                  SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                                  SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                                  SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                                  SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'])

        bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

        axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,  # Use left instead of bottom
                     color='skyblue',
                     height=0.3,
                     edgecolor='black')
        
        axes[j].errorbar(bottom_position + second_order_sum, positions[i] - 0.3, 
                         xerr=second_order_CI_sum[name], fmt='none', ecolor='black', capsize=2)

    # Legend handles
    handles = [
        mpatches.Patch(color='red', label='Total Order'),
        mpatches.Patch(color='blue', label='First Order'),
        mpatches.Patch(color='skyblue', label='Second Order')
    ]

    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)
    axes[j].set_xlabel('Sensitivity Indices', fontweight='bold', fontsize=12)
    axes[j].set_ylabel('Parameter', fontweight='bold', fontsize=12)
    axes[j].legend(handles=handles, loc='lower right', edgecolor='black')
    axes[j].set_title(time, fontweight='bold')

fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% just late, AR

scenario = 'ar'
time = 'late'
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
positions = np.arange(1, 7, 1)
names = ['kd','dispersivity','lamb','alpha','rho_b','theta']
names = ['theta','rho_b','alpha','lamb','dispersivity','kd']
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

for i, name in enumerate(names):
    # Total order
    
    if i < 3:
        axes.barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
        axes.errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                      xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
    else:
        axes.barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
        axes.errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                      xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2)
    # First order
    if i < 3:
        axes.barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
    else:
        axes.barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

    # Second order
    second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
    second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
    second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
    second_order_CI_sum = calculate_combo_CIs([name], 
                                              SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                              SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                              SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                              SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'])

    bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

    if i < 3:    
        axes.barh(positions[i] - 0.3, 
                 second_order_sum,
                 left=bottom_position,  # Use left instead of bottom
                 color='skyblue',
                 height=0.3,
                 edgecolor='black',
                 alpha=0.33)
    else:
        axes.barh(positions[i] - 0.3, 
                 second_order_sum,
                 left=bottom_position,  # Use left instead of bottom
                 color='skyblue',
                 height=0.3,
                 edgecolor='black')
    #axes.errorbar(bottom_position + second_order_sum, positions[i] - 0.3, 
    #                 xerr=second_order_CI_sum[name], fmt='none', ecolor='black', capsize=2)

# Legend handles
handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]

axes.axvline(0.2, c='black', linestyle='--')

axes.set_yticks(positions)
axes.set_yticklabels(greek_labels)
axes.set_xlabel('Sensitivity Indices', fontweight='bold', fontsize=12)
axes.set_ylabel('Parameter', fontweight='bold', fontsize=12)
axes.legend(handles=handles, loc='lower right', edgecolor='black')
axes.set_title('Advective Reaction Sensitivity Indices \nLate Time Tailing', fontweight='bold', fontsize=12)

#fig.suptitle(f'Scenario {scenario} Sensitivity Indices', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% automated sorting version
plt.style.use('default')
scenario = 'dt'
time = 'late'
threshold = 0.25
st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
sorted_names = sorted(st_values, key=st_values.get)
sorted_greek_labels = [greek_labels[names.index(name)] for name in sorted_names]
positions = np.arange(1,len(sorted_names) + 1, 1)

fig, axes = plt.subplots(1,1,figsize=(6,4))

for i, name in enumerate(sorted_names):
    # Total order
    if st_values[name] < threshold:    
        axes.barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
        axes.errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                  xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
    else:
        axes.barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
        axes.errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                  xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
    # First order
    if st_values[name] < threshold:
        axes.barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
    else:
        axes.barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

    # Second order
    second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
    second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
    second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
    second_order_CI_sum = calculate_combo_CIs([name], 
                                              SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                              SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                              SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                              SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'])

    bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

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
#axes.set_title(f'Scenario {scenario} Sensitivity Indices \n{time}', fontweight='bold', fontsize=12)

plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_scenario_{scenario}_horizontal_SIs.pdf', format='pdf', bbox_inches='tight')
plt.show()


#%% quad plot of late time tailing SIs for all scenarios
plt.style.use('default')
scenarios = ['dr','ar','dt','at']
titles = ['a. Dispersive Reaction (Pe<1, Da>1)','b. Advective Reaction (Pe>1, Da>1)','c. Dispersive Transport (Pe<1, Da<1)','d. Advective Transport (Pe>1, Da<1)']
time = 'late'
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()
for j,scenario in enumerate(scenarios):
    st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
    threshold = 0.25*np.max(list(st_values.values()))
    for i, name in enumerate(names):
       if st_values[name] < threshold:    
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
       else:
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
       # First order
       if st_values[name] < threshold:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

       # Second order
       second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
       second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
       second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
       second_order_CI_sum = calculate_combo_CIs([name], 
                                                 SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                                 SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                                 SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                                 SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'])

       bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

       if st_values[name] < threshold:  
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=1)
           
    axes[j].axvline(threshold, c='black', linestyle='--')
    axes[j].set_xlim(0,1.2)
    axes[j].set_title(titles[j], fontweight='bold', fontsize=12)
    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
fig.legend(handles=handles, loc='center', edgecolor='black', ncol=3, bbox_to_anchor=(0.52, -0.01))
plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_transport_SIs_quad.pdf', format='pdf', bbox_inches='tight')
plt.show()



#%% new quad plot with updated samples
plt.style.use('default')
colorblind_palette = sns.color_palette('colorblind', 4)
transports = ['dr','ar','dt','at']
SIs_dict = {}
for transport in transports:
    total_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_early_{transport}.csv', index_col=0)
    first_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_early_{transport}.csv', index_col=0)
    second_Si_early = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_early_{transport}.csv', index_col=0)
    total_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_peak_{transport}.csv', index_col=0)
    first_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_peak_{transport}.csv', index_col=0)
    second_Si_peak = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_peak_{transport}.csv', index_col=0)
    total_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/total_Si_late_{transport}.csv', index_col=0)
    first_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/first_Si_late_{transport}.csv', index_col=0)
    second_Si_late = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/second_Si_late_{transport}.csv', index_col=0)

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
    names= ['theta', 'rho_b','alpha','lamb','dispersivity','kd']
    

scenarios = ['dr','ar','dt','at']
titles = ['a. Dispersive Reaction (Pe<1, Da>1)','b. Advective Reaction (Pe>1, Da>1)','c. Dispersive Transport (Pe<1, Da<1)', 'd. Advective Transport (Pe>1, Da<1)']
time = 'late'
fig, axes = plt.subplots(2,2,figsize=(8,8))
axes = axes.flatten()
for j,scenario in enumerate(scenarios):
    st_values = {name: SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name] for name in names}
    threshold = 0.25*np.max(list(st_values.values()))
    for i, name in enumerate(names):
       if st_values[name] < threshold:    
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=0.33)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=0.33)
       else:
           axes[j].barh(positions[i], SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], height=0.3, color='red', edgecolor='black', alpha=1)
           axes[j].errorbar(SIs_dict[scenario]['total_Si_'+str(time)]['ST'][name], positions[i], 
                     xerr=SIs_dict[scenario]['total_Si_'+str(time)]['ST_conf'][name], fmt='none', ecolor='black', capsize=2, alpha=1)
       # First order
       if st_values[name] < threshold:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name], color='blue', height=0.3, edgecolor='black', alpha=1)

       # Second order
       second_order_contributions = SIs_dict[scenario]['second_Si_'+str(time)]['S2']
       second_order_conf_ints = SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf']
       second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: name in x)].sum()
       second_order_CI_sum = calculate_combo_CIs([name], 
                                                 SIs_dict[scenario]['first_Si_'+str(time)]['S1'],
                                                 SIs_dict[scenario]['first_Si_'+str(time)]['S1_conf'],
                                                 SIs_dict[scenario]['second_Si_'+str(time)]['S2'],
                                                 SIs_dict[scenario]['second_Si_'+str(time)]['S2_conf'])

       bottom_position = SIs_dict[scenario]['first_Si_'+str(time)]['S1'][name]

       if st_values[name] < threshold:  
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=0.33)
       else:
           axes[j].barh(positions[i] - 0.3, 
                     second_order_sum,
                     left=bottom_position,
                     color='skyblue',
                     height=0.3,
                     edgecolor='black',
                     alpha=1)
           
    axes[j].axvline(threshold, c='black', linestyle='--')
    axes[j].set_xlim(0,1.2)
    axes[j].set_title(titles[j], fontweight='bold', fontsize=12)
    axes[j].set_yticks(positions)
    axes[j].set_yticklabels(greek_labels)

handles = [
    mpatches.Patch(color='red', label='Total Order'),
    mpatches.Patch(color='blue', label='First Order'),
    mpatches.Patch(color='skyblue', label='Second Order')
]
fig.legend(handles=handles, loc='center', edgecolor='black', ncol=3, bbox_to_anchor=(0.52, -0.01))
plt.tight_layout()
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_transport_SIs_quad.pdf', format='pdf', bbox_inches='tight')
plt.show()













