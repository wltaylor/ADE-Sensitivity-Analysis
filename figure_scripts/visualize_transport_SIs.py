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
import matplotlib.pyplot as plt
import seaborn as sns
import ast
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


#%%
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
greek_names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
name_map = dict(zip(names, greek_names))

offset = 0.2
width = 0.2
indices = ['early','peak','late']
types = ['total','first','second']
names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
colors = sns.color_palette('colorblind', 6)
handles = ['Dispersive Transport','Dispersive Reaction','Advective Transport','Advective Reaction']
fig, ax = plt.subplots(3, 3, figsize=(12, 10))
for j,transport in enumerate(transports):
    bar_data = SIs_dict[transport]
    for i, x in enumerate(indices):
        total = f'total_Si_{x}'
        first = f'first_Si_{x}'
        second = f'second_Si_{x}'
        
        # Convert tuple indices to strings for the second order indices
        second_index = bar_data[second].index
        second_index = second_index.to_list()
        second_index_tuples = [ast.literal_eval(item) if isinstance(item, str) else item for item in second_index]
        second_index_greek = [fr"{name_map[v1]}-{name_map[v2]}" for v1, v2 in second_index_tuples]    
        
        # Create an array of positions for the bar plots
        positions = np.arange(len(names))
        second_positions = np.arange(len(second_index_greek))
        
        ax[i, 0].bar(positions + offset*j, bar_data[total]['ST'], color=colors[j], width=width)
        ax[i, 0].errorbar(positions + offset*j, bar_data[total]['ST'],
                          yerr=bar_data[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)
        
        ax[i, 0].set_title(f'Total {x.capitalize()}')
        ax[i, 0].set_ylabel('Sensitivity Index')
        ax[i, 0].set_xticks(positions + offset * (len(transports) - 1) / 2)
        ax[i, 0].set_xticklabels(greek_names, ha='center', fontweight='bold', fontsize=18)
        ax[i, 0].set_ylim(0, 1.2)
                              
        ax[i, 1].bar(positions + offset*j, bar_data[first]['S1'], color=colors[j], width=width)
        ax[i, 1].errorbar(positions + offset*j, bar_data[first]['S1'],
                          yerr=bar_data[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)
    
        ax[i, 1].set_title(f'First Order {x.capitalize()}')
        ax[i, 1].set_xticks(positions + offset * (len(transports) - 1) / 2)
        ax[i, 1].set_xticklabels(greek_names, ha='center', fontsize=18)
        ax[i, 1].set_ylim(0, 1.2)
    
        ax[i, 2].bar(second_positions + offset*j, bar_data[second]['S2'], color=colors[j], width=width)
        ax[i, 2].errorbar(second_positions + offset*j, bar_data[second]['S2'],
                          yerr=bar_data[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)
    
        ax[i, 2].set_title(f'Second Order {x.capitalize()}')
        ax[i, 2].set_xticks(second_positions + offset / 2)
        ax[i, 2].set_xticklabels(second_index_greek, rotation=45, ha='right')
        ax[i, 2].set_ylim(0, 1.2)

plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j], label=handles[j]) for j in range(len(handles))]

fig.legend(legend_handles, handles, bbox_to_anchor=(0.5,-0.05), ncol=4, loc='lower center')
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.show()    
#%% transport SIs without second order
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
greek_names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']

offset = 0.2
width = 0.2
indices = ['early','peak','late']
types = ['total','first','second']
names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
colors = sns.color_palette('colorblind', 6)
handles = ['Dispersive Transport','Dispersive Reaction','Advective Transport','Advective Reaction']
fig, ax = plt.subplots(3, 2, figsize=(12, 10))
for j,transport in enumerate(transports):
    bar_data = SIs_dict[transport]
    for i, x in enumerate(indices):
        total = f'total_Si_{x}'
        first = f'first_Si_{x}'
             
        # Create an array of positions for the bar plots
        positions = np.arange(len(names))
        
        ax[i, 0].bar(positions + offset*j, bar_data[total]['ST'], color=colors[j], width=width)
        ax[i, 0].errorbar(positions + offset*j, bar_data[total]['ST'],
                          yerr=bar_data[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)
        
        ax[i, 0].set_title(f'Total {x.capitalize()}')
        ax[i, 0].set_ylabel('Sensitivity Index')
        ax[i, 0].set_xticks(positions + offset * (len(transports) - 1) / 2)
        ax[i, 0].set_xticklabels(greek_names, ha='center', fontweight='bold', fontsize=18)
        ax[i, 0].set_ylim(0, 1.3)
                              
        ax[i, 1].bar(positions + offset*j, bar_data[first]['S1'], color=colors[j], width=width)
        ax[i, 1].errorbar(positions + offset*j, bar_data[first]['S1'],
                          yerr=bar_data[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)
    
        ax[i, 1].set_title(f'First Order {x.capitalize()}')
        ax[i, 1].set_xticks(positions + offset * (len(transports) - 1) / 2)
        ax[i, 1].set_xticklabels(greek_names, ha='center', fontsize=18)
        ax[i, 1].set_ylim(0, 1.3)
    

#plt.suptitle('Transport Type Sensitivity Indices', fontweight='bold', fontsize=24)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j], label=handles[j]) for j in range(len(handles))]

fig.legend(legend_handles, handles, bbox_to_anchor=(0.5,-0.05), ncol=4, loc='lower center')
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/transport_comp_without_S2.pdf', format='pdf', bbox_inches='tight')
plt.show()    

#%% transport SIs only second order
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
greek_names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
name_map = dict(zip(names, greek_names))

offset = 0.2
width = 0.2
indices = ['early','peak','late']
types = ['total','first','second']
names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
colors = sns.color_palette('colorblind', 6)
handles = ['Dispersive Transport','Dispersive Reaction','Advective Transport','Advective Reaction']
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
for j,transport in enumerate(transports):
    bar_data = SIs_dict[transport]
    for i, x in enumerate(indices):
      
        second = f'second_Si_{x}'
        
        # Convert tuple indices to strings for the second order indices
        second_index = bar_data[second].index
        second_index = second_index.to_list()
        second_index_tuples = [ast.literal_eval(item) if isinstance(item, str) else item for item in second_index]
        second_index_greek = [fr"{name_map[v1]}-{name_map[v2]}" for v1, v2 in second_index_tuples]    
        
        # Create an array of positions for the bar plots
        second_positions = np.arange(len(second_index_greek))
           
        ax[i].bar(second_positions + offset*j, bar_data[second]['S2'], color=colors[j], width=width)
        #ax[i].errorbar(second_positions + offset*j, bar_data[second]['S2'],
        #                  yerr=bar_data[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)
    
        ax[i].set_title(f'Second Order {x.capitalize()}')
        ax[i].set_xticks(second_positions + offset / 2)
        ax[i].set_xticklabels(second_index_greek, rotation=45, ha='right')
        ax[i].set_ylim(0, 0.3)

#plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j], label=handles[j]) for j in range(len(handles))]

fig.legend(legend_handles, handles, bbox_to_anchor=(0.5,-0.05), ncol=4, loc='lower center')
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/transport_comp_only_S2.pdf', format='pdf', bbox_inches='tight')
plt.show()    

#%%
total_Si_early_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_early_102.csv', index_col=0)
first_Si_early_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_early_102.csv', index_col=0)
second_Si_early_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_early_102.csv', index_col=0)
total_Si_peak_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_peak_102.csv', index_col=0)
first_Si_peak_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_peak_102.csv', index_col=0)
second_Si_peak_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_peak_102.csv', index_col=0)
total_Si_late_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_late_102.csv', index_col=0)
first_Si_late_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_late_102.csv', index_col=0)
second_Si_late_102 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_late_102.csv', index_col=0)

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
names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']


total_Si_early_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_early_106.csv', index_col=0)
first_Si_early_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_early_106.csv', index_col=0)
second_Si_early_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_early_106.csv', index_col=0)
total_Si_peak_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_peak_106.csv', index_col=0)
first_Si_peak_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_peak_106.csv', index_col=0)
second_Si_peak_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_peak_106.csv', index_col=0)
total_Si_late_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/total_Si_late_106.csv', index_col=0)
first_Si_late_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/first_Si_late_106.csv', index_col=0)
second_Si_late_106 = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/second_Si_late_106.csv', index_col=0)

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
names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']

#%% boundary condition comparison, with second order effects
offset = 0.25
width = 0.25
fig, ax = plt.subplots(3, 3, figsize=(16, 12))
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
greek_names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
name_map = dict(zip(names, greek_names))

for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict_106[second].index
    second_index = second_index.to_list()
    second_index_tuples = [ast.literal_eval(item) if isinstance(item, str) else item for item in second_index]
    second_index_greek = [fr"{name_map[v1]}-{name_map[v2]}" for v1, v2 in second_index_tuples]    
    # Create an array of positions for the bar plots
    positions = np.arange(len(names))
    second_positions = np.arange(len(second_index_greek))
    
    ax[i, 0].bar(positions, SIs_dict_102[total]['ST'], color=colors[4], width=width)
    ax[i, 0].errorbar(positions, SIs_dict_102[total]['ST'],
                      yerr=SIs_dict_102[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)
    
    ax[i, 0].bar(positions + offset, SIs_dict_106[total]['ST'], color=colors[5], width=width)
    ax[i, 0].errorbar(positions + offset, SIs_dict_106[total]['ST'],
                      yerr=SIs_dict_106[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticks(positions + offset / 2)
    ax[i, 0].set_xticklabels(greek_names, ha='center', fontsize=16)
    ax[i, 0].set_ylim(0, 1.3)
    
    ax[i, 1].bar(positions, SIs_dict_102[first]['S1'], color=colors[4], width=width)
    ax[i, 1].errorbar(positions, SIs_dict_102[first]['S1'],
                      yerr=SIs_dict_102[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)
    
    ax[i, 1].bar(positions + offset, SIs_dict_106[first]['S1'], color=colors[5], width=width)
    ax[i, 1].errorbar(positions + offset, SIs_dict_106[first]['S1'],
                      yerr=SIs_dict_106[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticks(positions + offset / 2)
    ax[i, 1].set_xticklabels(greek_names, ha='center', fontsize=16)
    ax[i, 1].set_ylim(0, 1.3)

    ax[i, 2].bar(second_positions, SIs_dict_102[second]['S2'], color=colors[4], width=width)
    ax[i, 2].errorbar(second_positions, SIs_dict_102[second]['S2'],
                      yerr=SIs_dict_102[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)
    ax[i, 2].bar(second_positions + offset, SIs_dict_106[second]['S2'], color=colors[5], width=width)
    ax[i, 2].errorbar(second_positions + offset, SIs_dict_106[second]['S2'],
                      yerr=SIs_dict_106[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticks(second_positions + offset / 2)
    ax[i, 2].set_xticklabels(second_index_greek, ha='center', rotation=45)
    ax[i, 2].set_ylim(0, 1.3)

plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
type_1 = plt.Line2D([],[], lw=3, c=colors[4], label='Type I BC')
type_3 = plt.Line2D([],[], lw=3, c=colors[5], label='Type III BC')
fig.legend(handles=[type_1, type_3], ncol=2, loc='upper center', bbox_to_anchor=(0.5,-0.01))
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/bc_comp.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%% boundary condition comparison, without second order effects

offset = 0.25
width = 0.25
fig, ax = plt.subplots(3, 2, figsize=(12, 10))

for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
      
    # Create an array of positions for the bar plots
    positions = np.arange(len(names))
    
    ax[i, 0].bar(positions, SIs_dict_102[total]['ST'], color=colors[4], width=width)
    ax[i, 0].errorbar(positions, SIs_dict_102[total]['ST'],
                      yerr=SIs_dict_102[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)
    
    ax[i, 0].bar(positions + offset, SIs_dict_106[total]['ST'], color=colors[5], width=width)
    ax[i, 0].errorbar(positions + offset, SIs_dict_106[total]['ST'],
                      yerr=SIs_dict_106[total]['ST_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticks(positions + offset / 2)
    ax[i, 0].set_xticklabels(greek_names, rotation=0, ha='center', fontsize=16)
    ax[i, 0].set_ylim(0, 1.3)
    
    ax[i, 1].bar(positions, SIs_dict_102[first]['S1'], color=colors[4], width=width)
    ax[i, 1].errorbar(positions, SIs_dict_102[first]['S1'],
                      yerr=SIs_dict_102[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)
    
    ax[i, 1].bar(positions + offset, SIs_dict_106[first]['S1'], color=colors[5], width=width)
    ax[i, 1].errorbar(positions + offset, SIs_dict_106[first]['S1'],
                      yerr=SIs_dict_106[first]['S1_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticks(positions + offset / 2)
    ax[i, 1].set_xticklabels(greek_names, rotation=0, ha='center', fontsize=16)
    ax[i, 1].set_ylim(0, 1.3)

   
#plt.suptitle('Boundary Condition Sensitivity Indices', fontweight='bold', fontsize=24)
type_1 = plt.Line2D([],[], lw=3, c=colors[4], label='Type I BC')
type_3 = plt.Line2D([],[], lw=3, c=colors[5], label='Type III BC')
fig.legend(handles=[type_1, type_3], ncol=2, loc='upper center', bbox_to_anchor=(0.5,-0.01))
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/bc_comp_without_S2.pdf', format='pdf', bbox_inches='tight')
plt.show()


#%% boundary condition comparison, only second order effects
names= ['theta', 'rho_b','dispersivity','lamb','alpha','kd']
greek_names= [r'$\theta$', r'$\rho_b$',r'$\alpha_i$',r'$\lambda$',r'$\alpha$','$K_d$']
name_map = dict(zip(names, greek_names))

offset = 0.25
width = 0.4
fig, ax = plt.subplots(1, 3, figsize=(16, 5))

for i, x in enumerate(indices):
   
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict_106[second].index
    second_index = second_index.to_list()
    second_index_tuples = [ast.literal_eval(item) if isinstance(item, str) else item for item in second_index]
    second_index_greek = [fr"{name_map[v1]}-{name_map[v2]}" for v1, v2 in second_index_tuples]    
    # Create an array of positions for the bar plots
    positions = np.arange(len(names))
    second_positions = np.arange(len(second_index_greek))
    
    ax[i].bar(second_positions, SIs_dict_102[second]['S2'], color=colors[4], width=width)
    ax[i].errorbar(second_positions, SIs_dict_102[second]['S2'],
                      yerr=SIs_dict_102[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)
    ax[i].bar(second_positions + offset, SIs_dict_106[second]['S2'], color=colors[5], width=width)
    ax[i].errorbar(second_positions + offset, SIs_dict_106[second]['S2'],
                      yerr=SIs_dict_106[second]['S2_conf'], fmt='none', ecolor='black', capsize=2)

    ax[i].set_title(f'{x.capitalize()}', fontweight='bold')
    ax[i].set_xticks(second_positions + offset / 2)
    ax[i].set_xticklabels(second_index_greek, rotation=45, ha='center')
    ax[i].set_ylim(0, 0.5)

    if i == 0:
        ax[i].set_ylabel('Sensitivity Indices', fontweight='bold')
    else:
        ax[i].set_ylabel('')
        ax[i].set_yticklabels('')

plt.suptitle('Second Order Sensitivity Indices', fontweight='bold', fontsize=16)
type_1 = plt.Line2D([],[], lw=3, c=colors[4], label='Type I BC')
type_3 = plt.Line2D([],[], lw=3, c=colors[5], label='Type III BC')
fig.legend(handles=[type_1, type_3], ncol=2, loc='upper center', bbox_to_anchor=(0.5,-0.01))
fig.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/bc_comp_only_S2.pdf', format='pdf', bbox_inches='tight')
plt.show()


















