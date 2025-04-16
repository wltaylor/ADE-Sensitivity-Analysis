#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 08:53:28 2025

@author: williamtaylor
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


total_Si_early = pd.read_csv('results/total_Si_early_long_102.csv', index_col=0)
first_Si_early = pd.read_csv('results/first_Si_early_long_102.csv', index_col=0)
second_Si_early = pd.read_csv('results/second_Si_early_long_102.csv', index_col=0)
total_Si_peak = pd.read_csv('results/total_Si_peak_long_102.csv', index_col=0)
first_Si_peak = pd.read_csv('results/first_Si_peak_long_102.csv', index_col=0)
second_Si_peak = pd.read_csv('results/second_Si_peak_long_102.csv', index_col=0)
total_Si_late = pd.read_csv('results/total_Si_late_long_102.csv', index_col=0)
first_Si_late = pd.read_csv('results/first_Si_late_long_102.csv', index_col=0)
second_Si_late = pd.read_csv('results/second_Si_late_long_102.csv', index_col=0)

SIs_dict = {
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
names= ['theta', 'rho_b','D','lamb','alpha','kd']


fig, ax = plt.subplots(3,3,figsize=(16,12))
for i, x in enumerate(indices):
    total = f'total_Si_{x}'
    first = f'first_Si_{x}'
    second = f'second_Si_{x}'
    
    # Convert tuple indices to strings for the second order indices
    second_index = SIs_dict[second].index
    second_index_str = [str(idx) for idx in second_index]
    
    ax[i, 0].bar(names, SIs_dict[total]['ST'])
    ax[i, 0].errorbar(names, SIs_dict[total]['ST'], yerr=SIs_dict[total]['ST_conf'], fmt='none', color='black', capsize=3)
    ax[i, 0].set_title(f'Total {x.capitalize()}')
    ax[i, 0].set_ylabel('Sensitivity Index')
    ax[i, 0].set_xticklabels(names, rotation=45, ha='right')
    #ax[i, 0].set_ylim(0,1)
    
    ax[i, 1].bar(names, SIs_dict[first]['S1'])
    ax[i, 1].errorbar(names, SIs_dict[first]['S1'], yerr=SIs_dict[first]['S1_conf'], fmt='none', color='black', capsize=3)

    ax[i, 1].set_title(f'First Order {x.capitalize()}')
    ax[i, 1].set_xticklabels(names, rotation=45, ha='right')
    #ax[i, 1].set_ylim(0,1)

    ax[i, 2].bar(second_index_str, SIs_dict[second]['S2'])
    ax[i, 2].errorbar(second_index_str, SIs_dict[second]['S2'], yerr=SIs_dict[second]['S2_conf'], fmt='none', color='black', capsize=3)
    ax[i, 2].set_title(f'Second Order {x.capitalize()}')
    ax[i, 2].set_xticklabels(second_index_str, rotation=45, ha='right')
    #ax[i, 2].set_ylim(0,1)
fig.suptitle('Type I BC SA Results', fontsize=16, fontweight='bold')
fig.tight_layout()  # Adjust layout to prevent overlaping
plt.show()