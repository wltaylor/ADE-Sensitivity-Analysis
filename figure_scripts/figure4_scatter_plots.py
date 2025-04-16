#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:59:00 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.style.use('default')

time = 'early'

if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2

metrics = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/metrics_106.csv', index_col=0).to_numpy()

with open('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/btc_data_model106.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
axes = axes.flatten()
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

axes[0].scatter(kd, metrics[:,time_idx])
loess_kd = lowess(metrics[:,time_idx], kd, frac=0.3, return_sorted=True)
axes[0].plot(loess_kd[:, 0], loess_kd[:, 1], color='black', linewidth=2)
axes[0].set_xscale('log')
axes[0].set_ylabel('Early arrival \n(dimensionless time)', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$K_d$', fontsize=16)
axes[0].set_title('(a)', fontweight='bold', fontsize=16, loc='left')

axes[1].scatter(dispersivity, metrics[:,time_idx])
loess_disp = lowess(metrics[:,time_idx], dispersivity, frac=0.3, return_sorted=True)
axes[1].plot(loess_disp[:, 0], loess_disp[:, 1], color='black', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=16, loc='left')
axes[1].set_xlabel(r'$\alpha_i$', fontsize=16)

axes[2].scatter(alpha, metrics[:,time_idx])
loess_alpha = lowess(metrics[:,time_idx], alpha, frac=0.3, return_sorted=True)
axes[2].plot(loess_alpha[:, 0], loess_alpha[:, 1], color='black', linewidth=2)
axes[2].set_xscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=16, loc='left')
axes[2].set_xlabel(r'$\alpha$', fontsize=16)

# axes[3].scatter(decay, metrics[:,time_idx])
# loess_decay = lowess(metrics[:,time_idx], decay, frac=0.3, return_sorted=True)
# axes[3].plot(loess_decay[:, 0], loess_decay[:, 1], color='black', linewidth=2)
# axes[3].set_xscale('log')
# axes[3].set_title('(d)', fontweight='bold', fontsize=16, loc='left')
# axes[3].set_xlabel(r'$\rho_b$', fontsize=16)

plt.tight_layout()
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_full_range_scatter.pdf', format='pdf', bbox_inches='tight')
plt.show()