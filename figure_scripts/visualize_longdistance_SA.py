#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:54:45 2024

@author: williamtaylor
"""

import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# import sensitivity indices
total_Si_early = pd.read_csv('results/spatial/total_Si_early.csv', index_col=0)
first_Si_early = pd.read_csv('results/spatial/first_Si_early.csv', index_col=0)
second_Si_early = pd.read_csv('results/spatial/second_Si_early.csv', index_col=0)
total_Si_peak = pd.read_csv('results/spatial/total_Si_peak.csv', index_col=0)
first_Si_peak = pd.read_csv('results/spatial/first_Si_peak.csv', index_col=0)
second_Si_peak = pd.read_csv('results/spatial/second_Si_peak.csv', index_col=0)
total_Si_late = pd.read_csv('results/spatial/total_Si_late.csv', index_col=0)
first_Si_late = pd.read_csv('results/spatial/first_Si_late.csv', index_col=0)
second_Si_late = pd.read_csv('results/spatial/second_Si_late.csv', index_col=0)

variables = ['Water Content','Bulk Density','Dispersion','Pore Velocity','First Order\nDecay Constant','First Order\nDesorption\nConstant','Sorption\nDistribution\nCoefficient']
N = len(variables)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
titles = ['Total Order SI', 'First Order SI']
fig, axes = plt.subplots(1,2, figsize=(10,5), subplot_kw={'polar':True})

for i,ax in enumerate(axes):
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(variables, fontsize=7, fontweight='bold')
    ax.tick_params(axis='x', pad=15)
    ax.set_rlabel_position(0)
    #ax.set_yticks(size=7)
    ax.set_ylim(0,1.5)
    ax.set_title(titles[i], fontsize=12, fontweight='bold')

# plot values
# early
values = total_Si_early['ST'].values.flatten().tolist()
values += values[:1]
axes[0].plot(angles, values, linewidth=1, linestyle='solid', c='blue', label='Early')
axes[0].fill(angles, values, c='blue', alpha=0.1)

# peak
values = total_Si_peak['ST'].values.flatten().tolist()
values += values[:1]
axes[0].plot(angles, values, linewidth=1, linestyle='solid', c='red', label='Peak')
axes[0].fill(angles, values, c='red', alpha=0.1)

# late
values = total_Si_late['ST'].values.flatten().tolist()
values += values[:1]
axes[0].plot(angles, values, linewidth=1, linestyle='solid', c='green', label='late')
axes[0].fill(angles, values, c='green', alpha=0.1)

# plot values
# early
values = first_Si_early['S1'].values.flatten().tolist()
values += values[:1]
axes[1].plot(angles, values, linewidth=1, linestyle='solid', c='blue', label='Early')
axes[1].fill(angles, values, c='blue', alpha=0.1)

# peak
values = first_Si_peak['S1'].values.flatten().tolist()
values += values[:1]
axes[1].plot(angles, values, linewidth=1, linestyle='solid', c='red', label='Peak')
axes[1].fill(angles, values, c='red', alpha=0.1)

# late
values = first_Si_late['S1'].values.flatten().tolist()
values += values[:1]
axes[1].plot(angles, values, linewidth=1, linestyle='solid', c='green', label='late')
axes[1].fill(angles, values, c='green', alpha=0.1)


# legend
legend_early = plt.Line2D([0], [0], color='red', lw=2, label='Early')
legend_peak = plt.Line2D([0], [0], color='blue', lw=2, label='Peak')
legend_late = plt.Line2D([0], [0], color='green', lw=2, label='Late')

#fig.tight_layout(rect=[0,0.05,1,1])
fig.legend(handles=[legend_early, legend_peak, legend_late], loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
fig.suptitle('Sensitivity Indices for Long Distance Simulation', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.show()