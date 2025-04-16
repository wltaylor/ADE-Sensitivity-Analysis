#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:56:41 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
plt.style.use('ggplot')

with open('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_dt.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Load metrics
metrics = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_dt.csv', index_col=0).to_numpy()

# Scatter plot configuration
names = [theta, bulk, dispersivity, decay, alpha, kd]  # Use actual arrays
titles = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']
colors = sns.color_palette('colorblind')  # Define color scheme

#%%
# scatter plots of parameters vs time to peak concentration
fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Adjust figsize for clarity
axes = axes.flatten()

# Plot each parameter
for i, param in enumerate(names):
    axes[i].scatter(param, metrics[:, 1], c=colors[0], alpha=0.5)
    axes[i].set_title(titles[i], fontweight='bold')
    #if i > 1:
    #    axes[i].set_xscale('log')
    
    ax2 = axes[i].twinx()
    sns.kdeplot(param, ax=ax2, color='black', linewidth=1.5, label='Density')
    
axes[0].set_ylabel('Dimensionless Time', fontweight='bold')
axes[3].set_ylabel('Dimensionless Time', fontweight='bold')  

fig.suptitle('Parameter value influence on time to peak C - DT Scenario', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

#%% create a horizontal barchart of sensitivity indices

# import data
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

#%%

import numpy as np
import matplotlib.pyplot as plt

positions = np.arange(1, 9, 1)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
variable = 'kd'

# Total order
total_si = SIs_dict['ar']['total_Si_peak']['ST']['kd']
total_ci = SIs_dict['ar']['total_Si_peak']['ST_conf']['kd']

ax.bar(positions[0], total_si, color='tomato', label='Total Order')

# Plot confidence interval as a shaded region
ax.bar(positions[0], total_ci, bottom=total_si, color='tomato', alpha=0.4)

# First order
first_si = SIs_dict['ar']['first_Si_peak']['S1']['kd']
first_ci = SIs_dict['ar']['first_Si_peak']['S1_conf']['kd']

ax.bar(positions[1], first_si, color='skyblue', label='First Order')
ax.bar(positions[1], first_ci, bottom=first_si, color='skyblue', alpha=0.4)

# Second order
second_order_contributions = SIs_dict['ar']['second_Si_peak']['S2']
second_order_conf_ints = SIs_dict['ar']['second_Si_peak']['S2_conf']

second_order_sum = second_order_contributions.loc[second_order_contributions.index.map(lambda x: variable in x)].sum()
second_order_CI_sum = second_order_conf_ints.loc[second_order_conf_ints.index.map(lambda x: variable in x)].sum()

ax.bar(positions[1], second_order_sum, bottom=first_si, color='lightcoral', label='Second Order')
ax.bar(positions[1], second_order_CI_sum, bottom=first_si + second_order_sum, color='lightcoral', alpha=0.4)

# Labels and Display
ax.set_xticks([positions[0], positions[1]])
ax.set_xticklabels(["Total Order", "First + Second Order"])
ax.legend()
plt.show()



#%% just total order, rotated horizontally
colors = sns.color_palette('dark')
positions = np.arange(0,9,1)
fig, ax = plt.subplots(1,5,figsize=(12,4))

ax[0].bar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'], color=colors[0])
ax[0].errorbar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'], 
            yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['kd'], fmt='none', ecolor='black', capsize=2)

ax[0].bar(positions[1], SIs_dict['dt']['total_Si_peak']['ST']['dispersivity'], color=colors[1])
ax[0].errorbar(positions[1], SIs_dict['dt']['total_Si_peak']['ST']['dispersivity'], 
            yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['dispersivity'], fmt='none', ecolor='black', capsize=2)

ax[0].bar(positions[2], SIs_dict['dt']['total_Si_peak']['ST']['alpha'], color=colors[2])
ax[0].errorbar(positions[2], SIs_dict['dt']['total_Si_peak']['ST']['alpha'], 
            yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['alpha'], fmt='none', ecolor='black', capsize=2)

ax[0].bar(positions[3], SIs_dict['dt']['total_Si_peak']['ST']['theta'], color=colors[3])
ax[0].errorbar(positions[3], SIs_dict['dt']['total_Si_peak']['ST']['theta'], 
            yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['theta'], fmt='none', ecolor='black', capsize=2)

ax[0].set_xticks([positions[0], positions[1], positions[2], positions[3]])  # Set tick positions
ax[0].set_xticklabels(['$K_d$',r'$\alpha_i$',r'$\alpha$', r'$\theta$'])
ax[0].set_ylabel('Total Order SI')
ax[0].set_title('a.', fontweight='bold')

ax[1].scatter(kd, metrics[:,1], color=colors[0])
ax[1].set_xscale('log')
ax[1].set_title('b.', fontweight='bold')
ax[1].set_xlabel('$K_d$')
ax[1].set_ylabel('Time to peak C')

ax[2].scatter(dispersivity, metrics[:,1], color=colors[1])
ax[2].set_xscale('log')
ax[2].set_title('c.', fontweight='bold')
ax[2].set_xlabel(r'$\alpha_i$')
ax[2].set_ylabel('Time to peak C')

ax[3].scatter(alpha, metrics[:,1], color=colors[2])
ax[3].set_xlabel(r'$\alpha$')
ax[3].set_ylabel('Time to peak C')
ax[3].set_xscale('log')
ax[3].set_title('d.', fontweight='bold')

ax[4].scatter(theta, metrics[:,1], color=colors[3])
ax[4].set_xlabel(r'$\theta$')
ax[4].set_ylabel('Time to peak C')
ax[4].set_title('e.', fontweight='bold')

fig.suptitle('Parameters influencing time to peak concentration - dispersive transport', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

colors = sns.color_palette('crest')
positions = np.arange(0, 4, 1)  # Only 4 bars, so positions should match

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.4)

# Bar Chart (Left Panel)
ax0 = fig.add_subplot(gs[:, 0])  # Takes up both rows

ax0.bar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'], color=colors[0])
ax0.errorbar(positions[0], SIs_dict['dt']['total_Si_peak']['ST']['kd'], 
             yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['kd'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[1], SIs_dict['dt']['total_Si_peak']['ST']['dispersivity'], color=colors[1])
ax0.errorbar(positions[1], SIs_dict['dt']['total_Si_peak']['ST']['dispersivity'], 
             yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['dispersivity'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[2], SIs_dict['dt']['total_Si_peak']['ST']['alpha'], color=colors[2])
ax0.errorbar(positions[2], SIs_dict['dt']['total_Si_peak']['ST']['alpha'], 
             yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['alpha'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[3], SIs_dict['dt']['total_Si_peak']['ST']['theta'], color=colors[3])
ax0.errorbar(positions[3], SIs_dict['dt']['total_Si_peak']['ST']['theta'], 
             yerr=SIs_dict['dt']['total_Si_peak']['ST_conf']['theta'], fmt='none', ecolor='black', capsize=2)

ax0.set_xticks(positions)
ax0.set_xticklabels(['$K_d$', r'$\alpha_i$', r'$\alpha$', r'$\theta$'])
ax0.set_ylabel('Total Order SI')
ax0.set_title('a.', fontweight='bold')

# Scatter Plots (Right Panel, 2x2 Grid)
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

ax1.scatter(kd, metrics[:, 1], color=colors[0])
ax1.set_xscale('log')
ax1.set_title('b.', fontweight='bold')
ax1.set_xlabel('$K_d$')
ax1.set_ylabel('Time to peak C')

ax2.scatter(dispersivity, metrics[:, 1], color=colors[1])
ax2.set_xscale('log')
ax2.set_title('c.', fontweight='bold')
ax2.set_xlabel(r'$\alpha_i$')
ax2.set_ylabel('Time to peak C')

ax3.scatter(alpha, metrics[:, 1], color=colors[2])
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel('Time to peak C')
ax3.set_title('d.', fontweight='bold')
#ax3.set_xscale('log')

ax4.scatter(theta, metrics[:, 1], color=colors[3])
ax4.set_xlabel(r'$\theta$')
ax4.set_ylabel('Time to peak C')
ax4.set_title('e.', fontweight='bold')

# Add overall title
#fig.suptitle('Parameters influencing time to peak concentration - dispersive transport', fontsize=14, fontweight='bold')

plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{transport}_most_important_factors.pdf', format='pdf', bbox_inches='tight')

plt.show()

#%% late time tailing

colors = sns.color_palette('crest')
positions = np.arange(0, 4, 1)  # Only 4 bars, so positions should match

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.4)

# Bar Chart (Left Panel)
ax0 = fig.add_subplot(gs[:, 0])  # Takes up both rows

ax0.bar(positions[0], SIs_dict['dt']['total_Si_late']['ST']['kd'], color=colors[0])
ax0.errorbar(positions[0], SIs_dict['dt']['total_Si_late']['ST']['kd'], 
             yerr=SIs_dict['dt']['total_Si_late']['ST_conf']['kd'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[1], SIs_dict['dt']['total_Si_late']['ST']['dispersivity'], color=colors[1])
ax0.errorbar(positions[1], SIs_dict['dt']['total_Si_late']['ST']['dispersivity'], 
             yerr=SIs_dict['dt']['total_Si_late']['ST_conf']['dispersivity'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[2], SIs_dict['dt']['total_Si_late']['ST']['alpha'], color=colors[2])
ax0.errorbar(positions[2], SIs_dict['dt']['total_Si_late']['ST']['alpha'], 
             yerr=SIs_dict['dt']['total_Si_late']['ST_conf']['alpha'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[3], SIs_dict['dt']['total_Si_late']['ST']['theta'], color=colors[3])
ax0.errorbar(positions[3], SIs_dict['dt']['total_Si_late']['ST']['theta'], 
             yerr=SIs_dict['dt']['total_Si_late']['ST_conf']['theta'], fmt='none', ecolor='black', capsize=2)

ax0.set_xticks(positions)
ax0.set_xticklabels(['$K_d$', r'$\alpha_i$', r'$\alpha$', r'$\theta$'])
ax0.set_ylabel('Total Order SI')
ax0.set_title('a.', fontweight='bold')

# Scatter Plots (Right Panel, 2x2 Grid)
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

ax1.scatter(kd, metrics[:, 2], color=colors[0])
ax1.set_xscale('log')
ax1.set_title('b.', fontweight='bold')
ax1.set_xlabel('$K_d$')
ax1.set_ylabel('Time to peak C')

ax2.scatter(dispersivity, metrics[:, 2], color=colors[1])
ax2.set_xscale('log')
ax2.set_title('c.', fontweight='bold')
ax2.set_xlabel(r'$\alpha_i$')
ax2.set_ylabel('Time to peak C')

ax3.scatter(alpha, metrics[:, 2], color=colors[2])
ax3.set_xlabel(r'$\alpha$')
ax3.set_ylabel('Time to peak C')
ax3.set_title('d.', fontweight='bold')

ax4.scatter(theta, metrics[:, 2], color=colors[3])
ax4.set_xlabel(r'$\theta$')
ax4.set_ylabel('Time to peak C')
ax4.set_title('e.', fontweight='bold')

# Add overall title
#fig.suptitle('Parameters influencing time to peak concentration - dispersive transport', fontsize=14, fontweight='bold')

plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{transport}_most_important_factors.pdf', format='pdf', bbox_inches='tight')

plt.show()


#%% same but for any transport type
transport = 'at'

with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Load metrics
metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0).to_numpy()

# Scatter plot configuration
names = [theta, bulk, dispersivity, decay, alpha, kd]  # Use actual arrays
titles = [r'$\theta$', r'$\rho_b$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$', r'$K_d$']

colors = sns.color_palette('YlOrRd')
positions = np.arange(0, 4, 1)  # Only 4 bars, so positions should match

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1, 1], height_ratios=[1, 1], wspace=0.4, hspace=0.4)

# Bar Chart (Left Panel)
ax0 = fig.add_subplot(gs[:, 0])  # Takes up both rows

ax0.bar(positions[0], SIs_dict[transport]['total_Si_peak']['ST']['kd'], color=colors[0])
ax0.errorbar(positions[0], SIs_dict[transport]['total_Si_peak']['ST']['kd'], 
             yerr=SIs_dict[transport]['total_Si_peak']['ST_conf']['kd'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[1], SIs_dict[transport]['total_Si_peak']['ST']['dispersivity'], color=colors[1])
ax0.errorbar(positions[1], SIs_dict[transport]['total_Si_peak']['ST']['dispersivity'], 
             yerr=SIs_dict[transport]['total_Si_peak']['ST_conf']['dispersivity'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[2], SIs_dict[transport]['total_Si_peak']['ST']['lamb'], color=colors[2])
ax0.errorbar(positions[2], SIs_dict[transport]['total_Si_peak']['ST']['lamb'], 
             yerr=SIs_dict[transport]['total_Si_peak']['ST_conf']['lamb'], fmt='none', ecolor='black', capsize=2)

ax0.bar(positions[3], SIs_dict[transport]['total_Si_peak']['ST']['alpha'], color=colors[3])
ax0.errorbar(positions[3], SIs_dict[transport]['total_Si_peak']['ST']['alpha'], 
             yerr=SIs_dict[transport]['total_Si_peak']['ST_conf']['alpha'], fmt='none', ecolor='black', capsize=2)

ax0.set_xticks(positions)
ax0.set_xticklabels(['$K_d$', r'$\alpha_i$', r'$\lambda$', r'$\alpha$'])
ax0.set_ylabel('Total Order SI')
ax0.set_title('a.', fontweight='bold')

# Scatter Plots (Right Panel, 2x2 Grid)
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])

ax1.scatter(kd, metrics[:, 1], color=colors[0])
ax1.set_xscale('log')
#ax1.set_yscale('log')
ax1.set_title('b.', fontweight='bold')
ax1.set_xlabel('$K_d$')
ax1.set_ylabel('Time to peak C')

ax2.scatter(dispersivity, metrics[:, 1], color=colors[1])
ax2.set_xscale('log')
ax2.set_title('c.', fontweight='bold')
ax2.set_xlabel(r'$\alpha_i$')
ax2.set_ylabel('Time to peak C')

ax3.scatter(decay, metrics[:, 1], color=colors[2])
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel('Time to peak C')
ax3.set_xscale('log')
ax3.set_title('d.', fontweight='bold')

ax4.scatter(alpha, metrics[:, 1], color=colors[3])
ax4.set_xlabel(r'$\alpha$')
ax4.set_ylabel('Time to peak C')
ax4.set_xscale('log')
ax4.set_title('e.', fontweight='bold')

# Add overall title
#fig.suptitle('Parameters influencing time to peak concentration - advective reaction', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{transport}_most_important_factors.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%%
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from sklearn.metrics import r2_score, mean_squared_error

# # Extract data
# x = kd
# y = metrics[:, 1]  # Time to peak C

# # Define models
# def linear(x, a, b):
#     return a * x + b

# def quadratic(x, a, b, c):
#     return a * x**2 + b * x + c

# def cubic(x, a, b, c, d):
#     return a * x**3 + b * x**2 + c * x + d

# def inverse(x, a, b):
#     return a / x + b

# def log_log(x, a, b):
#     return np.exp(a) * x**b  # Equivalent to log(y) = a + b log(x)

# def logistic(x, a, b, c):
#     return a / (1 + np.exp(-b * (x - c)))

# def power_law(x, a, b):
#     return a * np.power(x, b)

# def exponential(x, a, b):
#     return a * np.exp(b * x)

# def logarithmic(x, a, b):
#     return a * np.log(x) + b

# # Fit models
# popt_linear, _ = curve_fit(linear, x, y)
# popt_quad, _ = curve_fit(quadratic, x, y)
# popt_cubic, _ = curve_fit(cubic, x, y)
# popt_inverse, _ = curve_fit(inverse, x, y)
# popt_log_log, _ = curve_fit(log_log, x, y)
# popt_logistic, _ = curve_fit(logistic, x, y)
# popt_power, _ = curve_fit(power_law, x, y)
# popt_exp, _ = curve_fit(exponential, x, y)
# popt_log, _ = curve_fit(logarithmic, x, y)
# # Generate model predictions
# y_pred_linear = linear(x, *popt_linear)
# y_pred_quad = quadratic(x, *popt_quad)
# y_pred_cubic = cubic(x, *popt_cubic)
# y_pred_inverse = inverse(x, *popt_inverse)
# y_pred_log_log = log_log(x, *popt_log_log)
# y_pred_logistic = logistic(x, *popt_logistic)
# y_pred_power = power_law(x, *popt_power)
# y_pred_exp = exponential(x, *popt_exp)
# y_pred_log = logarithmic(x, *popt_log)

# # Compute goodness-of-fit metrics
# def compute_metrics(y_true, y_pred, num_params):
#     r2 = r2_score(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     n = len(y_true)
#     aic = n * np.log(mean_squared_error(y_true, y_pred)) + 2 * num_params
#     return r2, rmse, aic

# models = {
#     "Linear": (y_pred_linear, popt_linear, 2),
#     "Quadratic": (y_pred_quad, popt_quad, 3),
#     "Cubic": (y_pred_cubic, popt_cubic, 4),
#     "Inverse": (y_pred_inverse, popt_inverse, 2),
#     "Log-Log": (y_pred_log_log, popt_log_log, 2),
#     "Logistic": (y_pred_logistic, popt_logistic, 3),
#     "Power": (y_pred_power, popt_power, 2),
#     "Exponential": (y_pred_exp, popt_exp, 2),
#     "Logarithmic": (y_pred_log, popt_log, 2),

# }

# print("\nGoodness of Fit Metrics:")
# for name, (y_pred, params, num_params) in models.items():
#     r2, rmse, aic = compute_metrics(y, y_pred, num_params)
#     print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}, AIC = {aic:.3f}")

# # Plot results
# x_fit = np.linspace(min(x), max(x), 100)
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, label="Data", color="black", alpha=0.5)
# plt.plot(x_fit, linear(x_fit, *popt_linear), label="Linear", linestyle="--")
# plt.plot(x_fit, quadratic(x_fit, *popt_quad), label="Quadratic", linestyle="-.")
# plt.plot(x_fit, cubic(x_fit, *popt_cubic), label="Cubic", linestyle=":")
# plt.plot(x_fit, inverse(x_fit, *popt_inverse), label="Inverse", linestyle="dotted")
# plt.plot(x_fit, log_log(x_fit, *popt_log_log), label="Log-Log", linestyle="dashdot")
# plt.plot(x_fit, logistic(x_fit, *popt_logistic), label="Logistic", linestyle="solid")
# plt.plot(x_fit, power_law(x_fit, *popt_power), label="Power Law", linestyle="solid", c='red')
# plt.plot(x_fit, exponential(x_fit, *popt_exp), label="Exponential", linestyle="solid", c='pink')
# plt.plot(x_fit, logarithmic(x_fit, *popt_log), label="Logarithmic", linestyle="solid", c='orange')

# plt.ylim(0,1000)
# plt.xscale("log")
# plt.xlabel("$K_d$")
# plt.ylabel("Time to peak C")
# plt.legend()
# plt.title("Model Fits for Time to Peak C vs. $K_d$")
# plt.show()


