#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 11:42:03 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
plt.style.use('ggplot')
import json
from scipy.stats import binned_statistic_2d
import itertools
#%%
# Load data
# transport = 'dt'
# with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
#     btc_data = json.load(f)

# # Extract parameters into NumPy arrays
# theta = np.array([entry['params'][0] for entry in btc_data])
# bulk = np.array([entry['params'][1] for entry in btc_data])
# dispersivity = np.array([entry['params'][2] for entry in btc_data])
# decay = np.array([entry['params'][3] for entry in btc_data])
# alpha = np.array([entry['params'][4] for entry in btc_data])
# kd = np.array([entry['params'][5] for entry in btc_data])

# # Store parameters in a dictionary for easy looping
# param_dict = {
#     'Theta': theta,
#     'Bulk': bulk,
#     'Dispersivity': dispersivity,
#     'Decay': decay,
#     'Alpha': alpha,
#     'Kd': kd
# }

# # Get all unique pairs of parameters
# param_pairs = list(itertools.combinations(param_dict.keys(), 2))

# metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0).to_numpy()
# output_metric = metrics[:,2] # late time tailing
# # Define number of bins
# num_bins = 30

# # Create subplots
# n_pairs = len(param_pairs)
# n_cols = 4  # Number of columns in the grid
# n_rows = int(np.ceil(n_pairs / n_cols))  # Number of rows needed
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))

# # Flatten axes array if there's more than one row
# axes = axes.flatten()

# # Loop through parameter pairs and plot heatmaps
# heatmap_obj = None  # Store the last heatmap for colorbar reference

# for i, (param_x, param_y) in enumerate(param_pairs):
#     ax = axes[i]
    
#     log_x_param = np.log10(param_dict[param_x])
#     log_y_param = np.log10(param_dict[param_y])
    
#     # Compute binned statistics
#     stat, log_x_edges, log_y_edges, binnumber = binned_statistic_2d(
#         log_x_param, log_y_param, output_metric, statistic='std', bins=num_bins
#     )
    
#     variance = np.nan_to_num(stat, nan=0) ** 2  # Convert NaNs to 0
#     X, Y = np.meshgrid(log_x_edges[:-1], log_y_edges[:-1])

#     # Plot the heatmap without individual colorbars
#     heatmap_obj = sns.heatmap(variance.T, cmap="viridis", cbar=False, vmin=0, vmax=400000, ax=ax)

#     # Set log-scale tick labels
#     x_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
#     y_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
    
#     x_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_x_edges[x_tick_positions]]
#     y_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_y_edges[y_tick_positions]]
    
#     ax.set_xticks(x_tick_positions)
#     ax.set_xticklabels(x_tick_labels)
#     ax.set_yticks(y_tick_positions)
#     ax.set_yticklabels(y_tick_labels)

#     ax.set_xlabel(param_x, fontweight='bold')
#     ax.set_ylabel(param_y, fontweight='bold')

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# # Add a single colorbar to the right of the figure
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
# fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax, label="Variance")

# # Adjust layout and show the plot
# fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)

# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
# plt.show()

#%% pick four second order interactions to highlight: alpha-kd, dispersivity-kd, 
# num_bins = 30

# def plot_heatmap(x_param, y_param, output_metric, ax, x_label, y_label, title):
    
#     log_x_param = np.log10(x_param)
#     log_y_param = np.log10(y_param)

#     # Compute binned statistics
#     stat, log_x_edges, log_y_edges, binnumber = binned_statistic_2d(
#         log_x_param, log_y_param, output_metric, statistic='std', bins=num_bins
#     )
      
#     variance = np.nan_to_num(stat, nan=0) ** 2  # Convert NaNs to 0
#     X, Y = np.meshgrid(log_x_edges[:-1], log_y_edges[:-1])

#     # Plot the heatmap without individual colorbars
#     heatmap_obj = sns.heatmap(np.log10(variance.T), cmap="viridis", cbar=False, ax=ax)

#     # Set log-scale tick labels
#     x_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
#     y_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
    
#     x_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_x_edges[x_tick_positions]]
#     y_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_y_edges[y_tick_positions]]
    
#     ax.set_xticks(x_tick_positions)
#     ax.set_xticklabels(x_tick_labels)
#     ax.set_yticks(y_tick_positions)
#     ax.set_yticklabels(y_tick_labels)

#     ax.set_xlabel(x_label, fontweight='bold', fontsize=16)
#     ax.set_ylabel(y_label, fontweight='bold', fontsize=16)
#     ax.invert_yaxis()
#     ax.set_title(title, fontweight='bold', fontsize=16)

#     return ax, variance

# # Load data
# transport = 'dt'
# with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
#     btc_data = json.load(f)

# # Extract parameters into NumPy arrays
# theta = np.array([entry['params'][0] for entry in btc_data])
# bulk = np.array([entry['params'][1] for entry in btc_data])
# dispersivity = np.array([entry['params'][2] for entry in btc_data])
# decay = np.array([entry['params'][3] for entry in btc_data])
# alpha = np.array([entry['params'][4] for entry in btc_data])
# kd = np.array([entry['params'][5] for entry in btc_data])

# # Store parameters in a dictionary for easy looping
# param_dict = {
#     'Theta': theta,
#     'Bulk': bulk,
#     'Dispersivity': dispersivity,
#     'Decay': decay,
#     'Alpha': alpha,
#     'Kd': kd
# }

# metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0).to_numpy()
# output_metric = metrics[:,2] # late time tailing

# fig, axes = plt.subplots(1,4,figsize=(12,3))

# axes = axes.flatten()

# plot_heatmap(alpha, kd, output_metric, axes[0], r'$\alpha$', r'$K_d$', 'a.')
# plot_heatmap(dispersivity, kd, output_metric, axes[1], r'$\alpha_i$', r'$K_d$', 'b.')
# plot_heatmap(bulk, kd, output_metric, axes[2], r'$\rho_b$', r'$K_d$', 'c.')
# plot_heatmap(bulk, decay, output_metric, axes[3], r'$\lambda$', r'$\rho_b$', 'd.')

# cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
# fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax, label="Log Variance")
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

# for ax in axes:
#     ax.set_rasterized(True)
# # Adjust layout and show the plot
# #fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)
# #plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{transport}_second_order_interactions.pdf', format='pdf', bbox_inches='tight')
# plt.show()

#%% all combos with greek letter labels

# # Load data
# transport = 'dt'
# with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
#     btc_data = json.load(f)

# # Extract parameters into NumPy arrays
# theta = np.array([entry['params'][0] for entry in btc_data])
# bulk = np.array([entry['params'][1] for entry in btc_data])
# dispersivity = np.array([entry['params'][2] for entry in btc_data])
# decay = np.array([entry['params'][3] for entry in btc_data])
# alpha = np.array([entry['params'][4] for entry in btc_data])
# kd = np.array([entry['params'][5] for entry in btc_data])

# # Store parameters in a dictionary for easy looping
# param_dict = {
#     r'$\theta$': theta,
#     r'$\rho_b$': bulk,
#     r'$\alpha_i$': dispersivity,
#     r'$\lambda$': decay,
#     r'$\alpha$': alpha,
#     r'$K_d$': kd
# }

# # Get all unique pairs of parameters
# param_pairs = list(itertools.combinations(param_dict.keys(), 2))

# metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0).to_numpy()
# output_metric = metrics[:,2] # late time tailing
# # Define number of bins
# num_bins = 30

# # Create subplots
# n_pairs = len(param_pairs)
# n_cols = 4  # Number of columns in the grid
# n_rows = int(np.ceil(n_pairs / n_cols))  # Number of rows needed
# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 16))

# # Flatten axes array if there's more than one row
# axes = axes.flatten()

# # Loop through parameter pairs and plot heatmaps
# heatmap_obj = None  # Store the last heatmap for colorbar reference

# for i, (param_x, param_y) in enumerate(param_pairs):
#     ax = axes[i]
    
#     log_x_param = np.log10(param_dict[param_x])
#     log_y_param = np.log10(param_dict[param_y])
    
#     # Compute binned statistics
#     stat, log_x_edges, log_y_edges, binnumber = binned_statistic_2d(
#         log_x_param, log_y_param, output_metric, statistic='std', bins=num_bins
#     )
    
#     variance = np.nan_to_num(stat, nan=0) ** 2  # Convert NaNs to 0
#     X, Y = np.meshgrid(log_x_edges[:-1], log_y_edges[:-1])

#     # Plot the heatmap without individual colorbars
#     #heatmap_obj = sns.heatmap(variance.T, cmap="viridis", cbar=False, vmin=0, vmax=400000, ax=ax)
#     heatmap_obj = sns.heatmap(np.log10(variance.T), cmap="viridis", cbar=False, ax=ax)

#     # Set log-scale tick labels
#     x_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
#     y_tick_positions = np.linspace(0, num_bins - 1, num=5, dtype=int)
    
#     x_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_x_edges[x_tick_positions]]
#     y_tick_labels = [f"$10^{{{tick:.0f}}}$" for tick in log_y_edges[y_tick_positions]]
    
#     ax.set_xticks(x_tick_positions)
#     ax.set_xticklabels(x_tick_labels)
#     ax.set_yticks(y_tick_positions)
#     ax.set_yticklabels(y_tick_labels)

#     ax.set_xlabel(param_x, fontweight='bold', fontsize=16)
#     ax.set_ylabel(param_y, fontweight='bold', fontsize=16)

#     ax.invert_yaxis()

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# # Add a single colorbar to the right of the figure
# cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
# fig.colorbar(heatmap_obj.collections[0], cax=cbar_ax, label="Log Variance")

# # Adjust layout and show the plot
# #fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)

# plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

# for ax in axes:
#     ax.set_rasterized(True)
# # Adjust layout and show the plot
# #plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{transport}_all_second_order_interactions.pdf', format='pdf', bbox_inches='tight')

# plt.show()


#%%
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

# Load data
transport = 'ar'
with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/btc_data_{transport}.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Store parameters in a dictionary for easy looping
param_dict = {
    'Theta': theta,
    'Bulk': bulk,
    'Dispersivity': dispersivity,
    'Decay': decay,
    'Alpha': alpha,
    'Kd': kd
}
time = 'peak'
if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2
metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/metrics_{transport}.csv', index_col=0).to_numpy()
output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(2,3,figsize=(10,6))

axes = axes.flatten()

sc= axes[0].scatter(alpha, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(decay, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[2].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

axes[3].scatter(alpha, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[3].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[3].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=14, loc='left')

axes[4].scatter(alpha, decay, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[4].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[4].set_ylabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[4].set_xscale('log')
axes[4].set_yscale('log')
axes[4].set_title('(e)', fontweight='bold', fontsize=14, loc='left')

axes[5].scatter(dispersivity, decay, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[5].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[5].set_ylabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[5].set_xscale('log')
axes[5].set_yscale('log')
axes[5].set_title('(f)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label="Time to peak concentration (log)")
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)
# Adjust layout and show the plot
#fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/time_{time}_{transport}_second_order_interactions.pdf', format='pdf', bbox_inches='tight')
plt.show()

#%% second order interaction plot but for the full range analysis

greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

# Load data
with open('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/btc_data_model106.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Store parameters in a dictionary for easy looping
param_dict = {
    'Theta': theta,
    'Bulk': bulk,
    'Dispersivity': dispersivity,
    'Decay': decay,
    'Alpha': alpha,
    'Kd': kd
}
time = 'early'
if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2
metrics = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/metrics_106.csv', index_col=0).to_numpy()
output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(2,3,figsize=(10,6))

axes = axes.flatten()

sc= axes[0].scatter(alpha, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(decay, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[2].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

axes[3].scatter(alpha, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[3].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[3].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=14, loc='left')

axes[4].scatter(alpha, decay, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[4].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[4].set_ylabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[4].set_xscale('log')
axes[4].set_yscale('log')
axes[4].set_title('(e)', fontweight='bold', fontsize=14, loc='left')

axes[5].scatter(dispersivity, decay, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[5].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[5].set_ylabel(r'$\lambda$', fontweight='bold', fontsize=12)
axes[5].set_xscale('log')
axes[5].set_yscale('log')
axes[5].set_title('(f)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label="Time to peak concentration (log)")
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)
# Adjust layout and show the plot
#fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/time_{time}_model_106_second_order_interactions.pdf', format='pdf', bbox_inches='tight')
plt.show()



#%% second order interaction plot but for the full range analysis, updated important parameters
plt.style.use('default')
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

# Load data
with open('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/btc_data_model106.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

# Store parameters in a dictionary for easy looping
param_dict = {
    'Theta': theta,
    'Bulk': bulk,
    'Dispersivity': dispersivity,
    'Decay': decay,
    'Alpha': alpha,
    'Kd': kd
}
time = 'early'
if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2
metrics = pd.read_csv('/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results_larger_sample/metrics_106.csv', index_col=0).to_numpy()
output_metric = metrics[:,time_idx] 

fig, axes = plt.subplots(2,3,figsize=(10,6))

axes = axes.flatten()

sc= axes[0].scatter(dispersivity, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[0].set_xlabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[0].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_title('(a)', fontweight='bold', fontsize=14, loc='left')

axes[1].scatter(bulk, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[1].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=12)
axes[1].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
#axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=14, loc='left')

axes[2].scatter(alpha, kd, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[2].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[2].set_ylabel(r'$K_d$', fontweight='bold', fontsize=12)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=14, loc='left')

axes[3].scatter(bulk, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[3].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=12)
axes[3].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
#axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=14, loc='left')

axes[4].scatter(alpha, dispersivity, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[4].set_xlabel(r'$\alpha$', fontweight='bold', fontsize=12)
axes[4].set_ylabel(r'$\alpha_i$', fontweight='bold', fontsize=12)
axes[4].set_xscale('log')
axes[4].set_yscale('log')
axes[4].set_title('(e)', fontweight='bold', fontsize=14, loc='left')

axes[5].scatter(bulk, alpha, c=np.log10(output_metric), cmap='viridis', s=3, vmin=-1, vmax=2.8)
axes[5].set_xlabel(r'$\rho_b$', fontweight='bold', fontsize=12)
axes[5].set_ylabel(r'$\alpha$', fontweight='bold', fontsize=12)
#axes[5].set_xscale('log')
axes[5].set_yscale('log')
axes[5].set_title('(f)', fontweight='bold', fontsize=14, loc='left')

cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
fig.colorbar(sc, cmap='viridis', cax=cbar_ax, label="Early arrival (log)")
plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar

for ax in axes:
    ax.set_rasterized(True)
# Adjust layout and show the plot
#fig.suptitle(f'All pairwise interaction plots - {transport} scenario', fontweight='bold', fontsize=20)
plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/time_{time}_model_106_second_order_interactions.pdf', format='pdf', bbox_inches='tight')
plt.show()






















