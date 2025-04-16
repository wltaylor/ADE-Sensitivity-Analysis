#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:46:00 2025

@author: williamtaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')

# import data
transports = ['dr','ar','dt','at']

#plt.style.use('default')
time = 'late'

scenario = 'dr'

if time == 'early':
    time_idx = 0
if time == 'peak':
    time_idx = 1
if time == 'late':
    time_idx = 2

metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{scenario}.csv', index_col=0).to_numpy()

with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{scenario}.json', 'r') as f:
    btc_data = json.load(f)

# Extract parameters into NumPy arrays
theta = np.array([entry['params'][0] for entry in btc_data])
bulk = np.array([entry['params'][1] for entry in btc_data])
dispersivity = np.array([entry['params'][2] for entry in btc_data])
decay = np.array([entry['params'][3] for entry in btc_data])
alpha = np.array([entry['params'][4] for entry in btc_data])
kd = np.array([entry['params'][5] for entry in btc_data])

fig, axes = plt.subplots(1,4, figsize=(12,4), sharey=True)
axes = axes.flatten()
greek_labels = [r'$\theta$', r'$\rho_b$', r'$\alpha$', r'$\lambda$', r'$\alpha_i$', r'$K_d$']

axes[0].scatter(kd, metrics[:,time_idx])
axes[0].set_xscale('log')
axes[0].set_ylabel('Time to Peak Concentration', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$K_d$', fontsize=16)
axes[0].set_title('(a)', fontweight='bold', fontsize=16, loc='left')

axes[1].scatter(dispersivity, metrics[:,time_idx])
axes[1].set_xscale('log')
axes[1].set_title('(b)', fontweight='bold', fontsize=16, loc='left')
axes[1].set_xlabel(r'$\alpha_i$', fontsize=16)

axes[2].scatter(alpha, metrics[:,time_idx])
axes[2].set_xscale('log')
axes[2].set_title('(c)', fontweight='bold', fontsize=16, loc='left')
axes[2].set_xlabel(r'$\alpha$', fontsize=16)

axes[3].scatter(decay, metrics[:,time_idx])
axes[3].set_xscale('log')
axes[3].set_title('(d)', fontweight='bold', fontsize=16, loc='left')
axes[3].set_xlabel(r'$\lambda$', fontsize=16)

plt.tight_layout()
#plt.savefig(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/figures/{time}_scenario_{scenario}_scatter.pdf', format='pdf', bbox_inches='tight')
plt.show()


#%% compare all the dispersivity scatter plots
from scipy.stats import linregress

fig, axes = plt.subplots(2,2,figsize=(8,8))
axes=axes.flatten()
for i,transport in enumerate(transports):
    # read in metrics
    metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0).to_numpy()
    print(f'Transport scenario {transport} variance: {metrics.var()}')
    # read in btc data
    with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
        btc_data = json.load(f)
        
    # pull out dispersivity
    dispersivity = np.array([entry['params'][2] for entry in btc_data])
    y = metrics[:,2]

    axes[i].scatter(dispersivity, y)
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[i].set_ylim(10**-2, 10**4)
    axes[i].set_ylabel('Late time tailing')
    axes[i].set_xlabel(r'$\alpha_i$')
    axes[i].set_title(transport, fontweight='bold')

    # log transform x for linear regression
    log_x = np.log10(dispersivity)
    slope, intercept, r_value, p_value, _ = linregress(log_x, y)
    trend_x = np.linspace(log_x.min(), log_x.max(), 100)
    trend_y = slope * trend_x + intercept
    
    axes[i].plot(10**trend_x, trend_y, color='black', label='Trend')
    axes[i].text(0.05, 0.95, f'$r$ = {r_value:.2f}\n$p$ = {p_value:.2g}', 
                 transform=axes[i].transAxes, verticalalignment='top',
                 fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))


plt.tight_layout()
plt.show()

#%% deep dive into dispersivity difference

from scipy.stats import spearmanr

transport = 'ar'

metrics = pd.read_csv(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/metrics_{transport}.csv', index_col=0)

with open(f'/Users/williamtaylor/Documents/Github/ADE-Sensitivity-Analysis/results/btc_data_{transport}.json', 'r') as f:
    btc_data = json.load(f)

param_names = ['Kd', 'alpha_i', 'lambda', 'alpha', 'rho_b', 'theta']
params = pd.DataFrame([entry['params'] for entry in btc_data], columns=param_names)

# Add output metric (late-time tailing)
params['late_tailing'] = metrics.to_numpy()[:, 2]  # assuming it's column index 2

# Compute Spearman correlation matrix
corr = params.corr(method='spearman')

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title(f"Spearman Correlation Matrix - {transport}")
plt.tight_layout()
plt.show()


#%%

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import numpy as np

# # Normalize input parameters
# X = StandardScaler().fit_transform(params[param_names])
# y = params['late_tailing'].values

# # Run PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot PCA with output colored
# plt.figure(figsize=(7, 6))
# scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
# plt.colorbar(scatter, label='Late-time tailing')
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
# plt.title(f"PCA of Input Parameters - {transport}")
# plt.tight_layout()
# plt.show()


#%%
plt.style.use('ggplot')
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data
X = StandardScaler().fit_transform(params[param_names])
y = params['late_tailing'].values

# Run PCA
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X)

# Get the PCA components
components = pca.components_  # shape (2, n_features)

# Project the unit vector of alpha_i onto PCA space
alpha_i_index = param_names.index('alpha_i')
unit_vector = np.zeros(len(param_names))
unit_vector[alpha_i_index] = 1

# Projection of alpha_i direction into PCA space
alpha_i_proj = components @ unit_vector  # shape (2,)

# Scale the arrow for plotting
arrow_scale = 3  # adjust to fit well in the plot
arrow = alpha_i_proj * arrow_scale

# Plot
plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=y)
plt.colorbar(scatter, label='Late-time tailing')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title(f"PCA of Input Parameters - {transport}")

# Add arrow for alpha_i
plt.arrow(0, 0, arrow[0], arrow[1], color='red', width=0.05, head_width=0.2, length_includes_head=True)
plt.text(arrow[0]*1.1, arrow[1]*1.1, r'$\alpha_i$', color='red', fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.show()

#%% scree plot

plt.figure(figsize=(6, 4))
n_components = len(pca.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.xticks(range(1, n_components + 1))
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example Sobol indices matrix (rows = metrics, columns = parameters)
sobol_matrix = pd.DataFrame({
    'theta': [0.10, 0.09, 0.11],
    'rho_b': [0.21, 0.20, 0.20],
    'dispersivity': [0.6, 0.65, 0.50],
    'lamb': [0.15, 0.19, 0.38],
    'alpha': [0.21, 0.17, 0.30],
    'kd': [0.85, 0.87, 0.90],
}, index=['early', 'mid', 'late'])

# Standardize across metrics (rows)
X = StandardScaler().fit_transform(sobol_matrix)

# Run PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Scree plot
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, 'o-', color='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot (PCA on Sobol Indices)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Biplot: rows = metrics, arrows = parameter loadings
plt.figure(figsize=(7, 6))
for i, txt in enumerate(sobol_matrix.index):
    plt.scatter(X_pca[i, 0], X_pca[i, 1], color='black')
    plt.text(X_pca[i, 0]+0.05, X_pca[i, 1], txt, fontsize=12)

# Add arrows for each parameter
loadings = pca.components_.T  # shape (n_params, n_PCs)
for i, param in enumerate(sobol_matrix.columns):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='red', head_width=0.03, length_includes_head=True)
    plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, param, color='red', fontsize=12)

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title('PCA Biplot of Sobol Indices')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.tight_layout()
plt.show()



